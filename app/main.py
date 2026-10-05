import asyncio,base64,binascii,hashlib,io,json,logging,time
from contextlib import asynccontextmanager
from typing import Literal
from uuid import UUID,uuid4
import cv2,httpx,numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError
from fastapi import FastAPI,Depends,HTTPException,UploadFile,File,Form,Request,WebSocket,WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from pydantic import BaseModel,Field,ConfigDict,field_validator
from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from .config import Settings
from .supabase import SupabaseGateway
from .engine import Engine,MODEL_IDS,quality,rank_candidates
from .media import DetectionResult,LiveDetectionResult,VideoDetectionResult,detect_frame,detect_video,resize_frame
from .live import LiveSessions
from redis.exceptions import RedisError
from ultralytics.utils.patches import _image_open as pillow_open

settings=Settings();log=logging.getLogger(__name__)
bearer=HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(timeout=20) as client:
        app.state.gateway=SupabaseGateway(settings,client)
        app.state.engine=Engine(settings.models)
        app.state.thresholds=app.state.engine.bundle['thresholds']
        app.state.model_stamp=(settings.models/'pipeline.pkl').stat().st_mtime_ns
        app.state.model_reload_lock=asyncio.Lock()
        app.state.inference_slots=asyncio.BoundedSemaphore(settings.inference_concurrency)
        app.state.live_sessions=LiveSessions(settings.redis_url,settings.live_interval_ms)
        if not await app.state.live_sessions.available():log.warning('Redis is unavailable; live detection is disabled until it reconnects.')
        try:yield
        finally:await app.state.live_sessions.close()

app=FastAPI(title='ResQ face evidence service',version='2.1.0',lifespan=lifespan)
app.add_middleware(RequestBodyLimitMiddleware,max_body_size=max(settings.max_video_bytes,settings.max_bytes)+1048576)
app.add_middleware(CORSMiddleware,allow_origins=settings.cors,allow_methods=['GET','POST','PATCH'],allow_headers=['Authorization','Content-Type'])

def gateway(connection:HTTPConnection):
    gw = getattr(connection.app.state, 'gateway', None)
    if gw is None:
        client = httpx.AsyncClient(timeout=20)
        gw = SupabaseGateway(settings, client)
        connection.app.state.gateway = gw
    return gw

async def model_engine(request:Request):
    """Load a complete verified release atomically; each request keeps one snapshot."""
    state=request.app.state
    try:
        stamp=(settings.models/'pipeline.pkl').stat().st_mtime_ns
        current_stamp = getattr(state, 'model_stamp', None)
        if stamp!=current_stamp or not hasattr(state, 'engine'):
            lock = getattr(state, 'model_reload_lock', None)
            if lock is None:
                lock = asyncio.Lock()
                state.model_reload_lock = lock
            async with lock:
                current_stamp = getattr(state, 'model_stamp', None)
                if stamp!=current_stamp or not hasattr(state, 'engine'):
                    fresh=await run_in_threadpool(Engine,settings.models)
                    state.engine=fresh
                    state.thresholds=fresh.bundle['thresholds']
                    state.model_stamp=stamp
        return state.engine
    except (OSError,ValueError,KeyError,RuntimeError,cv2.error):
        log.exception('Model release could not be loaded')
        raise HTTPException(503,'The current model release failed validation. Ask the server administrator to restore the model files.') from None

async def run_inference(request,function,*args):
    slots = getattr(request.app.state, 'inference_slots', None)
    if slots is None:
        slots = asyncio.BoundedSemaphore(settings.inference_concurrency)
        request.app.state.inference_slots = slots
    try:await asyncio.wait_for(slots.acquire(),timeout=settings.inference_queue_seconds)
    except TimeoutError:raise HTTPException(503,'The model service is busy. Try again shortly.',headers={'Retry-After':'3'}) from None
    try:return await run_in_threadpool(function,*args)
    finally:slots.release()

async def actor(credentials:HTTPAuthorizationCredentials|None=Depends(bearer),gw=Depends(gateway)):
    if not credentials or credentials.scheme.lower()!='bearer':raise HTTPException(401,'Sign in to continue.')
    return await gw.authenticate(credentials.credentials)

def roles(*allowed):
    async def check(user=Depends(actor)):
        if user['role'] not in allowed:raise HTTPException(403,'Your account does not have permission for this action.')
        return user
    return check

member=roles('rescuer','verifier','admin');rescuer=roles('rescuer','admin');verifier=roles('verifier','admin');admin=roles('admin')

class CaseInput(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    name:str=Field(min_length=1,max_length=120)
    age:int|None=Field(default=None,ge=0,le=120)
    last_seen:str=Field(default='',max_length=240)
    notes:str=Field(default='',max_length=2000)
    consent_basis:Literal['self','family','authority','research']
    status:Literal['pending','urgent','ongoing','completed','closed']='pending'
    @field_validator('name')
    @classmethod
    def nonempty(cls,v):
        if not v.strip():raise ValueError('Name is required')
        return v

class CaseUpdateInput(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    status:Literal['pending','urgent','ongoing','completed','closed']

class ReviewInput(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    decision:Literal['verified','rejected']
    reason:str=Field(min_length=3,max_length=2000)

class RoleInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    role:Literal['pending','rescuer','verifier']

class ProfileInput(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    display_name:str=Field(min_length=1,max_length=120)
    avatar_key:Literal['scout','tech','pilot','medic','ranger','lead','cadet','analyst','diver','spark','short-hair','curly-hair','tied-hair']='scout'
    avatar_color:Literal['violet','cyan','blue','amber','emerald','coral','rose','indigo','lime','silver']='violet'
    avatar_shape:Literal['squircle','circle','hexagon','shield','diamond','octagon','rounded-square','pentagon','star']='squircle'
    avatar_style:Literal['3d','2d']='3d'
    @field_validator('display_name')
    @classmethod
    def valid_name(cls,value):
        if any(ord(char)<32 or ord(char)==127 for char in value):raise ValueError('Use a name without control characters')
        return value

def decode_image(data,limits):
    try:
        # The pinned YOLO package patches Image.open to auto-install HEIF decoders
        # on bad input. Use its retained original Pillow decoder for our strict
        # JPEG/PNG/WebP API so malformed uploads never initiate dependency installs.
        with pillow_open(io.BytesIO(data),formats=['JPEG','PNG','WEBP']) as im:
            if im.width*im.height>limits.max_pixels:raise HTTPException(413,'Image dimensions exceed the allowed limit.')
            if im.format not in ('JPEG','PNG','WEBP'):raise HTTPException(415,'Choose a JPEG, PNG or WebP image.')
            im.load();converted=ImageOps.exif_transpose(im).convert('RGB');buffer=io.BytesIO();converted.save(buffer,format='JPEG',quality=92)
            image=cv2.cvtColor(np.asarray(converted),cv2.COLOR_RGB2BGR)
            clean=buffer.getvalue()
    except HTTPException:raise
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError):raise HTTPException(415,'The image could not be decoded.') from None
    return image,clean

def describe(data,modality,cropped,engine,limits,face_index=None):
    image,clean=decode_image(data,limits)
    try:
        if modality=='face':vector,q=engine.extract_face(image,face_index)
        else:vector,q=engine.describe(image,modality,cropped),quality(image)
    except ValueError as error:raise HTTPException(422,str(error)) from None
    except cv2.error:raise HTTPException(422,'The selected face could not be aligned. Use a clearer photograph.') from None
    return vector,q,clean

async def read_image(upload):
    data=await upload.read(settings.max_bytes+1)
    if len(data)>settings.max_bytes:raise HTTPException(413,f'Choose an image smaller than {settings.max_bytes//(1024*1024)} MB.')
    if not data:raise HTTPException(422,'Choose an image first.')
    return data

@app.get('/', tags=['System'])
async def root():
    return {
        'service': 'ResQ face evidence service',
        'status': 'online',
        'version': app.version,
        'health': '/health',
        'docs': '/docs'
    }

@app.get('/health')
async def health(request:Request,engine=Depends(model_engine)):
    bundle=engine.bundle
    sessions = getattr(request.app.state, 'live_sessions', None)
    if sessions is None:
        sessions = LiveSessions(settings.redis_url, settings.live_interval_ms)
        request.app.state.live_sessions = sessions
    redis_connected=await sessions.available()
    return {'service':'ResQ model API','supabase_configured':settings.configured,'models':MODEL_IDS,
            'detector':bundle['detector']['architecture'],'detector_sha256':bundle['detector']['sha256'],
            'local_training':bundle['detector']['local_training'],'ready':True,
            'ear_recognition_enabled':False,'disaster_validated':bundle['deployment']['disaster_validated'],
            'calibration':bundle['calibration']['status'],'scope':'human-reviewed candidate assistance',
            'api_version':app.version,'capabilities':['image_detection','video_detection','live_websocket_detection','selected_face_search'],
            'redis_connected':redis_connected,'live_ready':redis_connected,
            'limits':{'image_bytes':settings.max_bytes,'image_pixels':settings.max_pixels,'video_bytes':settings.max_video_bytes,
                      'video_seconds':settings.max_video_seconds,'video_sample_frames':60,'video_returned_frames':12},
            'live_transport':'WebSocket /api/faces/stream; Redis session leases and frame pacing',
            'live_interval_ms':settings.live_interval_ms}

@app.post('/api/faces/detect',response_model=DetectionResult,tags=['Face detection'])
async def detect_faces(request:Request,image:UploadFile=File(...),include_frame:bool=Form(True),user=Depends(member),engine=Depends(model_engine)):
    data=await read_image(image)
    def detect():
        decoded,_=decode_image(data,settings)
        return detect_frame(decoded,engine,include_frame)
    return await run_inference(request,detect)

@app.post('/api/faces/live',response_model=LiveDetectionResult,tags=['Face detection'])
async def detect_live_frame(request:Request,image:UploadFile=File(...),sequence:int=Form(0,ge=0),captured_at_ms:int|None=Form(None,ge=0),user=Depends(member),engine=Depends(model_engine)):
    """Analyze one camera frame. Clients pace requests; face indexes are frame-local."""
    data=await read_image(image)
    def detect():
        decoded,_=decode_image(data,settings)
        return {**detect_frame(resize_frame(decoded),engine), 'sequence':sequence,'captured_at_ms':captured_at_ms}
    return await run_inference(request,detect)

@app.post('/api/faces/video',response_model=VideoDetectionResult,tags=['Face detection'])
async def detect_video_faces(request:Request,video:UploadFile=File(...),interval_seconds:float=Form(1,ge=0.5,le=10),max_frames:int=Form(30,ge=1,le=60),result_limit:int=Form(8,ge=1,le=12),user=Depends(member),engine=Depends(model_engine)):
    """Sample a bounded local upload and return timestamped, displayable face frames."""
    try:
        return await run_inference(request,detect_video,video.file,engine,settings,interval_seconds,max_frames,result_limit)
    finally:
        await video.close()

@app.websocket('/api/faces/stream')
async def stream_faces(socket:WebSocket,gw=Depends(gateway)):
    """Authenticate first, then analyze one binary JPEG or JSON base64 frame at a time."""
    origin=socket.headers.get('origin')
    # React Native Android supplies the endpoint's HTTP(S) origin automatically.
    # Allow that exact origin as well as configured browser origins, never every
    # LAN host or an arbitrary Origin supplied by a cross-origin browser page.
    endpoint_origin=str(socket.url.replace(scheme='https' if socket.url.scheme=='wss' else 'http',path='',query='',fragment=''))
    if origin and origin!=endpoint_origin and origin not in settings.cors:
        await socket.close(code=1008,reason='Origin is not allowed.')
        return
    await socket.accept()
    sessions=socket.app.state.live_sessions
    session_id=str(uuid4());user=None;claimed=False;last_sequence=-1
    async def fail(status,message,sequence=None,retry_after_ms=None):
        payload={'type':'error','status':status,'message':message,'sequence':sequence}
        if retry_after_ms is not None:payload['retry_after_ms']=retry_after_ms
        await socket.send_json(payload)
    try:
        auth_message=await asyncio.wait_for(socket.receive(),timeout=10)
        if auth_message['type']=='websocket.disconnect':return
        auth_text=auth_message.get('text')
        if not isinstance(auth_text,str):raise HTTPException(401,'Send an auth JSON message before sending frames.')
        if len(auth_text)>8192:raise HTTPException(401,'Invalid authentication message.')
        try:authentication=json.loads(auth_text)
        except ValueError:raise HTTPException(401,'Send an auth message before sending frames.') from None
        if not isinstance(authentication,dict) or authentication.get('type')!='auth' or not isinstance(authentication.get('token'),str):
            raise HTTPException(401,'Send an auth message before sending frames.')
        token=authentication['token']
        user=await gw.authenticate(token)
        if user['role'] not in ('rescuer','verifier','admin'):raise HTTPException(403,'Your account does not have permission for live detection.')
        claimed=await sessions.claim(user['id'],session_id)
        if not claimed:
            await fail(409,'A live camera session is already open for this account. Stop it before starting another.')
            await socket.close(code=4409)
            return
        await model_engine(socket)
        await socket.send_json({'type':'ready','session_id':session_id,'min_interval_ms':settings.live_interval_ms,
                                'max_frame_bytes':settings.live_frame_bytes,'session_seconds':settings.live_session_seconds,
                                'transport':'websocket','coordination':'redis'})
        started=time.monotonic();last_auth=started
        while time.monotonic()-started < settings.live_session_seconds:
            message=await asyncio.wait_for(socket.receive(),timeout=30)
            if message['type']=='websocket.disconnect':break
            sequence=last_sequence+1;captured_at=None
            try:
                if message.get('bytes') is not None:
                    data=message['bytes']
                else:
                    text=message.get('text','')
                    if len(text)>settings.live_frame_bytes*4//3+4096:raise HTTPException(413,'The camera frame is too large. Reduce its resolution.')
                    try:body=json.loads(text)
                    except ValueError:raise HTTPException(422,'Send a binary image or a frame JSON message.') from None
                    if not isinstance(body,dict):raise HTTPException(422,'Send a frame JSON object.')
                    if body.get('type')=='stop':break
                    if body.get('type')=='ping':
                        active,_=await sessions.touch(user['id'],session_id)
                        if active<0:raise HTTPException(409,'The live session expired. Reconnect to continue.')
                        await socket.send_json({'type':'pong'})
                        continue
                    if body.get('type')!='frame':raise HTTPException(422,'Unknown live message type.')
                    sequence=body.get('sequence')
                    if type(sequence) is not int or sequence<=last_sequence:raise HTTPException(422,'Frame sequence must increase for each submitted frame.')
                    captured_at=body.get('captured_at_ms')
                    if captured_at is not None and (type(captured_at) is not int or captured_at<0):raise HTTPException(422,'Invalid frame capture timestamp.')
                    encoded=body.get('image_base64')
                    if not isinstance(encoded,str):raise HTTPException(422,'Frame image_base64 is required.')
                    try:data=base64.b64decode(encoded,validate=True)
                    except (ValueError,binascii.Error):raise HTTPException(415,'The camera image is not valid base64.') from None
                if len(data)>settings.live_frame_bytes:raise HTTPException(413,'The camera frame is too large. Reduce its resolution.')
                if not data:raise HTTPException(422,'The camera frame is empty.')
                if time.monotonic()-last_auth >= 60:
                    user=await gw.authenticate(token);last_auth=time.monotonic()
                    if user['role'] not in ('rescuer','verifier','admin'):raise HTTPException(403,'Your live detection access is no longer active.')
                active,retry=await sessions.touch(user['id'],session_id,frame=True)
                if active<0:raise HTTPException(409,'The live session expired. Reconnect to continue.')
                if not active:
                    await fail(429,'Wait for the next camera sample.',sequence,retry)
                    continue
                engine=await model_engine(socket)
                def detect():
                    decoded,_=decode_image(data,settings)
                    return detect_frame(resize_frame(decoded),engine)
                detected=await run_inference(socket,detect)
                last_sequence=sequence
                await socket.send_json({'type':'detection',**LiveDetectionResult(**detected,sequence=sequence,captured_at_ms=captured_at).model_dump()})
            except HTTPException as error:
                await fail(error.status_code,error.detail,sequence)
                if error.status_code in (401,403,409):
                    await socket.close(code=4401 if error.status_code==401 else 4403 if error.status_code==403 else 4409)
                    return
        await socket.close(code=1000,reason='Live session ended. Reconnect to continue.')
    except (WebSocketDisconnect,RuntimeError):
        pass
    except TimeoutError:
        await socket.close(code=4408,reason='Live connection timed out.')
    except HTTPException as error:
        await fail(error.status_code,error.detail)
        await socket.close(code=4401 if error.status_code==401 else 4403 if error.status_code==403 else 1013)
    except RedisError:
        log.warning('Redis live-session coordination is unavailable.')
        await fail(503,'Live detection is temporarily unavailable. Redis must be connected.')
        await socket.close(code=1013)
    finally:
        if claimed and user:
            try:await sessions.release(user['id'],session_id)
            except RedisError:log.warning('Live lease cleanup deferred to its expiry.')

@app.get('/api/me')
async def me(user=Depends(actor)):return user

@app.patch('/api/me')
async def update_me(body:ProfileInput,user=Depends(actor),gw=Depends(gateway)):
    # Identity comes exclusively from the verified session, never the request body.
    payload=body.model_dump(exclude_unset=True)
    # Return only persisted choices. A database failure must never fabricate a
    # successful avatar save or silently replace it with a different portrait.
    rows=await gw.call('PATCH','/rest/v1/profiles',params={'id':'eq.'+user['id'],'select':'*'},
                       json=payload,headers={'Prefer':'return=representation'})
    if not isinstance(rows,list) or len(rows)!=1 or rows[0].get('id')!=user['id']:raise HTTPException(404,'Your team profile could not be updated.')
    return {**rows[0],'email':user.get('email',''),'avatar_url':rows[0].get('avatar_url') or user.get('avatar_url','')}

@app.get('/api/cases')
async def cases(user=Depends(member),gw=Depends(gateway)):
    return await gw.list('persons','*',order='created_at.desc',limit='200')

@app.post('/api/cases',status_code=201)
async def create_case(body:CaseInput,user=Depends(rescuer),gw=Depends(gateway)):
    return await gw.insert('persons',{**body.model_dump(),'created_by':user['id']})

@app.get('/api/cases/{case_id}')
async def case_detail(case_id:UUID,user=Depends(member),gw=Depends(gateway)):
    rows=await gw.list('persons','*',id='eq.'+str(case_id))
    if not rows:raise HTTPException(404,'Case not found.')
    refs=await gw.list('reference_images','id,modality,model,storage_path,created_at',person_id='eq.'+str(case_id),order='created_at.desc')
    public_refs=[]
    for ref in refs:
        public_ref={key:ref[key] for key in ['id','modality','model','created_at']}
        public_ref['photo_url']=await gw.sign_photo(ref['storage_path'])
        public_refs.append(public_ref)
    return {**rows[0],'references':public_refs}

@app.patch('/api/cases/{case_id}')
async def update_case(case_id:UUID,body:CaseUpdateInput,user=Depends(rescuer),gw=Depends(gateway)):
    payload=body.model_dump(exclude_unset=True)
    rows=await gw.call('PATCH','/rest/v1/persons',params={'id':'eq.'+str(case_id),'select':'*'},
                       json=payload,headers={'Prefer':'return=representation'})
    if not isinstance(rows,list) or len(rows)!=1:raise HTTPException(404,'Case record could not be updated.')
    return rows[0]

@app.post('/api/cases/{case_id}/references',status_code=201)
async def enroll(case_id:UUID,request:Request,modality:Literal['face','ear']=Form(...),cropped:bool=Form(False),face_index:int|None=Form(None,ge=0,le=99),detector_version:str|None=Form(None,max_length=64),image:UploadFile=File(...),user=Depends(rescuer),gw=Depends(gateway),engine=Depends(model_engine)):
    if modality=='face' and detector_version is not None and detector_version!=engine.bundle['detector']['sha256']:
        raise HTTPException(409,'The detector has been updated. Detect the faces again before enrolling.')
    persons=await gw.list('persons','id',id='eq.'+str(case_id))
    if not persons:raise HTTPException(404,'Case not found.')
    data=await read_image(image)
    vector,q,clean=await run_inference(request,describe,data,modality,cropped,engine,settings,face_index)
    rid=str(uuid4());path=f'{case_id}/{rid}.jpg'
    await gw.upload(path,clean,'image/jpeg')
    try:
        ref=await gw.insert('reference_images',{'id':rid,'person_id':str(case_id),'modality':modality,'model':MODEL_IDS[modality],'embedding':vector.tolist(),'quality':q,'storage_path':path,'source_sha256':hashlib.sha256(data).hexdigest(),'created_by':user['id']})
    except Exception:
        try:await gw.delete_photo(path)
        except Exception:log.warning('Failed to clean up a reference photo after database failure; reference=%s',rid)
        raise
    return {'id':ref['id'],'modality':modality,'quality':q}

@app.post('/api/search')
async def search(request:Request,face:UploadFile|None=File(None),ear:UploadFile|None=File(None),face_cropped:bool=Form(False),face_index:int|None=Form(None,ge=0,le=99),detector_version:str|None=Form(None,max_length=64),top_k:int=Form(5,ge=1,le=20),user=Depends(member),gw=Depends(gateway),engine=Depends(model_engine)):
    if not face:raise HTTPException(422,'Add a face photograph to search. Ear-only identity recognition has no validated model; retain ear photos as case evidence.')
    if detector_version is not None and detector_version!=engine.bundle['detector']['sha256']:
        raise HTTPException(409,'The detector has been updated. Detect the faces again before searching.')
    query={};qualities={}
    for modality,upload in [('face',face),('ear',ear)]:
        if upload:
            data=await read_image(upload)
            vector,q,_=await run_inference(request,describe,data,modality,face_cropped if modality=='face' else False,engine,settings,face_index if modality=='face' else None)
            query[modality]=vector;qualities[modality]=q
    refs=await gw.references()
    try:ranked=await run_inference(request,rank_candidates,query,refs,engine.bundle['thresholds'],top_k)
    except ValueError:raise HTTPException(409,'Registry representations are incompatible with the current model. Re-enroll the affected references.') from None
    if ranked:
        search_id=str(uuid4())
        bundle=engine.bundle
        audited_thresholds={**bundle['thresholds'],'_pipeline':{'detector_sha256':bundle['detector']['sha256'],
                            'recognizer_sha256':bundle['recognizer']['sha256'],'calibration':bundle['calibration']['status'],
                            'quality_limits':bundle['quality'],'independent_review_required':True}}
        payload=[{'person_id':c['person_id'],'score':c['score'],'components':c['components'],'created_by':user['id'],'search_id':search_id,'thresholds':audited_thresholds} for c in ranked]
        saved=await gw.insert('matches',payload)
        lookup={r['person_id']:r for r in saved}
        person_ids=','.join(c['person_id'] for c in ranked)
        people=await gw.list('persons','id,name,age,last_seen,status',id=f'in.({person_ids})')
        names={p['id']:p for p in people}
        ranked=[{**c,'id':lookup[c['person_id']]['id'],'person':names.get(c['person_id'])} for c in ranked]
    return {'candidates':ranked,'quality':qualities,'thresholds':engine.bundle['thresholds'],'score_is_probability':False,
            'calibration':engine.bundle['calibration']['status'],
            'message':'Face similarity produces leads for independent review. Disaster identity accuracy is not validated.',
            'warnings':['Ear similarity is excluded from identity ranking.'] if ear else []}

@app.get('/api/reviews')
async def reviews(user=Depends(member),gw=Depends(gateway)):
    items=await gw.list('matches','id,person_id,search_id,score,components,thresholds,status,created_at,persons(id,name,age,last_seen,notes,consent_basis,status)',order='created_at.desc',limit='200')
    async def sign_item(item):
        async def get_search():
            try:return await gw.sign_photo(f"searches/{item.get('search_id')}.jpg")
            except Exception:return None
        async def get_case():
            try:
                pid=item.get('person_id')
                refs=await gw.list('reference_images','storage_path',person_id=f'eq.{pid}',modality='eq.face',order='created_at.desc',limit='1')
                if refs:return await gw.sign_photo(refs[0]['storage_path'])
            except Exception:pass
            return None
        search_url,case_url=await asyncio.gather(get_search(),get_case())
        item['search_photo_url']=search_url
        item['case_photo_url']=case_url
        return item
    return await asyncio.gather(*(sign_item(i) for i in items))

@app.post('/api/reviews/{match_id}')
async def review(match_id:UUID,body:ReviewInput,user=Depends(verifier),gw=Depends(gateway)):
    return await gw.call('POST','/rest/v1/rpc/review_match',json={'p_match_id':str(match_id),'p_actor':user['id'],'p_decision':body.decision,'p_reason':body.reason})

@app.get('/api/team')
async def team(user=Depends(admin),gw=Depends(gateway)):
    return await gw.list('profiles','*',order='created_at.desc',limit='200')

@app.patch('/api/team/{user_id}')
async def assign_role(user_id:UUID,body:RoleInput,user=Depends(admin),gw=Depends(gateway)):
    if str(user_id)==user['id']:raise HTTPException(409,'An administrator cannot change their own role here.')
    return await gw.call('POST','/rest/v1/rpc/assign_team_role',json={'p_user_id':str(user_id),'p_actor':user['id'],'p_role':body.role})
