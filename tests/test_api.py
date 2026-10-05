import asyncio,io,json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4
import cv2,httpx,numpy as np,pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app,gateway,describe,settings
from app.supabase import SupabaseGateway
from app.engine import MODEL_IDS

FIXTURES=Path(__file__).parent/'fixtures'
ACTORS={role:str(uuid4()) for role in ['pending','rescuer','verifier','admin']}

class FakeGateway:
    """Only the external Supabase boundary is replaced; real inference runs."""
    def __init__(self):self.tables={t:[] for t in ['persons','reference_images','matches','profiles']};self.objects={};self.fail_insert=False;self.calls=[]
    async def authenticate(self,token):
        if token not in ACTORS:raise HTTPException(401,'Invalid token')
        stored=next((row for row in self.tables['profiles'] if row['id']==ACTORS[token]),{})
        return {'id':ACTORS[token],'role':token,'display_name':token,'email':'test@example.invalid',**stored}
    async def list(self,table,select='*',**filters):
        rows=self.tables[table]
        for key,value in filters.items():
            if value.startswith('eq.'):rows=[r for r in rows if r[key]==value[3:]]
            elif value.startswith('in.'):rows=[r for r in rows if r[key] in value[4:-1].split(',')]
        return [dict(r) for r in rows]
    async def insert(self,table,value):
        if self.fail_insert:raise HTTPException(503,'Simulated database failure')
        multiple=isinstance(value,list);rows=value if multiple else [value];out=[]
        for row in rows:
            saved={'id':str(uuid4()),'created_at':'2026-10-04T00:00:00Z',**row}
            if table=='matches':saved['status']='pending'
            self.tables[table].append(saved);out.append(saved)
        return out if multiple else out[0]
    async def upload(self,path,data,mime):self.objects[path]=data
    async def delete_photo(self,path):self.objects.pop(path,None)
    async def sign_photo(self,path):return 'https://example.invalid/private/'+path
    async def references(self):return self.tables['reference_images']
    async def call(self,method,path,**kw):
        self.calls.append((method,path,kw))
        if method=='PATCH' and path in ('/rest/v1/profiles','/rest/v1/persons'):
            if self.fail_insert:raise HTTPException(503,'Simulated database failure')
            tbl='profiles' if path=='/rest/v1/profiles' else 'persons'
            rows=[row for row in self.tables[tbl] if 'eq.'+row['id']==kw['params']['id']]
            for row in rows:row.update(kw['json'])
            return [dict(row) for row in rows]
        return kw['json']

@pytest.fixture
def client():
    fake=FakeGateway();app.dependency_overrides[gateway]=lambda:fake
    with TestClient(app) as c:yield c,fake
    app.dependency_overrides.clear()

def auth(role):return {'Authorization':'Bearer '+role}
def case(c):return c.post('/api/cases',json={'name':'Reference subject','consent_basis':'research'},headers=auth('rescuer')).json()
def photo(name):return (FIXTURES/name).read_bytes()

def test_unauthenticated_and_pending_denied(client):
    c,_=client
    assert c.get('/api/cases').status_code==401
    assert c.get('/api/cases',headers=auth('pending')).status_code==403
    assert c.get('/api/me',headers=auth('pending')).json()['role']=='pending'

def test_server_role_cannot_be_forged(client):
    c,_=client
    assert c.post('/api/cases',json={'name':'x','consent_basis':'research'},headers=auth('verifier')).status_code==403
    assert c.post('/api/cases',json={'name':'x','consent_basis':'research','created_by':ACTORS['admin']},headers=auth('rescuer')).status_code==422
    assert c.patch('/api/team/'+ACTORS['rescuer'],json={'role':'verifier'},headers=auth('rescuer')).status_code==403
    assert c.patch('/api/team/'+ACTORS['admin'],json={'role':'pending'},headers=auth('admin')).status_code==409

@pytest.mark.parametrize('role',['pending','rescuer','verifier','admin'])
def test_profile_edit_updates_only_authenticated_name(client,role):
    c,fake=client
    own={'id':ACTORS[role],'display_name':'Old name','role':role}
    other={'id':str(uuid4()),'display_name':'Other teammate','role':'rescuer'}
    fake.tables['profiles']=[own,other]
    response=c.patch('/api/me',json={'display_name':'  Priya Rao  '},headers=auth(role))
    assert response.status_code==200,response.text
    assert response.json()['display_name']=='Priya Rao' and response.json()['role']==role
    assert c.get('/api/me',headers=auth(role)).json()['display_name']=='Priya Rao'
    assert other['display_name']=='Other teammate'
    method,path,payload=fake.calls[-1]
    assert method=='PATCH' and path=='/rest/v1/profiles'
    assert payload['params']['id']=='eq.'+ACTORS[role]
    assert payload['json']=={'display_name':'Priya Rao'}
    assert payload['headers']=={'Prefer':'return=representation'}

@pytest.mark.parametrize('body',[{'display_name':' '},{'display_name':'a'*121},{'display_name':'a\nb'},
                                 {'display_name':'Name','role':'admin'},{'display_name':'Name','id':str(uuid4())},
                                 {'display_name':'Name','email':'other@example.invalid'}])
def test_profile_edit_rejects_invalid_and_privileged_fields(client,body):
    c,fake=client
    assert c.patch('/api/me',json=body,headers=auth('rescuer')).status_code==422
    assert not fake.calls

def test_profile_edit_auth_missing_row_and_failure(client):
    c,fake=client
    assert c.patch('/api/me',json={'display_name':'Name'}).status_code==401
    assert c.patch('/api/me',json={'display_name':'Name'},headers=auth('rescuer')).status_code==404
    own={'id':ACTORS['rescuer'],'display_name':'Old name','role':'rescuer'}
    fake.tables['profiles']=[own];fake.fail_insert=True
    assert c.patch('/api/me',json={'display_name':'Name'},headers=auth('rescuer')).status_code==503
    assert own['display_name']=='Old name'

def test_profile_avatar_saved_and_returned_on_refresh(client):
    c,fake=client
    own={'id':ACTORS['rescuer'],'display_name':'Priya','role':'rescuer','avatar_key':'short-hair','avatar_color':'violet'}
    fake.tables['profiles']=[own]
    response=c.patch('/api/me',json={'display_name':'Priya','avatar_key':'tied-hair','avatar_color':'cyan'},headers=auth('rescuer'))
    assert response.status_code==200,response.text
    assert response.json()['avatar_key']=='tied-hair' and response.json()['avatar_color']=='cyan'
    assert c.get('/api/me',headers=auth('rescuer')).json()['avatar_key']=='tied-hair'
    assert own['role']=='rescuer'
    assert c.patch('/api/me',json={'display_name':'Priya'},headers=auth('rescuer')).status_code==200
    assert own['avatar_key']=='tied-hair' # Name-only edits preserve existing choices.

@pytest.mark.parametrize('body',[{'display_name':'Name','avatar_key':'https://example.invalid/pic.png'},
                                 {'display_name':'Name','avatar_color':'#000000'},{'display_name':'Name','avatar_key':None}])
def test_profile_avatar_rejects_unknown_catalog_choices(client,body):
    c,fake=client
    assert c.patch('/api/me',json=body,headers=auth('rescuer')).status_code==422
    assert not fake.calls

@pytest.mark.parametrize('color',['rose','indigo','lime','silver'])
@pytest.mark.parametrize('shape',['rounded-square','pentagon','star'])
def test_premium_avatar_choices_persist(client,color,shape):
    c,fake=client
    own={'id':ACTORS['rescuer'],'display_name':'Priya','role':'rescuer'}
    fake.tables['profiles']=[own]
    response=c.patch('/api/me',json={'display_name':'Priya','avatar_color':color,'avatar_shape':shape,'avatar_style':'3d'},headers=auth('rescuer'))
    assert response.status_code==200,response.text
    refreshed=c.get('/api/me',headers=auth('rescuer')).json()
    assert refreshed['avatar_color']==color and refreshed['avatar_shape']==shape
    assert own['role']=='rescuer'

@pytest.mark.parametrize('body',[{'name':' ','consent_basis':'family'},{'name':'x','consent_basis':'missing'},{'name':'x','consent_basis':'research','age':-1}])
def test_invalid_case(client,body):assert client[0].post('/api/cases',json=body,headers=auth('rescuer')).status_code==422

@pytest.mark.parametrize('modality,name',[('face','face-reference.jpg')])
def test_actual_model_enrollment_search_and_private_detail(client,modality,name):
    c,fake=client;p=case(c)
    response=c.post('/api/cases/'+p['id']+'/references',data={'modality':modality},files={'image':(name,photo(name))},headers=auth('rescuer'))
    assert response.status_code==201,response.text
    ref=fake.tables['reference_images'][0]
    assert len(ref['embedding'])==(128 if modality=='face' else 3780)
    assert np.isclose(np.linalg.norm(ref['embedding']),1)
    assert ref['model']==MODEL_IDS[modality]
    response=c.post('/api/search',files={modality:(name,photo(name))},headers=auth('rescuer'))
    assert response.status_code==200,response.text
    result=response.json();assert result['score_is_probability'] is False
    assert result['candidates'][0]['person_id']==p['id'] and result['candidates'][0]['score']>0.99
    assert result['candidates'][0]['status']=='pending' and 'id' in result['candidates'][0]
    assert fake.tables['matches'][0]['thresholds']['_pipeline']['independent_review_required'] is True
    detail=c.get('/api/cases/'+p['id'],headers=auth('verifier')).json()
    assert 'embedding' not in detail['references'][0] and 'photo_url' in detail['references'][0]
    assert len(fake.objects)==1 # Search query photos are not persisted.

def test_upload_validation_and_cleanup(client):
    c,fake=client;p=case(c)
    assert c.post('/api/cases/'+p['id']+'/references',data={'modality':'ear'},files={'image':('x.jpg',b'bad')},headers=auth('rescuer')).status_code==415
    fake.fail_insert=True
    assert c.post('/api/cases/'+p['id']+'/references',data={'modality':'ear'},files={'image':('ear.jpg',photo('ear.jpg'))},headers=auth('rescuer')).status_code==503
    assert not fake.objects

def test_search_contract(client):
    c,_=client
    assert c.post('/api/search',headers=auth('rescuer')).status_code==422
    assert c.post('/api/search',data={'top_k':'21'},files={'ear':('ear.jpg',photo('ear.jpg'))},headers=auth('rescuer')).status_code==422
    assert c.post('/api/search',files={'ear':('ear.jpg',photo('ear.jpg'))},headers=auth('rescuer')).status_code==422
    assert c.post('/api/search',files={'face':('face.jpg',photo('face-reference.jpg'))},headers=auth('rescuer')).json()['candidates']==[]

def test_ear_is_stored_as_evidence_but_never_creates_identity_candidates(client):
    c,fake=client;p=case(c)
    r=c.post('/api/cases/'+p['id']+'/references',data={'modality':'ear'},files={'image':('ear.jpg',photo('ear.jpg'))},headers=auth('rescuer'))
    assert r.status_code==201
    assert len(fake.tables['reference_images'][0]['embedding'])==3780
    assert c.post('/api/search',files={'ear':('ear.jpg',photo('ear.jpg'))},headers=auth('rescuer')).status_code==422
    r=c.post('/api/search',files={'face':('face.jpg',photo('face-reference.jpg')),'ear':('ear.jpg',photo('ear.jpg'))},headers=auth('rescuer'))
    assert r.status_code==200 and r.json()['candidates']==[] and r.json()['warnings']

def test_actual_yolo_detection_and_crop_cannot_bypass_quality(client):
    c,_=client
    assert c.post('/api/faces/detect',files={'image':('face.jpg',photo('face-reference.jpg'))}).status_code==401
    result=c.post('/api/faces/detect',files={'image':('face.jpg',photo('face-reference.jpg'))},headers=auth('rescuer')).json()
    assert result['detector']=='YOLOv8n-face' and len(result['faces'])==1
    assert result['faces'][0]['usable'] and result['faces'][0]['confidence']>0.4
    r=c.post('/api/search',data={'face_cropped':'true'},files={'face':('tiny.png',photo('face.png'))},headers=auth('rescuer'))
    assert r.status_code==422
    buffer=io.BytesIO();Image.new('RGB',(256,256),'white').save(buffer,format='PNG')
    r=c.post('/api/faces/detect',files={'image':('blank.png',buffer.getvalue())},headers=auth('rescuer'))
    assert r.status_code==200 and r.json()['faces']==[]
    r=c.post('/api/search',data={'face_cropped':'true'},files={'face':('blank.png',buffer.getvalue())},headers=auth('rescuer'))
    assert r.status_code==422

def test_group_photo_requires_explicit_face_selection(client):
    c,fake=client;p=case(c)
    image=cv2.imdecode(np.frombuffer(photo('face-reference.jpg'),dtype=np.uint8),cv2.IMREAD_COLOR)
    x1,y1,x2,y2=app.state.engine.extract_face(image)[1]['box']
    margin=30
    image=image[max(0,y1-margin):y2+margin,max(0,x1-margin):x2+margin]
    group=cv2.hconcat([image,image]);ok,encoded=cv2.imencode('.jpg',group);assert ok
    data=encoded.tobytes()
    detected=c.post('/api/faces/detect',files={'image':('group.jpg',data)},headers=auth('rescuer')).json()
    assert len(detected['faces'])==2 and detected['selection_required']
    assert c.post('/api/search',files={'face':('group.jpg',data)},headers=auth('rescuer')).status_code==422
    assert c.post('/api/cases/'+p['id']+'/references',data={'modality':'face'},files={'image':('group.jpg',data)},headers=auth('rescuer')).status_code==422
    r=c.post('/api/cases/'+p['id']+'/references',data={'modality':'face','face_index':'1'},files={'image':('group.jpg',data)},headers=auth('rescuer'))
    assert r.status_code==201,r.text
    assert fake.tables['reference_images'][0]['quality']['box'][0]>image.shape[1]
    r=c.post('/api/search',data={'face_index':'1'},files={'face':('group.jpg',data)},headers=auth('rescuer'))
    assert r.status_code==200 and r.json()['candidates'][0]['status']=='pending'
    assert r.json()['candidates'][0]['score']>0.99
    assert c.post('/api/search',data={'face_index':'99'},files={'face':('group.jpg',data)},headers=auth('rescuer')).status_code==422

def test_health_reports_actual_model_and_validation_state(client):
    result=client[0].get('/health').json()
    assert result['ready'] and result['detector']=='YOLOv8n-face'
    assert len(result['detector_sha256'])==64
    assert not result['ear_recognition_enabled'] and not result['disaster_validated']

def test_stale_detector_selection_is_rejected(client):
    c,_=client;p=case(c)
    data={'face_index':'0','detector_version':'0'*64}
    assert c.post('/api/search',data=data,files={'face':('face.jpg',photo('face-reference.jpg'))},headers=auth('rescuer')).status_code==409
    assert c.post('/api/cases/'+p['id']+'/references',data={**data,'modality':'face'},files={'image':('face.jpg',photo('face-reference.jpg'))},headers=auth('rescuer')).status_code==409

def test_busy_service_returns_retryable_error(client,monkeypatch):
    import app.main as service
    c,_=client
    monkeypatch.setattr(service,'settings',replace(settings,inference_queue_seconds=0.01))
    app.state.inference_slots=asyncio.Semaphore(0)
    r=c.post('/api/faces/detect',files={'image':('face.jpg',photo('face-reference.jpg'))},headers=auth('rescuer'))
    assert r.status_code==503 and r.headers['retry-after']=='3'

def test_model_release_reload_is_atomic_and_corruption_fails_closed(client,monkeypatch,tmp_path):
    import shutil
    import app.main as service
    from ml.artifacts import read_bundle,write_bundle
    c,_=client
    bundle=read_bundle(settings.models/'pipeline.pkl')
    for kind in ('detector','recognizer','landmarks'):
        shutil.copy2(settings.models/bundle[kind]['file'],tmp_path/bundle[kind]['file'])
    bundle['thresholds']['face']=0.99
    write_bundle(tmp_path/'pipeline.pkl',bundle)
    monkeypatch.setattr(service,'settings',replace(settings,models=tmp_path))
    assert c.get('/health').status_code==200
    assert app.state.engine.bundle['thresholds']['face']==0.99
    previous=app.state.engine
    (tmp_path/'pipeline.pkl').write_bytes(b'corrupted release')
    assert c.get('/health').status_code==503
    assert app.state.engine is previous

def test_image_limits_before_decode(client):
    # Use a real source photo with a deliberately smaller configured dimension cap.
    with pytest.raises(HTTPException) as error:describe(photo('ear.jpg'),'ear',False,app.state.engine,replace(settings,max_pixels=10))
    assert error.value.status_code==413

def test_review_passes_authenticated_actor_and_enforces_role(client):
    c,_=client;mid=str(uuid4())
    assert c.post('/api/reviews/'+mid,json={'decision':'verified','reason':'Independent evidence checked'},headers=auth('rescuer')).status_code==403
    result=c.post('/api/reviews/'+mid,json={'decision':'verified','reason':'Independent evidence checked'},headers=auth('verifier'))
    assert result.json()['p_actor']==ACTORS['verifier']
    assert c.post('/api/reviews/'+mid,json={'decision':'verified','reason':'  '},headers=auth('verifier')).status_code==422

def test_gateway_uses_remote_auth_and_keeps_new_secret_out_of_bearer():
    seen=[]
    def handler(r):
        seen.append(r)
        if r.url.path=='/auth/v1/user':return httpx.Response(200,json={'id':ACTORS['rescuer'],'email':'test@example.invalid'})
        return httpx.Response(200,json=[{'id':ACTORS['rescuer'],'role':'rescuer','display_name':'test'}])
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gw=SupabaseGateway(replace(settings,url='https://example.invalid',public_key='sb_publishable_test',service_key='sb_secret_test'),client)
            assert (await gw.authenticate('user-token'))['role']=='rescuer'
    asyncio.run(run())
    assert seen[0].headers['authorization']=='Bearer user-token'
    assert seen[1].headers['apikey']=='sb_secret_test' and 'authorization' not in seen[1].headers

def test_gateway_pages_all_references_and_maps_duplicate_conflict():
    cursor=str(uuid4());seen=[]
    def handler(r):
        seen.append(r)
        if 'rpc' in r.url.path:return httpx.Response(400,json={'code':'RQ409','message':'private details'})
        return httpx.Response(200,json=[{'id':cursor}]*1000 if len(seen)==1 else [{'id':str(uuid4())}])
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gw=SupabaseGateway(replace(settings,url='https://example.invalid',public_key='public',service_key='secret'),client)
            assert len(await gw.references())==1001
            with pytest.raises(HTTPException) as error:await gw.call('POST','/rest/v1/rpc/review_match',json={})
            assert error.value.status_code==409 and 'private' not in error.value.detail
    asyncio.run(run());assert seen[1].url.params['id']=='gt.'+cursor

def test_case_status_lifecycle_updates(client):
    c,_=client
    p=case(c)
    assert p.get('status','pending')=='pending'

    # Rescuer transitions through valid statuses
    for new_status in ['urgent','ongoing','completed','closed','pending']:
        res=c.patch('/api/cases/'+p['id'],json={'status':new_status},headers=auth('rescuer'))
        assert res.status_code==200,res.text
        assert res.json()['status']==new_status

    # Invalid status is rejected by Pydantic validation
    bad_res=c.patch('/api/cases/'+p['id'],json={'status':'invalid_status'},headers=auth('rescuer'))
    assert bad_res.status_code==422

    # Verifier or pending user cannot transition case status
    denied=c.patch('/api/cases/'+p['id'],json={'status':'ongoing'},headers=auth('verifier'))
    assert denied.status_code==403
