import asyncio,json
from uuid import uuid4
import httpx
from app.config import Settings
from app.supabase import SupabaseGateway
from app.profile_identity import identity_details

def test_google_metadata_aliases_and_identity_fallback():
    assert identity_details({'user_metadata':{'name':' Google Name ','picture':'https://example.invalid/photo'}})==('Google Name','https://example.invalid/photo')
    assert identity_details({'identities':[{'provider':'google','identity_data':{'full_name':'Identity Name','avatar_url':'https://example.invalid/avatar'}}]})[0]=='Identity Name'
    assert identity_details({'user_metadata':{'name':{},'avatar_url':'javascript:alert(1)'},'identities':None})==('','')
    assert len(identity_details({'user_metadata':{'full_name':'x'*150}})[0])==120

def test_verified_session_repairs_blank_profile_and_keeps_server_role():
    uid=str(uuid4());seen=[];profile={'id':uid,'display_name':'','role':'pending','avatar_key':'tied-hair','avatar_color':'amber'}
    def handle(request):
        seen.append(request)
        if request.url.path=='/auth/v1/user':return httpx.Response(200,json={'id':uid,'email':'google@example.invalid','user_metadata':{'full_name':'Google Person','picture':'https://example.invalid/pic','role':'admin'}})
        if request.method=='PATCH':
            assert request.url.params['id']=='eq.'+uid and request.url.params['display_name']=='eq.'
            assert json.loads(request.content)=={'display_name':'Google Person'}
            profile.update(json.loads(request.content))
        return httpx.Response(200,json=[profile])
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result=await SupabaseGateway(Settings(url='https://example.invalid',public_key='public',service_key='sb_secret_test'),client).authenticate('verified-token')
            assert result['display_name']=='Google Person' and result['role']=='pending'
            assert result['avatar_key']=='tied-hair' and result['avatar_url']=='https://example.invalid/pic'
            assert result['email']=='google@example.invalid'
    asyncio.run(run())
    assert seen[0].headers['authorization']=='Bearer verified-token' and len(seen)==3

def test_google_signin_preserves_custom_name_and_no_writes_on_complete_profile():
    uid=str(uuid4());seen=[]
    def handle(request):
        seen.append(request)
        if request.url.path=='/auth/v1/user':return httpx.Response(200,json={'id':uid,'user_metadata':{'full_name':'Provider Name'}})
        return httpx.Response(200,json=[{'id':uid,'display_name':'Custom Name','role':'rescuer','avatar_key':'curly-hair','avatar_color':'cyan'}])
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result=await SupabaseGateway(Settings(url='https://example.invalid',public_key='public',service_key='sb_secret_test'),client).authenticate('verified-token')
            assert result['display_name']=='Custom Name' and result['avatar_color']=='cyan'
    asyncio.run(run());assert len(seen)==2 and all(r.method=='GET' for r in seen)

def test_name_repair_race_preserves_new_custom_name():
    uid=str(uuid4());reads=0
    def handle(request):
        nonlocal reads
        if request.url.path=='/auth/v1/user':return httpx.Response(200,json={'id':uid,'user_metadata':{'name':'Google Person'}})
        if request.method=='PATCH':return httpx.Response(200,json=[])
        reads+=1
        return httpx.Response(200,json=[{'id':uid,'display_name':'' if reads==1 else 'Just Edited','role':'verifier'}])
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result=await SupabaseGateway(Settings(url='https://example.invalid',public_key='public',service_key='sb_secret_test'),client).authenticate('verified-token')
            assert result['display_name']=='Just Edited'
    asyncio.run(run())
