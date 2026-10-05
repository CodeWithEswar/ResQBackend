"""Server-only Supabase Auth, PostgREST and private Storage integration."""
import logging
from uuid import UUID
import httpx
from fastapi import HTTPException
from .config import Settings
from .profile_identity import identity_details

log=logging.getLogger(__name__)

class SupabaseGateway:
    def __init__(self,settings:Settings,client:httpx.AsyncClient):
        self.settings=settings;self.client=client

    def headers(self):
        key=self.settings.service_key
        # New sb_secret keys are API keys, not bearer JWTs.
        result={'apikey':key}
        if not key.startswith('sb_secret_'):result['Authorization']='Bearer '+key
        return result

    async def call(self,method,path,**kwargs):
        if not self.settings.configured:raise HTTPException(503,'Supabase is not configured on the server.')
        headers=self.headers();headers.update(kwargs.pop('headers',{}))
        try:r=await self.client.request(method,self.settings.url+path,headers=headers,**kwargs)
        except httpx.HTTPError:raise HTTPException(503,'Supabase is temporarily unavailable.') from None
        if r.is_error:
            # Return known transaction conflicts without exposing upstream data or keys.
            try:code=r.json().get('code')
            except ValueError:code=None
            if code=='RQ409':raise HTTPException(409,'This candidate already has a final review.')
            if code=='RQ404':raise HTTPException(404,'Candidate not found.')
            if code=='RQ403':raise HTTPException(403,'Your account cannot perform this action.')
            log.warning('Supabase request failed: method=%s status=%s code=%s',method,r.status_code,code)
            raise HTTPException(503,'Database or storage request failed. Check server configuration and migrations.')
        if not r.content:return None
        try:return r.json()
        except ValueError:raise HTTPException(503,'Unexpected response from Supabase.') from None

    async def authenticate(self,token):
        if not self.settings.configured:raise HTTPException(503,'Supabase is not configured on the server.')
        try:r=await self.client.get(self.settings.url+'/auth/v1/user',headers={'apikey':self.settings.public_key,'Authorization':'Bearer '+token})
        except httpx.HTTPError:raise HTTPException(503,'Authentication is temporarily unavailable.') from None
        if r.status_code in (401,403):raise HTTPException(401,'Session expired. Please sign in again.')
        if r.is_error:raise HTTPException(503,'Authentication is temporarily unavailable.')
        try:user=r.json();uid=str(UUID(user['id']))
        except (ValueError,KeyError,TypeError):raise HTTPException(401,'Invalid authenticated user.') from None
        profiles=await self.call('GET','/rest/v1/profiles',params={'id':'eq.'+uid,'select':'*'})
        name,photo=identity_details(user)
        if profiles and profiles[0]['id']==uid and not profiles[0].get('display_name','').strip() and name:
            # Comparing the old value preserves a concurrent custom name edit.
            repaired=await self.call('PATCH','/rest/v1/profiles',params={'id':'eq.'+uid,'display_name':'eq.'+profiles[0].get('display_name',''),'select':'*'},json={'display_name':name},headers={'Prefer':'return=representation'})
            profiles=repaired or await self.call('GET','/rest/v1/profiles',params={'id':'eq.'+uid,'select':'*'})
        if not profiles or profiles[0]['id']!=uid:raise HTTPException(403,'No team profile is available.')
        profile=profiles[0]
        return {key:profile[key] for key in ('id','display_name','role','avatar_key','avatar_color','avatar_shape','avatar_style','created_at') if key in profile} | {'email':user.get('email',''),'avatar_url':profile.get('avatar_url') or photo}

    async def list(self,table,select='*',**filters):
        return await self.call('GET','/rest/v1/'+table,params={'select':select,**filters})

    async def insert(self,table,value):
        rows=await self.call('POST','/rest/v1/'+table,json=value,headers={'Prefer':'return=representation'})
        return rows if isinstance(value,list) else rows[0]

    async def upload(self,path,data,mime):
        await self.call('POST','/storage/v1/object/case-photos/'+path,content=data,headers={'Content-Type':mime,'x-upsert':'false'})

    async def delete_photo(self,path):
        await self.call('DELETE','/storage/v1/object/case-photos',json={'prefixes':[path]})

    async def sign_photo(self,path):
        result=await self.call('POST','/storage/v1/object/sign/case-photos/'+path,json={'expiresIn':300})
        signed=result.get('signedURL') or result.get('signedUrl')
        if not signed:raise HTTPException(503,'Photo link could not be generated.')
        return self.settings.url+'/storage/v1'+signed

    async def references(self):
        # Page by immutable UUID, avoiding Supabase's default 1,000-row cap.
        out=[];cursor=None
        while True:
            params={'order':'id.asc','limit':'1000'}
            if cursor:params['id']='gt.'+cursor
            batch=await self.list('reference_images','id,person_id,modality,model,embedding',**params)
            out.extend(batch)
            if len(batch)<1000:return out
            cursor=batch[-1]['id']
