"""Repair Google presentation data with server credentials; output counts only.

Dry-run by default. This does not grant roles or overwrite chosen avatars/names.
Run: python -m ml.backfill_profiles [--apply]
"""
import argparse,asyncio,json
import httpx
from app.config import Settings
from app.supabase import SupabaseGateway
from app.profile_identity import identity_details

async def repair(apply=False):
    counts={'google_users':0,'profiles_repaired':0,'auth_metadata_repaired':0,'missing_name_after':0,'profile_photo_column_available':False,'applied':apply}
    async with httpx.AsyncClient(timeout=20) as client:
        gw=SupabaseGateway(Settings(),client);page=1
        while True:
            result=await gw.call('GET','/auth/v1/admin/users',params={'page':page,'per_page':1000})
            users=result.get('users',[])
            for item in users:
                if 'google' not in (item.get('app_metadata') or {}).get('providers',[]):continue
                user=await gw.call('GET','/auth/v1/admin/users/'+item['id'])
                uid=user['id'];counts['google_users']+=1
                rows=await gw.list('profiles','*',id='eq.'+uid)
                profile=rows[0] if rows else {}
                name,photo=identity_details(user)
                custom_name=profile.get('display_name','').strip()
                metadata=user.get('user_metadata') or {};patch={}
                if custom_name or name:
                    canonical=custom_name or name
                    if metadata.get('display_name')!=canonical:patch['display_name']=canonical
                if photo and not metadata.get('avatar_url'):patch['avatar_url']=photo
                for key in ('avatar_key','avatar_color','avatar_shape','avatar_style'):
                    if profile.get(key) and metadata.get(key)!=profile[key]:patch[key]=profile[key]
                if patch:
                    counts['auth_metadata_repaired']+=1
                    if apply:await gw.call('PUT','/auth/v1/admin/users/'+uid,json={'user_metadata':patch})
                if not profile:
                    counts['profiles_repaired']+=1
                    if apply:
                        await gw.call('POST','/rest/v1/profiles',json={'id':uid,'display_name':name},headers={'Prefer':'resolution=ignore-duplicates'})
                else:
                    details={}
                    if not custom_name and name:details['display_name']=name
                    if 'email' in profile and profile['email']!=user.get('email',''):details['email']=user.get('email','')
                    if 'avatar_url' in profile:
                        counts['profile_photo_column_available']=True
                        if photo and profile['avatar_url']!=photo:details['avatar_url']=photo
                    if details:
                        counts['profiles_repaired']+=1
                        if apply:
                            await gw.call('PATCH','/rest/v1/profiles',params={'id':'eq.'+uid,'display_name':'eq.'+profile.get('display_name','')},json=details)
                if apply:
                    saved=(await gw.list('profiles','*',id='eq.'+uid))[0]
                    assert not profile or saved['role']==profile['role'],'Backfill must preserve access'
                    for key in ('avatar_key','avatar_color','avatar_shape','avatar_style'):
                        assert not profile.get(key) or saved.get(key)==profile[key],'Backfill must preserve custom avatars'
                    counts['missing_name_after']+=int(not saved.get('display_name','').strip())
                else:counts['missing_name_after']+=int(not (custom_name or name))
            if len(users)<1000:break
            page+=1
    return counts

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    print(json.dumps(asyncio.run(repair(parser.parse_args().apply))))
