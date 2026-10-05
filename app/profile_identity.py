"""Presentation metadata only. Never derive workspace access from these values."""
def identity_details(user):
    metadata=user.get('user_metadata') or {}
    identities=[i.get('identity_data') or {} for i in (user.get('identities') or []) if i.get('provider')=='google']
    sources=[metadata,*identities]
    name=next((v.strip()[:120] for source in sources for key in ('display_name','full_name','name')
               if isinstance(v:=source.get(key),str) and v.strip()),'')
    photo=next((v for source in sources for key in ('avatar_url','picture')
                if isinstance(v:=source.get(key),str) and v.startswith('https://') and len(v)<=2048),'')
    return name,photo
