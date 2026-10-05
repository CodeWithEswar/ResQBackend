-- Google sends full_name/name and avatar_url/picture, not ResQ's display_name.
-- Authorization continues to come exclusively from profiles.role.
begin;
alter table public.profiles add column if not exists email text not null default '';
alter table public.profiles add column if not exists avatar_url text not null default '';
create schema if not exists resq_private;
revoke all on schema resq_private from public,anon,authenticated;

create or replace function resq_private.normalize_auth_profile() returns trigger
language plpgsql security definer set search_path = '' as $$
declare m jsonb := coalesce(new.raw_user_meta_data,'{}'::jsonb); n text; photo text;
begin
  n := left(coalesce(nullif(btrim(m->>'display_name'),''),nullif(btrim(m->>'full_name'),''),nullif(btrim(m->>'name'),''),''),120);
  photo := coalesce(nullif(m->>'avatar_url',''),nullif(m->>'picture',''),'');
  if n <> '' then m := m || jsonb_build_object('display_name',n); end if;
  if photo <> '' then m := m || jsonb_build_object('avatar_url',photo); end if;
  new.raw_user_meta_data := m;
  return new;
end $$;
revoke all on function resq_private.normalize_auth_profile() from public,anon,authenticated,service_role;
drop trigger if exists resq_normalize_auth_profile on auth.users;
create trigger resq_normalize_auth_profile before insert or update of raw_user_meta_data on auth.users
for each row execute function resq_private.normalize_auth_profile();

create or replace function public.create_resq_profile() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  insert into public.profiles(id,display_name,email,avatar_url)
  values(new.id,left(coalesce(nullif(btrim(new.raw_user_meta_data->>'display_name'),''),nullif(btrim(new.raw_user_meta_data->>'full_name'),''),nullif(btrim(new.raw_user_meta_data->>'name'),''),''),120),coalesce(new.email,''),
    coalesce(nullif(new.raw_user_meta_data->>'avatar_url',''),nullif(new.raw_user_meta_data->>'picture',''),''))
  on conflict (id) do update set
    display_name = case when btrim(profiles.display_name) = '' then excluded.display_name else profiles.display_name end,
    email = excluded.email, avatar_url = excluded.avatar_url
  where (btrim(profiles.display_name) = '' and excluded.display_name <> '')
    or profiles.email is distinct from excluded.email or profiles.avatar_url is distinct from excluded.avatar_url;
  return new;
end $$;
revoke all on function public.create_resq_profile() from public,anon,authenticated,service_role;
drop trigger if exists resq_auth_profile_updated on auth.users;
create trigger resq_auth_profile_updated after update of raw_user_meta_data,email on auth.users
for each row execute function public.create_resq_profile();

-- Profile edits persist to Auth in the same transaction. A guarded UPDATE
-- prevents a trigger cycle. Provider full_name/name are preserved.
create or replace function resq_private.persist_profile_metadata() returns trigger
language plpgsql security definer set search_path = '' as $$
declare patch jsonb;
begin
  patch := jsonb_strip_nulls(jsonb_build_object('display_name',new.display_name,'avatar_key',new.avatar_key,'avatar_color',new.avatar_color,
    'avatar_shape',to_jsonb(new)->'avatar_shape','avatar_style',to_jsonb(new)->'avatar_style'));
  update auth.users set raw_user_meta_data = coalesce(raw_user_meta_data,'{}'::jsonb) || patch
  where id = new.id and not (coalesce(raw_user_meta_data,'{}'::jsonb) @> patch);
  return new;
end $$;
revoke all on function resq_private.persist_profile_metadata() from public,anon,authenticated,service_role;
drop trigger if exists resq_profile_metadata on public.profiles;
create trigger resq_profile_metadata after insert or update on public.profiles
for each row execute function resq_private.persist_profile_metadata();

-- Repair legacy metadata, retaining a customized profile name and team role.
with provider as (
  select distinct on (user_id) user_id, identity_data from auth.identities
  where provider = 'google' order by user_id,id
), repaired as (
  select u.id,coalesce(u.raw_user_meta_data,'{}'::jsonb) || jsonb_strip_nulls(jsonb_build_object(
    'display_name',nullif(left(coalesce(nullif(btrim(p.display_name),''),nullif(btrim(u.raw_user_meta_data->>'display_name'),''),nullif(btrim(u.raw_user_meta_data->>'full_name'),''),nullif(btrim(u.raw_user_meta_data->>'name'),''),nullif(btrim(g.identity_data->>'full_name'),''),nullif(btrim(g.identity_data->>'name'),''),''),120),''),
    'full_name',coalesce(nullif(u.raw_user_meta_data->>'full_name',''),nullif(g.identity_data->>'full_name',''),nullif(g.identity_data->>'name','')),
    'avatar_url',coalesce(nullif(u.raw_user_meta_data->>'avatar_url',''),nullif(u.raw_user_meta_data->>'picture',''),nullif(g.identity_data->>'avatar_url',''),nullif(g.identity_data->>'picture',''))
  )) as metadata
  from auth.users u left join public.profiles p on p.id=u.id left join provider g on g.user_id=u.id
)
update auth.users u set raw_user_meta_data = r.metadata from repaired r
where u.id=r.id and u.raw_user_meta_data is distinct from r.metadata;

insert into public.profiles(id,display_name,email,avatar_url)
select id,left(coalesce(nullif(btrim(raw_user_meta_data->>'display_name'),''),nullif(btrim(raw_user_meta_data->>'full_name'),''),nullif(btrim(raw_user_meta_data->>'name'),''),''),120),coalesce(email,''),coalesce(nullif(raw_user_meta_data->>'avatar_url',''),nullif(raw_user_meta_data->>'picture',''),'') from auth.users
on conflict (id) do update set
  display_name = case when btrim(profiles.display_name) = '' then excluded.display_name else profiles.display_name end,
  email = excluded.email, avatar_url = excluded.avatar_url;
notify pgrst,'reload schema';
commit;
