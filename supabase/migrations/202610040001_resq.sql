-- Apply once in a new Supabase project via SQL Editor or `supabase db push`.
begin;

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text not null default '' check (length(display_name)<=120),
  role text not null default 'pending' check (role in ('pending','rescuer','verifier','admin')),
  created_at timestamptz not null default now()
);

create function public.create_resq_profile() returns trigger
language plpgsql security definer set search_path=public,pg_temp as $$
begin
  insert into public.profiles(id,display_name) values(new.id,left(coalesce(new.raw_user_meta_data->>'display_name',''),120));
  return new;
end $$;
create trigger resq_signup after insert on auth.users for each row execute function public.create_resq_profile();
-- Include accounts that existed before the migration, without trusting client-supplied roles.
insert into public.profiles(id,display_name)
select id,left(coalesce(raw_user_meta_data->>'display_name',''),120) from auth.users;

create table public.persons (
  id uuid primary key default gen_random_uuid(),
  name text not null check (length(trim(name)) between 1 and 120),
  age integer check (age between 0 and 120),
  last_seen text not null default '' check(length(last_seen)<=240),
  notes text not null default '' check(length(notes)<=2000),
  consent_basis text not null check (consent_basis in ('self','family','authority','research')),
  created_by uuid not null references public.profiles(id),
  created_at timestamptz not null default now()
);
create table public.reference_images (
  id uuid primary key default gen_random_uuid(),
  person_id uuid not null references public.persons(id),
  modality text not null check(modality in ('face','ear')),
  model text not null,
  embedding jsonb not null check(jsonb_typeof(embedding)='array'),
  quality jsonb not null default '{}',
  storage_path text not null unique,
  source_sha256 text not null check (source_sha256 ~ '^[0-9a-f]{64}$'),
  created_by uuid not null references public.profiles(id),
  created_at timestamptz not null default now(),
  check ((modality='face' and model='opencv-sface-2021dec-128' and jsonb_array_length(embedding)=128)
      or (modality='ear' and model='opencv-hog-64x128-3780-v1' and jsonb_array_length(embedding)=3780))
);
create index reference_person on public.reference_images(person_id);
create table public.matches (
  id uuid primary key default gen_random_uuid(),
  person_id uuid not null references public.persons(id),
  search_id uuid not null,
  score double precision not null check(score between -1 and 1),
  components jsonb not null,
  thresholds jsonb not null,
  status text not null default 'pending' check(status in ('pending','verified','rejected')),
  created_by uuid not null references public.profiles(id),
  created_at timestamptz not null default now(),
  unique(search_id,person_id)
);
create index matches_recent on public.matches(created_at desc);
create table public.decisions (
  id uuid primary key default gen_random_uuid(),
  match_id uuid not null unique references public.matches(id),
  actor_id uuid not null references public.profiles(id),
  decision text not null check(decision in ('verified','rejected')),
  reason text not null check(length(trim(reason)) between 3 and 2000),
  created_at timestamptz not null default now()
);
create table public.audit_events (
  id bigint generated always as identity primary key,
  actor_id uuid references public.profiles(id),
  action text not null,
  entity_id uuid not null,
  detail jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create function public.resq_role() returns text
language sql stable security definer set search_path=public,pg_temp as $$
  select role from public.profiles where id=auth.uid()
$$;
revoke all on function public.resq_role() from public;
grant execute on function public.resq_role() to authenticated;

alter table public.profiles enable row level security;
alter table public.persons enable row level security;
alter table public.reference_images enable row level security;
alter table public.matches enable row level security;
alter table public.decisions enable row level security;
alter table public.audit_events enable row level security;
create policy own_profile on public.profiles for select to authenticated using(id=auth.uid());
create policy member_cases on public.persons for select to authenticated using(public.resq_role() in ('rescuer','verifier','admin'));
create policy member_matches on public.matches for select to authenticated using(public.resq_role() in ('rescuer','verifier','admin'));
create policy member_decisions on public.decisions for select to authenticated using(public.resq_role() in ('rescuer','verifier','admin'));
create policy admin_audit on public.audit_events for select to authenticated using(public.resq_role()='admin');

-- Browser/mobile clients have no write permissions or access to embeddings/private objects.
revoke all on public.profiles,public.persons,public.reference_images,public.matches,public.decisions,public.audit_events from anon,authenticated;
grant select on public.profiles,public.persons,public.matches,public.decisions,public.audit_events to authenticated;
grant all on public.profiles,public.persons,public.reference_images,public.matches,public.decisions,public.audit_events to service_role;
grant usage,select on sequence public.audit_events_id_seq to service_role;

create function public.audit_created_record() returns trigger
language plpgsql security definer set search_path=public,pg_temp as $$
begin
  insert into public.audit_events(actor_id,action,entity_id) values(new.created_by,TG_TABLE_NAME||'.created',new.id);
  return new;
end $$;
create trigger case_created after insert on public.persons for each row execute function public.audit_created_record();
create trigger reference_created after insert on public.reference_images for each row execute function public.audit_created_record();
create trigger match_created after insert on public.matches for each row execute function public.audit_created_record();

create function public.prevent_evidence_edit() returns trigger
language plpgsql set search_path=public,pg_temp as $$
begin raise exception 'Recorded evidence cannot be edited or deleted' using errcode='RQ409'; end $$;
create trigger audit_immutable before update or delete on public.audit_events for each row execute function public.prevent_evidence_edit();
create trigger decision_immutable before update or delete on public.decisions for each row execute function public.prevent_evidence_edit();

-- A database transaction serializes decisions, writes the review and appends its audit event.
create function public.review_match(p_match_id uuid,p_actor uuid,p_decision text,p_reason text) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare current_match public.matches;
begin
  if not exists(select 1 from public.profiles where id=p_actor and role in ('verifier','admin')) then
    raise exception 'Reviewer role required' using errcode='RQ403';
  end if;
  if p_decision not in ('verified','rejected') or length(trim(coalesce(p_reason,''))) not between 3 and 2000 then
    raise exception 'Invalid decision or reason' using errcode='22023';
  end if;
  select * into current_match from public.matches where id=p_match_id for update;
  if not found then raise exception 'Candidate not found' using errcode='RQ404'; end if;
  if current_match.status<>'pending' then raise exception 'Candidate already reviewed' using errcode='RQ409'; end if;
  update public.matches set status=p_decision where id=p_match_id;
  insert into public.decisions(match_id,actor_id,decision,reason) values(p_match_id,p_actor,p_decision,trim(p_reason));
  insert into public.audit_events(actor_id,action,entity_id,detail) values(p_actor,'match.reviewed',p_match_id,jsonb_build_object('decision',p_decision));
  return jsonb_build_object('id',p_match_id,'status',p_decision);
end $$;

create function public.assign_team_role(p_user_id uuid,p_actor uuid,p_role text) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare previous_role text;
begin
  if not exists(select 1 from public.profiles where id=p_actor and role='admin') then
    raise exception 'Administrator required' using errcode='RQ403';
  end if;
  if p_role not in ('pending','rescuer','verifier') or p_user_id=p_actor then
    raise exception 'Role change is not permitted' using errcode='RQ403';
  end if;
  select role into previous_role from public.profiles where id=p_user_id for update;
  if not found then raise exception 'Profile not found' using errcode='RQ404'; end if;
  if previous_role='admin' then raise exception 'Administrator roles require owner configuration' using errcode='RQ403'; end if;
  update public.profiles set role=p_role where id=p_user_id;
  insert into public.audit_events(actor_id,action,entity_id,detail) values(p_actor,'profile.role_changed',p_user_id,jsonb_build_object('previous',previous_role,'role',p_role));
  return jsonb_build_object('id',p_user_id,'role',p_role);
end $$;
revoke all on function public.review_match(uuid,uuid,text,text),public.assign_team_role(uuid,uuid,text) from public,anon,authenticated;
grant execute on function public.review_match(uuid,uuid,text,text),public.assign_team_role(uuid,uuid,text) to service_role;
-- Trigger functions cannot be called as ordinary client RPCs.
revoke all on function public.create_resq_profile(),public.audit_created_record(),public.prevent_evidence_edit() from public,anon,authenticated;

insert into storage.buckets(id,name,public,file_size_limit,allowed_mime_types)
values('case-photos','case-photos',false,10485760,array['image/jpeg']);
-- No storage.objects policies are granted: authenticated users obtain short-lived URLs through the API.
commit;
