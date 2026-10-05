-- Owner explicitly designated by the project owner. No password or account is created here.
begin;
create schema if not exists resq_private;
revoke all on schema resq_private from public,anon,authenticated,service_role;
create table resq_private.owner_bootstrap (
  email text primary key,
  consumed_by uuid,
  consumed_at timestamptz
);
revoke all on resq_private.owner_bootstrap from public,anon,authenticated,service_role;
insert into resq_private.owner_bootstrap(email) values('eswarch2004y@gmail.com');
create function resq_private.approve_verified_owner() returns trigger
language plpgsql security definer set search_path=pg_catalog,pg_temp as $$
declare owner_email text;
begin
  if new.email_confirmed_at is null then return new; end if;
  select email into owner_email from resq_private.owner_bootstrap
    where email=lower(new.email) and consumed_at is null for update;
  if not found then return new; end if;
  update public.profiles set role='admin' where id=new.id;
  if not found then raise exception 'Owner profile missing'; end if;
  insert into public.audit_events(actor_id,action,entity_id,detail)
    values(new.id,'owner.bootstrapped',new.id,'{"source":"verified_owner_allowlist"}'::jsonb);
  update resq_private.owner_bootstrap set consumed_by=new.id,consumed_at=now() where email=owner_email;
  return new;
end $$;
revoke all on function resq_private.approve_verified_owner() from public,anon,authenticated,service_role;
-- PostgreSQL runs same-event triggers alphabetically; signup creates the profile first.
create trigger z_resq_verified_owner after insert or update of email_confirmed_at on auth.users
  for each row execute function resq_private.approve_verified_owner();
-- Also covers the designated owner if they confirmed their email before this migration.
update auth.users set email_confirmed_at=email_confirmed_at
  where lower(email)='eswarch2004y@gmail.com' and email_confirmed_at is not null;
commit;
