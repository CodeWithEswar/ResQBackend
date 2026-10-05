-- Hosted Supabase grants privileges to service_role/anon on newly created functions.
-- Restrict helper functions explicitly, including grants inherited from platform defaults.
begin;
revoke all on function public.resq_role() from public,anon,service_role;
grant execute on function public.resq_role() to authenticated;
revoke all on function public.create_resq_profile(),public.audit_created_record(),public.prevent_evidence_edit() from public,anon,authenticated,service_role;
revoke all on function public.review_match(uuid,uuid,text,text),public.assign_team_role(uuid,uuid,text) from public,anon,authenticated;
grant execute on function public.review_match(uuid,uuid,text,text),public.assign_team_role(uuid,uuid,text) to service_role;
commit;
