-- Persist only choices from the app's local avatar catalog, never user image uploads.
begin;
alter table public.profiles
  add column if not exists avatar_key text not null default 'short-hair',
  add column if not exists avatar_color text not null default 'violet';
-- Existing RLS and server-only write permissions remain in force.
commit;
