-- Add status column to public.persons for case lifecycle management.
begin;
set local lock_timeout = '3s';
set local statement_timeout = '30s';

alter table public.persons
  add column if not exists status text not null default 'pending';

alter table public.persons
  drop constraint if exists persons_status_check,
  add constraint persons_status_check
    check (status in ('pending', 'urgent', 'ongoing', 'completed', 'closed')) not valid;

create index if not exists persons_status_idx on public.persons(status);

commit;

-- Validate constraint after releasing schema lock
begin;
set local lock_timeout = '3s';
set local statement_timeout = '30s';
alter table public.persons
  validate constraint persons_status_check;
commit;
