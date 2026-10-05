-- Expand the local avatar catalog without changing existing choices or access.
begin;
set local lock_timeout = '3s';
set local statement_timeout = '30s';
alter table public.profiles
  drop constraint if exists profiles_avatar_color_check,
  drop constraint if exists profiles_avatar_shape_check,
  add constraint profiles_avatar_color_check
    check (avatar_color in ('violet','cyan','blue','amber','emerald','coral','rose','indigo','lime','silver')) not valid,
  add constraint profiles_avatar_shape_check
    check (avatar_shape in ('squircle','circle','hexagon','shield','diamond','octagon','rounded-square','pentagon','star')) not valid;
commit;

-- Validate old rows after releasing the exclusive schema lock.
begin;
set local lock_timeout = '3s';
set local statement_timeout = '30s';
alter table public.profiles
  validate constraint profiles_avatar_color_check,
  validate constraint profiles_avatar_shape_check;
commit;
