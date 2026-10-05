-- ResQ Avatar System: 10 Snapchat-style 3D Bitmoji options, 2D/3D shapes, and vibrant color palettes
begin;

-- Update avatar_key check constraint to allow all 10 avatars plus legacy fallbacks
alter table public.profiles drop constraint if exists profiles_avatar_key_check;
alter table public.profiles
  add constraint profiles_avatar_key_check
  check (avatar_key in (
    'scout','tech','pilot','medic','ranger','lead','cadet','analyst','diver','spark',
    'short-hair','curly-hair','tied-hair'
  ));

-- Update avatar_color check constraint to include emerald and coral
alter table public.profiles drop constraint if exists profiles_avatar_color_check;
alter table public.profiles
  add constraint profiles_avatar_color_check
  check (avatar_color in ('violet','cyan','blue','amber','emerald','coral'));

-- Add avatar_shape column for 2D/3D frame shapes
alter table public.profiles
  add column if not exists avatar_shape text not null default 'squircle'
    check (avatar_shape in ('squircle','circle','hexagon','shield','diamond','octagon'));

-- Add avatar_style column for 2D vs 3D styling
alter table public.profiles
  add column if not exists avatar_style text not null default '3d'
    check (avatar_style in ('3d','2d'));

commit;
