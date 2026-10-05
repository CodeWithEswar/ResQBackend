import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url = os.environ.get("SUPABASE_URL")
pub_key = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("SUPABASE_KEY")
service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

print(f"Connecting to Supabase: {url}")
if not url or not pub_key:
    print("[ERROR] SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY is missing from .env")
    exit(1)

# 1. Test Publishable Key Client
try:
    client: Client = create_client(url, pub_key)
    print("[OK] Public client initialized.")
except Exception as err:
    print(f"[ERROR] Public client failed: {err}")

# 2. Test Service Role Key Client
if service_key and not service_key.startswith("YOUR_"):
    try:
        admin_client: Client = create_client(url, service_key)
        profiles = admin_client.table("profiles").select("id", count="exact").execute()
        persons = admin_client.table("persons").select("id", count="exact").execute()
        print(f"[OK] Service role client active! Database tables verified:")
        print(f"     - profiles: {profiles.count} rows")
        print(f"     - persons:  {persons.count} rows")
        
        # Check storage bucket
        buckets = admin_client.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        if "case-photos" in bucket_names:
            print("     - storage bucket 'case-photos': [OK]")
        else:
            print(f"     - storage buckets found: {bucket_names}")
    except Exception as admin_err:
        print(f"[ERROR] Service role query failed: {admin_err}")
else:
    print("[INFO] SUPABASE_SERVICE_ROLE_KEY is not configured yet.")
