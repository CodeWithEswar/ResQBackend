"""
ResQ Backend Startup Script
Runs the FastAPI server accessible from internal (localhost) and external (mobile/LAN) devices.
Checks port availability and configures local mobile development access.
"""
import os
import socket
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
FRONTEND_ENV = ROOT_DIR / "frontend" / ".env"
BACKEND_ENV = BACKEND_DIR / ".env"
load_dotenv(BACKEND_ENV)

PORT = int(os.getenv('RESQ_PORT','8000'))
HOST = os.getenv('RESQ_HOST','0.0.0.0')

def get_lan_ip() -> str:
    """Detect the local machine's primary Wi-Fi / LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Route determination without sending actual packets
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def free_port(port: int):
    """Check availability without terminating unrelated processes."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:s.bind((HOST,port))
        except OSError as error:
            raise SystemExit(f'Cannot bind {HOST}:{port}. Stop the existing ResQ server or set RESQ_PORT to an available port. {error}') from None

def sync_frontend_api_url(lan_ip: str):
    """Automatically ensure frontend/.env points to this machine's LAN IP so phone connects."""
    if not FRONTEND_ENV.exists():
        return
    
    try:
        content = FRONTEND_ENV.read_text(encoding="utf-8")
        target_line = f"EXPO_PUBLIC_MODEL_API_URL=http://{lan_ip}:{PORT}"
        
        lines = content.splitlines()
        updated = False
        new_lines = []
        for line in lines:
            if line.startswith("EXPO_PUBLIC_MODEL_API_URL="):
                new_lines.append(target_line)
                updated = True
            else:
                new_lines.append(line)
        
        if not updated:
            new_lines.append(target_line)
            
        FRONTEND_ENV.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception as err:
        print(f"[!] Warning syncing frontend .env: {err}")

def update_cors_lan_ip(lan_ip: str):
    """Ensure CORS allows external mobile and web clients."""
    if not BACKEND_ENV.exists():
        return
    try:
        content = BACKEND_ENV.read_text(encoding="utf-8")
        if "CORS_ORIGINS" in content:
            lines = content.splitlines()
            new_lines = []
            for line in lines:
                if line.startswith("CORS_ORIGINS="):
                    # Add only local preview origins, without widening to every website.
                    origins = [o.strip() for o in line.split("=", 1)[1].split(",") if o.strip()]
                    for allowed in [f"http://{lan_ip}:8081", f"http://{lan_ip}:8082", f"http://{lan_ip}:19006", "http://localhost:8081", "http://localhost:8082", "http://127.0.0.1:8081", "http://127.0.0.1:8082"]:
                        if allowed not in origins:
                            origins.append(allowed)
                    new_lines.append("CORS_ORIGINS=" + ",".join(origins))
                else:
                    new_lines.append(line)
            BACKEND_ENV.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception as err:
        print(f"[!] Warning updating CORS: {err}")

def main():
    os.chdir(BACKEND_DIR)
    
    # 1. Clean port 8000
    free_port(PORT)
    
    # 2. Detect Network IP
    lan_ip = get_lan_ip()
    sync_frontend_api_url(lan_ip)
    update_cors_lan_ip(lan_ip)

    # 3. Print access dashboard
    print("=" * 65)
    print("             ResQ Model & Biometrics Backend Server")
    print("=" * 65)
    print(f" [*] Local (Internal / This PC):   http://localhost:{PORT}")
    print(f"                                   http://127.0.0.1:{PORT}")
    print(f" [*] Network (External / Phone):   http://{lan_ip}:{PORT}")
    print(f" [*] Swagger API Documentation:    http://{lan_ip}:{PORT}/docs")
    print("-" * 65)
    print(f" [OK] Frontend auto-configured:     EXPO_PUBLIC_MODEL_API_URL=http://{lan_ip}:{PORT}")
    print(" [*] Model validation and warmup must finish before requests are accepted.")
    print("=" * 65)
    reload_enabled=os.getenv('RESQ_RELOAD','1')=='1'
    print(f"\nStarting uvicorn (reload={reload_enabled}; Press Ctrl+C to stop)...\n")

    # 4. Start uvicorn
    import uvicorn
    from app.config import Settings
    limits=Settings()
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=reload_enabled,
                ws_max_size=limits.live_frame_bytes*4//3+4096, ws_max_queue=2, ws_per_message_deflate=False)

if __name__ == "__main__":
    main()
