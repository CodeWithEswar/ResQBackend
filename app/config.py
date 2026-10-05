import os
from dataclasses import dataclass,field
from pathlib import Path
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')

@dataclass(frozen=True)
class Settings:
    url: str=field(default_factory=lambda:os.getenv('SUPABASE_URL','').rstrip('/'))
    public_key: str=field(default_factory=lambda:os.getenv('SUPABASE_PUBLISHABLE_KEY') or os.getenv('SUPABASE_KEY', ''))
    service_key: str=field(default_factory=lambda:os.getenv('SUPABASE_SERVICE_ROLE_KEY',''))
    models: Path=field(default_factory=lambda:ROOT/os.getenv('MODELS_DIR','models'))
    max_bytes: int=field(default_factory=lambda:int(os.getenv('MAX_UPLOAD_BYTES','10485760')))
    max_pixels: int=field(default_factory=lambda:int(os.getenv('MAX_IMAGE_PIXELS','20000000')))
    max_video_bytes: int=field(default_factory=lambda:int(os.getenv('MAX_VIDEO_BYTES','52428800')))
    max_video_seconds: float=field(default_factory=lambda:float(os.getenv('MAX_VIDEO_SECONDS','60')))
    video_processing_seconds: float=field(default_factory=lambda:float(os.getenv('VIDEO_PROCESSING_SECONDS','45')))
    redis_url: str=field(default_factory=lambda:os.getenv('REDIS_URL','redis://127.0.0.1:6379/0'))
    live_interval_ms: int=field(default_factory=lambda:max(250,int(os.getenv('LIVE_INTERVAL_MS','1000'))))
    live_frame_bytes: int=field(default_factory=lambda:int(os.getenv('MAX_LIVE_FRAME_BYTES','2097152')))
    live_session_seconds: int=field(default_factory=lambda:int(os.getenv('LIVE_SESSION_SECONDS','600')))
    inference_concurrency: int=field(default_factory=lambda:int(os.getenv('INFERENCE_CONCURRENCY','2')))
    inference_queue_seconds: float=field(default_factory=lambda:float(os.getenv('INFERENCE_QUEUE_SECONDS','10')))
    # HTTP and WebSocket origins use the same explicit list. A literal wildcard
    # cannot be matched by the live origin check and must not widen HTTP access.
    cors: list[str]=field(default_factory=lambda:[s.strip() for s in os.getenv('CORS_ORIGINS','http://localhost:8081,http://127.0.0.1:8081').split(',') if s.strip() and s.strip()!='*'])

    @property
    def configured(self):
        values=[self.url,self.public_key,self.service_key]
        return all(values) and not any('YOUR_' in v for v in values) and self.url.startswith('https://')
