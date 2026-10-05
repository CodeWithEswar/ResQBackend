# ResQ detection API integration

The running service exposes version 2.1 at the configured `RESQ_PORT`. Local development now uses `http://127.0.0.1:8001`; `start.ps1` also updates the mobile app's LAN address. `/docs` describes the HTTP request and response schemas. `/health` reports model hashes, training/calibration, Redis connectivity and resource limits.

All detection and search requests require a verified Supabase user with an approved rescuer, verifier or administrator role. Supply `Authorization: Bearer <access_token>` for HTTP. An API key or service-role credential is not a user session.

## Images

`POST /api/faces/detect`, multipart fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `image` | Yes | JPEG, PNG or WebP, up to 10 MB and 20 million pixels |
| `include_frame` | No | `true` by default; set `false` for boxes without image previews |

The response contains `width`, `height`, `faces`, `selection_required`, `detector_version`, `processing_ms` and `frame`. Each face has a frame-local `index`, pixel-coordinate `[left, top, right, bottom]` box, detection confidence, quality measurements, `usable` flag and explanations.

`frame.image` is the clean image and `frame.annotated_image` is its numbered face-box preview. Both are JPEG data URIs, with actual dimensions in `frame.width` and `frame.height`. A large source image may have a smaller preview. Position client overlays as fractions of the top-level detection width/height. EXIF orientation is corrected. Confidence describes detection, not identity probability.

## Uploaded video

`POST /api/faces/video`, multipart fields:

| Field | Default | Limits |
| --- | --- | --- |
| `video` | Required | MP4/MOV with an `ftyp` container, AVI or WebM; decodable video codec; 50 MB; 60 seconds |
| `interval_seconds` | `1` | `0.5`–`10` seconds requested sampling interval |
| `max_frames` | `30` | `1`–`60` decoded samples |
| `result_limit` | `8` | `1`–`12` returned frames containing faces |

Samples are distributed across the full clip. When a limit forces sparser sampling, the response explains it. This is sampled analysis; a person visible only between samples may be missed. `frames_with_faces` counts sample frames, not unique people. Returned frames prioritize usable, sharper face crops and are ordered by timestamp. Frames with detected but unusable faces remain visible with quality explanations.

The response provides `duration_seconds`, `fps`, `total_frames`, `sampled_frames`, `frames_with_faces`, `sampling_interval_seconds`, `processing_ms`, `warnings`, and `frames`. Each frame has the image-detection fields plus `frame_index` and `timestamp_seconds`. Show `frame.annotated_image` to users, and use the clean `frame.image` for subsequent comparison. Temporary uploads are removed after success, failure and processing timeout. No video or detection image is inserted into Supabase.

```javascript
const form = new FormData();
form.append('video', selectedFile);
form.append('max_frames', '30');
const response = await fetch(`${apiUrl}/api/faces/video`, {
  method: 'POST', headers: { Authorization: `Bearer ${accessToken}` }, body: form,
});
const result = await response.json();
if (!response.ok) throw new Error(result.detail);
// Display result.frames[n].frame.annotated_image and its timestamp_seconds.
```

## Live WebSocket

Connect to `ws://127.0.0.1:8001/api/faces/stream` locally, or `wss://<your-api-host>/api/faces/stream` behind production TLS. The token is sent in the first message, never in the URL. An Origin must match the explicit `CORS_ORIGINS` list or the exact HTTP(S) origin of the WebSocket endpoint, including its port. This supports Android React Native, which automatically sends the endpoint origin. Literal wildcard entries are ignored. Native clients without an Origin header authenticate identically. Every connection still requires a valid user token and an approved workspace role.

1. Send `{"type":"auth","token":"<Supabase access token>"}` within 10 seconds.
2. Wait for `{"type":"ready", ...}`. It includes `session_id`, `min_interval_ms`, `max_frame_bytes`, `session_seconds`, `transport:"websocket"` and `coordination:"redis"`.
3. Send a binary JPEG/PNG/WebP frame, or the JSON envelope below. Wait for its response before sending another, and respect `min_interval_ms` (1,000 by default). Maximum decoded frame size is 2 MB by default.
4. A `type:"detection"` message contains the image-detection fields plus echoed `sequence` and optional `captured_at_ms`. Display its annotated image. Face indexes apply only to that frame; they do not track identities across frames.
5. Send `{"type":"stop"}` or close the connection to release the session. `{"type":"ping"}` renews the lease and returns `{"type":"pong"}` while an idle camera is connected. Idle connections expire after 30 seconds; sessions last at most 10 minutes. Reconnect with a fresh user token afterward.

```javascript
const ws = new WebSocket(`${apiUrl.replace(/^http/, 'ws')}/api/faces/stream`);
ws.onopen = () => ws.send(JSON.stringify({ type: 'auth', token: accessToken }));
// After ready, with only one frame awaiting a response:
ws.send(JSON.stringify({
  type: 'frame', sequence: 0, captured_at_ms: Date.now(),
  image_base64: jpegBase64WithoutDataUriPrefix,
}));
// Or: ws.send(jpegArrayBuffer).
```

Errors have `type:"error"`, `status`, `message`, and the submitted `sequence` when available. A 429 also includes `retry_after_ms`. Authentication/access loss closes with 4401/4403; a duplicate or expired session uses 4409; a timeout uses 4408; unavailable Redis uses 1013. Oversized messages may be closed by the WebSocket transport with 1009. Supabase verifies access again at least every 60 seconds during active frame processing. Only trusted operator model artifacts are loaded.

Redis provides atomic session ownership, 45-second crash-recovery leases and frame pacing across workers. Namespaced, hashed account keys contain only a random session ID and short-lived pacing state. No tokens, photos, face embeddings or identities are cached. A Redis outage disables live streaming with an explicit 503; image/video detection continues. Configure `REDIS_URL` (use `rediss://` for a remote TLS deployment), `LIVE_INTERVAL_MS`, `MAX_LIVE_FRAME_BYTES` and `LIVE_SESSION_SECONDS` in the backend environment. Do not expose Redis directly to mobile clients.

`POST /api/faces/live` remains available to integrators sending individual multipart images. It accepts `image`, optional nonnegative `sequence` and `captured_at_ms`; the app uses the WebSocket transport.

## Compare a selected face

Decode a returned clean JPEG data URI into a file, then call `POST /api/search` with multipart `face`, selected `face_index` and `detector_version`. The app materializes the frame into its local image cache for native upload. Do not submit the annotated image. When a detector release changes, the server returns 409 and asks for detection again.

Detection does not enroll people. Real matches require actual authorized case-reference photographs. An empty registry correctly returns no candidates. Scores are cosine similarities and every returned lead requires independent review; video detection does not change this.

## Operational checks

```powershell
./.venv/Scripts/python.exe -m ml.check_backend --remote
./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
./start.ps1
```

Run the media integration tests with a real local Redis service, or set `REDIS_URL` to a dedicated test instance. Tests use real YOLO, SFace, OpenCV decoding and Redis; only the external Auth/database boundary is isolated. They do not populate the live case registry. The detector is trained on annotated WIDER FACE data and recognition is evaluated on LFW; these research datasets do not establish disaster-field identity accuracy. Training reports and model hashes remain in `runs/` and `models/pipeline.json`.
