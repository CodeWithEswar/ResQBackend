# ResQ face evidence backend

This service runs actual YOLO face detection, YuNet landmark alignment and pretrained SFace recognition on CPU. It compares a selected face with enrolled case references, returning pending candidates for independent review. It does not automatically establish identity.

The case registry lives in the configured Supabase project. No reference photographs means no candidates. Query photos are decoded in memory and are not saved. Reference photographs stay in private storage, accessible through short-lived signed links and staff roles.

## Run locally

```powershell
./setup.ps1
./start.ps1
```

Configure server-only Supabase credentials in `.env`. The frontend uses a user access token; it must never receive the service-role key. Existing Supabase migrations remain applicable: face vectors are still normalized SFace descriptors of length 128. Startup no longer kills a process occupying the port or automatically adds a wildcard CORS origin. Set `RESQ_PORT` to choose another port and `RESQ_RELOAD=1` to opt into development code reload.

```powershell
./.venv/Scripts/python.exe -m ml.check_backend --remote
./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
```

`/health` reports the actual detector hash, local training status, calibration status and disaster validation status. Startup verifies all three model hashes and warms YOLO before accepting requests. Missing or corrupt models fail startup; there is no synthetic inference fallback. Inference dependencies are installed at setup, never on a user upload. The API limits concurrent inference and returns a retryable 503 when its queue timeout expires.

## API flow

Google profile data uses `display_name`, falling back to Google's `full_name`/`name`, and `avatar_url`/`picture`. Migration `20261004181845_resq_google_profile_sync.sql` fills profile names, email and provider photographs on signup and Auth updates, and persists edited names and avatar catalog choices to Auth metadata in the same transaction. Provider photos are stored separately from the chosen 3D avatar. Metadata never determines workspace roles. The API also repairs blank legacy names using a verified Auth response and a conditional update that preserves concurrent edits.

For existing Google accounts, `python -m ml.backfill_profiles` reports a dry run; add `--apply` to repair missing metadata with server credentials. It prints counts only, preserves custom names/avatars/roles, and paginates Auth users. Apply migrations through an authenticated Supabase SQL connection or SQL Editor before relying on automatic synchronization. `npm run test:db` verifies the real trigger SQL in embedded PostgreSQL; `python -m pytest tests/test_profile_sync.py -q` checks authenticated name repair and concurrent edits.

1. `POST /api/faces/detect` with multipart `image`: returns bounding boxes, deterministic left-to-right indexes and quality issues. It processes a group photograph without assigning identities.
2. `POST /api/cases/{id}/references` with `modality=face`, `image` and optional `face_index`: enrolls the chosen face. An index is required if more than one face is detected.
3. `POST /api/search` with `face` and optional `face_index`: ranks face references with exact cosine similarity. Scores are not probabilities. Results remain pending until a verifier records independent evidence.
4. `POST /api/faces/video` with multipart `video`: samples the entire clip and returns numbered face frames with timestamps. Default limits are 50 MB, 60 seconds, 30 samples and eight returned frames.
5. `WS /api/faces/stream`: authenticated live camera frames with Redis session leases and frame pacing. The app sends one frame at a time and shows the actual annotated response. Redis must be reachable; no synthetic or in-memory live fallback is used.

Full HTTP and WebSocket integration contracts, error codes and examples are in [API.md](API.md). Image detection now also returns clean and annotated JPEG previews. Query video files are decoded in private temporary directories and removed after analysis. Redis stores only short-lived session coordination, never camera images or face embeddings.

Local development is configured on port 8001 because the previous process still occupies 8000. `start.py` reads `RESQ_PORT` from the backend environment and updates the app's LAN endpoint. The camera and image-conversion Expo modules require a new native development/release build. Web camera capture requires a secure context (HTTPS or localhost); uploaded photos/videos work without camera access.

The app exposes numbered face selection for enrollment and search. Small, blurred, badly lit or unalignable faces are rejected. The legacy `cropped`/`face_cropped` parameters remain accepted for older clients but cannot bypass detection or alignment.

Ear images may be stored for human inspection. The old HOG descriptor is retained for database compatibility, but it has no trained identity model and cannot create or raise an identity candidate. Ear-only searches return a descriptive 422 response.

## Datasets and reproducible training

Published initialization: [akanametov/yolo-face](https://github.com/akanametov/yolo-face). Detection data: [CUHK WIDER FACE](https://huggingface.co/datasets/CUHK-CSE/wider_face). Verification data: [LFW through scikit-learn's checksummed archive definitions](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py).

```powershell
./.venv/Scripts/python.exe -m ml.fetch_assets --dataset
./.venv/Scripts/python.exe -m ml.prepare_wider
./.venv/Scripts/python.exe -m ml.train_detector --epochs 100 --device cpu --name wider-full --promote
```

With a CUDA-capable machine and CUDA PyTorch, use `--device 0`. The local environment is CPU PyTorch. Full training may take many hours on a laptop.

For the bounded local run performed while implementing this change:

```powershell
./.venv/Scripts/python.exe -m ml.prepare_wider --output data/wider/video-1024 --train-limit 1024 --val-limit 256
./.venv/Scripts/python.exe -m ml.train_detector --data data/wider/video-1024/data.yaml --weights models/resq-face-289b57fb99f0.pt --epochs 3 --batch 8 --imgsz 640 --device cpu --threads 4 --name wider-video-1024 --lr 0.00001 --freeze 10 --promote
```

The converter uses official bounding-box annotations, filters invalid annotations, clips boxes, samples across event categories, deduplicates identical images and writes explicit split lists. The published initialization has an auxiliary five-landmark branch; the training script copies its 355 shared detection tensors into a one-class detection model. WIDER box-only data does not invent landmark targets. Training varies lighting, scale, position and rotation.

Every run writes `runs/<name>/evaluation.json` with original and candidate precision, recall, mAP, dataset hashes and training counts. A candidate can replace deployed weights only if it meets minimum detection scores and does not materially regress from initialization. Failed gates preserve the currently deployed model. WIDER's validation split was also used by the published initialization; these measurements are a regression benchmark, not evidence of generalization to unseen disaster imagery.

The completed three-epoch run used 1,024 training images with 10,663 annotated faces and 256 validation images with 2,497 faces. The promoted detector is `resq-face-495d4fa74021.pt`. Validation recall rose from 0.6207 to 0.6243 and mAP50 from 0.6976 to 0.6986; mAP50–95 changed from 0.3926 to 0.3890. These are modest changes, with the full comparison recorded in `runs/wider-video-1024/evaluation.json`. Replacing a detector invalidates its previous recognition calibration until fresh evaluation completes.

## Recognition evaluation

```powershell
./.venv/Scripts/python.exe -m ml.fetch_lfw
./.venv/Scripts/python.exe -m ml.evaluate_lfw --train-per-class 250 --test-per-class 0 --promote
```

The calibration script uses genuine and impostor development pairs. LFW identifies the central portrait even when background faces remain; its published evaluation region supplies explicit face-selection metadata. This measures the same selected-face flow used by the app. It does not identify everyone in a group without a selection. The script excludes calibration subjects from evaluation, records detection/quality failures and reports true-accept rate, false-accept rate and finite-sample uncertainty. It never calibrates against same-image copies or lowers the threshold below the published SFace cosine reference of 0.363. Rejected photographs are included in coverage figures and are excluded from the conditional recognition denominators. Results are saved in `work/lfw-evaluation.json` and in the deployed manifest if promotion gates pass. Re-run calibration after replacing detector weights.

The new detector was recalibrated on 500 development pairs and evaluated on 1,000 separate, subject-disjoint test pairs. At threshold 0.363, true-accept rate was 0.972 on 500 genuine pairs, with zero false accepts among 499 usable impostor pairs. One blurred pair was rejected. The 95% upper bound for false-accept rate is 0.00764, so zero observed errors does not imply zero risk. Research calibration was promoted; disaster validation remains false.

## Model files and deployment

`models/pipeline.pkl` is a versioned, primitive-only pickle manifest that the running API actually reads. It records the active YOLO checkpoint, ONNX recognition/alignment models, hashes, preprocessing quality limits, threshold and evaluation evidence. An equivalent `pipeline.json` makes it reviewable. A restricted unpickler forbids executable Python classes and persistent references. There is no model-upload endpoint.

YOLO tensors stay in their native `.pt` checkpoint format; SFace and YuNet stay in ONNX. Renaming these to `.pkl` would not convert them into a different trained model. Only operator-controlled, verified model files are loaded.

```powershell
./.venv/Scripts/python.exe -m ml.package_model
```

Deploy the complete `work/resq-model.zip` contents to `MODELS_DIR` and install the pinned dependencies. Model files have immutable release filenames; the manifest is replaced last. The API verifies and warms a changed manifest before atomically activating it on the next inference or health request. Each request retains its original snapshot. An invalid release returns 503 rather than silently serving mismatched weights. The archive contains the active weights and manifest, plus recognition/alignment files and licenses. It excludes training datasets and case photographs. Keep model hashes and evaluation reports with each release.

## Practical deployment limits

WIDER FACE and LFW are research benchmarks; neither establishes reliability on injured, deceased, heavily occluded or low-resolution disaster victims. The API explicitly reports `disaster_validated=false`. Real field deployment still requires authorized, labeled, representative disaster data, separate held-out evaluation, threshold validation at the intended registry size, and operational load testing. Tiny faces may be detected but cannot be safely matched.

WIDER FACE is CC BY-NC-ND 4.0; its research use does not grant unrestricted commercial rights. The source face checkpoint repository uses GPL-3.0, Ultralytics uses AGPL-3.0 or its enterprise license, SFace uses Apache-2.0, and YuNet uses MIT. Verify the applicable rights for the intended deployment. Public dataset photographs are not inserted into the live missing-person registry.
