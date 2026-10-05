"""Real model, video decoder, WebSocket protocol and Redis integration checks."""
import asyncio
import base64
from dataclasses import replace
from pathlib import Path
import time
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from redis import Redis
from starlette.websockets import WebSocketDisconnect

import app.main as service
from app.live import LiveSessions
from app.main import app, gateway, settings
from test_api import FakeGateway, auth, photo, ACTORS


@pytest.fixture(scope='module')
def media_client():
    fake = FakeGateway()
    app.dependency_overrides[gateway] = lambda: fake
    with TestClient(app) as client:
        app.state.live_sessions.prefix = 'resq:test:live:' + uuid4().hex
        yield client, fake
    app.dependency_overrides.pop(gateway, None)


@pytest.fixture(scope='module')
def clip(tmp_path_factory, media_client):
    path = tmp_path_factory.mktemp('video') / 'real-face.avi'
    source = cv2.imdecode(np.frombuffer(photo('face-reference.jpg'), np.uint8), cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = app.state.engine.extract_face(source)[1]['box']
    margin = round(max(x2 - x1, y2 - y1) * 0.6)
    source = source[max(0, y1 - margin):y2 + margin, max(0, x1 - margin):x2 + margin]
    height, width = source.shape[:2]
    scale = min(640 / width, 480 / height)
    sample = cv2.resize(source, (round(width * scale), round(height * scale)))
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:sample.shape[0], :sample.shape[1]] = sample
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (640, 480))
    assert writer.isOpened(), 'The installed OpenCV must support MJPEG encoding for the decoder test'
    for index in range(20):
        writer.write(np.zeros_like(frame) if index < 8 else frame)
    writer.release()
    return path.read_bytes()


def decoded(uri):
    return cv2.imdecode(np.frombuffer(base64.b64decode(uri.split(',', 1)[1]), np.uint8), cv2.IMREAD_COLOR)


def test_image_upload_returns_real_displayable_detection_and_no_persistence(media_client):
    client, fake = media_client
    response = client.post('/api/faces/detect', files={'image': ('face.jpg', photo('face-reference.jpg'), 'image/jpeg')}, headers=auth('rescuer'))
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result['faces']) == 1 and result['faces'][0]['usable']
    frame = result['frame']
    original, annotated = decoded(frame['image']), decoded(frame['annotated_image'])
    assert original.shape == annotated.shape == (frame['height'], frame['width'], 3)
    assert np.count_nonzero(original != annotated) > 100
    assert result['processing_ms'] >= 0 and len(result['detector_version']) == 64
    assert not fake.objects and not fake.tables['reference_images']
    response = client.post('/api/faces/detect', data={'include_frame': 'false'}, files={'image': ('face.jpg', photo('face-reference.jpg'))}, headers=auth('rescuer'))
    assert response.json()['frame'] is None


def test_rotated_phone_photo_decodes_to_upright_frame(media_client):
    import io
    original = Image.open(io.BytesIO(photo('face-reference.jpg')))
    rotated = original.transpose(Image.Transpose.ROTATE_90)
    exif = rotated.getexif(); exif[274] = 6
    buffer = io.BytesIO(); rotated.save(buffer, 'JPEG', exif=exif, quality=95)
    response = media_client[0].post('/api/faces/detect', files={'image': ('phone.jpg', buffer.getvalue())}, headers=auth('rescuer'))
    assert response.status_code == 200
    result = response.json()
    assert (result['width'], result['height']) == original.size
    assert len(result['faces']) == 1


def test_video_scans_full_timeline_returns_frames_and_selected_frame_can_search(media_client, clip):
    client, fake = media_client
    response = client.post('/api/faces/video', data={'interval_seconds': '0.5', 'max_frames': '6', 'result_limit': '2'}, files={'video': ('clip.avi', clip, 'video/x-msvideo')}, headers=auth('rescuer'))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['duration_seconds'] == 4 and result['total_frames'] == 20 and result['sampled_frames'] == 6
    assert result['frames_with_faces'] >= 2 and len(result['frames']) == 2 and result['warnings']
    assert all(frame['timestamp_seconds'] >= 1.6 for frame in result['frames'])
    assert result['frames'][0]['frame_index'] < result['frames'][1]['frame_index']
    chosen = result['frames'][-1]
    assert chosen['faces'][0]['usable'] and decoded(chosen['frame']['image']).shape[:2] == (480, 640)
    jpeg = base64.b64decode(chosen['frame']['image'].split(',', 1)[1])
    response = client.post('/api/search', data={'face_index': '0', 'detector_version': chosen['detector_version']}, files={'face': ('frame.jpg', jpeg)}, headers=auth('rescuer'))
    assert response.status_code == 200, response.text
    assert response.json()['candidates'] == [] and not fake.objects


def test_video_limits_permissions_and_invalid_uploads(media_client, clip, monkeypatch):
    client, _ = media_client
    assert client.post('/api/faces/video', files={'video': ('x.avi', clip)}).status_code == 401
    assert client.post('/api/faces/video', files={'video': ('x.avi', clip)}, headers=auth('pending')).status_code == 403
    assert client.post('/api/faces/video', files={'video': ('x.m3u8', b'#EXTM3U\nhttp://127.0.0.1/private')}, headers=auth('rescuer')).status_code == 415
    assert client.post('/api/faces/video', files={'video': ('x.mp4', b'\x00\x00\x00\x18ftypisom' + b'x' * 128)}, headers=auth('rescuer')).status_code == 415
    assert client.post('/api/faces/video', data={'max_frames': '61'}, files={'video': ('x.avi', clip)}, headers=auth('rescuer')).status_code == 422
    monkeypatch.setattr(service, 'settings', replace(settings, max_video_bytes=100))
    assert client.post('/api/faces/video', files={'video': ('x.avi', clip)}, headers=auth('rescuer')).status_code == 413
    monkeypatch.setattr(service, 'settings', replace(settings, max_video_seconds=1))
    assert client.post('/api/faces/video', files={'video': ('x.avi', clip)}, headers=auth('rescuer')).status_code == 413
    monkeypatch.setattr(service, 'settings', replace(settings, video_processing_seconds=0))
    assert client.post('/api/faces/video', files={'video': ('x.avi', clip)}, headers=auth('rescuer')).status_code == 503


def test_oversized_request_is_rejected_before_multipart_spooling(media_client):
    origin = settings.cors[0]
    response = media_client[0].post('/api/faces/video', content=b'not-a-video', headers={
        **auth('rescuer'), 'Origin': origin, 'Content-Type': 'multipart/form-data; boundary=qa',
        'Content-Length': str(max(settings.max_video_bytes, settings.max_bytes) + 1048577)})
    assert response.status_code == 413
    assert response.headers['access-control-allow-origin'] == origin


def test_video_decoder_releases_temp_files_on_errors(media_client, clip, monkeypatch, tmp_path):
    import app.media as media
    real_temporary = media.tempfile.TemporaryDirectory
    paths = []
    def temporary(**kwargs):
        folder = real_temporary(dir=tmp_path, **kwargs); paths.append(Path(folder.name)); return folder
    monkeypatch.setattr(media.tempfile, 'TemporaryDirectory', temporary)
    monkeypatch.setattr(service, 'settings', replace(settings, max_video_seconds=1))
    response = media_client[0].post('/api/faces/video', files={'video': ('x.avi', clip)}, headers=auth('rescuer'))
    assert response.status_code == 413 and paths and all(not path.exists() for path in paths)


def test_live_rest_frame_returns_sequence_and_boxes(media_client):
    response = media_client[0].post('/api/faces/live', data={'sequence': '17', 'captured_at_ms': '123456'}, files={'image': ('face.jpg', photo('face-reference.jpg'))}, headers=auth('rescuer'))
    assert response.status_code == 200
    result = response.json()
    assert result['sequence'] == 17 and result['captured_at_ms'] == 123456 and len(result['faces']) == 1 and result['frame']


def test_websocket_auth_permission_origin_and_duplicate_session(media_client):
    client, _ = media_client
    for token, status in [('bad', 401), ('pending', 403)]:
        with client.websocket_connect('/api/faces/stream') as socket:
            socket.send_json({'type': 'auth', 'token': token})
            assert socket.receive_json()['status'] == status
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/api/faces/stream', headers={'origin': 'https://unapproved.invalid'}):
            pass
    with client.websocket_connect('/api/faces/stream') as first:
        first.send_json({'type': 'auth', 'token': 'rescuer'})
        ready = first.receive_json()
        assert ready['type'] == 'ready' and ready['coordination'] == 'redis'
        with client.websocket_connect('/api/faces/stream') as second:
            second.send_json({'type': 'auth', 'token': 'rescuer'})
            assert second.receive_json()['status'] == 409


@pytest.mark.parametrize('endpoint,origin', [
    ('ws://192.168.31.44:8001/api/faces/stream', 'http://192.168.31.44:8001'),
    ('wss://api.resq.test/api/faces/stream', 'https://api.resq.test'),
    ('ws://testserver/api/faces/stream', settings.cors[0]),
])
def test_websocket_accepts_android_endpoint_origin_and_configured_browser_origin(media_client, endpoint, origin):
    client, _ = media_client
    with client.websocket_connect(endpoint, headers={'origin': origin}) as socket:
        socket.send_json({'type': 'auth', 'token': 'rescuer'})
        assert socket.receive_json()['type'] == 'ready'
        socket.send_bytes(photo('face-reference.jpg'))
        result = socket.receive_json()
        assert result['type'] == 'detection' and len(result['faces']) == 1


@pytest.mark.parametrize('origin', [
    'https://192.168.31.44:8001', 'http://192.168.31.44:9000',
    'http://192.168.31.44:8001.unapproved.invalid', 'null', '*',
])
def test_websocket_rejects_other_origins_before_auth(media_client, origin):
    with pytest.raises(WebSocketDisconnect) as rejected:
        with media_client[0].websocket_connect('ws://192.168.31.44:8001/api/faces/stream', headers={'origin': origin}):
            pass
    assert rejected.value.code == 1008


def test_websocket_same_origin_still_requires_valid_auth(media_client):
    with media_client[0].websocket_connect('ws://192.168.31.44:8001/api/faces/stream', headers={'origin': 'http://192.168.31.44:8001'}) as socket:
        socket.send_json({'type': 'auth', 'token': 'bad'})
        assert socket.receive_json()['status'] == 401


def test_websocket_uses_http_connection_dependency_in_production_route(media_client):
    client, fake = media_client
    override = app.dependency_overrides.pop(gateway)
    original = app.state.gateway
    app.state.gateway = fake
    try:
        with client.websocket_connect('/api/faces/stream') as socket:
            socket.send_json({'type': 'auth', 'token': 'rescuer'})
            assert socket.receive_json()['type'] == 'ready'
            socket.send_bytes(photo('face-reference.jpg'))
            assert socket.receive_json()['type'] == 'detection'
    finally:
        app.state.gateway = original
        app.dependency_overrides[gateway] = override


def test_websocket_real_binary_and_json_frames_invalid_payload_and_rate_limit(media_client):
    client, _ = media_client
    with client.websocket_connect('/api/faces/stream') as socket:
        socket.send_json({'type': 'auth', 'token': 'rescuer'})
        assert socket.receive_json()['type'] == 'ready'
        socket.send_bytes(photo('face-reference.jpg'))
        result = socket.receive_json()
        assert result['type'] == 'detection' and result['sequence'] == 0 and len(result['faces']) == 1
        assert result['frame']['annotated_image'].startswith('data:image/jpeg;base64,')
        socket.send_json({'type': 'frame', 'sequence': 1, 'image_base64': base64.b64encode(photo('face-reference.jpg')).decode()})
        paced = socket.receive_json()
        assert paced['status'] == 429 and paced['retry_after_ms'] > 0
        time.sleep(paced['retry_after_ms'] / 1000 + 0.05)
        socket.send_json({'type': 'frame', 'sequence': 2, 'captured_at_ms': 42, 'image_base64': base64.b64encode(photo('face-reference.jpg')).decode()})
        result = socket.receive_json()
        assert result['type'] == 'detection' and result['sequence'] == 2 and result['captured_at_ms'] == 42
        socket.send_json({'type': 'frame', 'sequence': 2, 'image_base64': 'bad'})
        assert socket.receive_json()['status'] == 422
        socket.send_json({'type': 'frame', 'sequence': 3, 'image_base64': '!not-base64!'})
        assert socket.receive_json()['status'] == 415
        socket.send_json({'type': 'ping'})
        assert socket.receive_json()['type'] == 'pong'
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        lease, pace = app.state.live_sessions.keys(ACTORS['rescuer'])
        # Only lease identifiers live in Redis; closing the socket removes both.
        assert redis.get(lease) is None and redis.get(pace) is None
    finally: redis.close()


def test_redis_lease_expiry_and_atomic_owner_release():
    async def run():
        sessions = LiveSessions(settings.redis_url, lease_seconds=1, prefix='resq:test:live:' + uuid4().hex)
        try:
            assert await sessions.available()
            assert await sessions.claim('test-user', 'first')
            assert not await sessions.claim('test-user', 'second')
            assert await sessions.release('test-user', 'second') == 0
            assert (await sessions.touch('test-user', 'first', frame=True))[0] == 1
            assert (await sessions.touch('test-user', 'first', frame=True))[0] == 0
            await asyncio.sleep(1.1)
            assert await sessions.claim('test-user', 'second')
            assert await sessions.release('test-user', 'first') == 0
            assert (await sessions.touch('test-user', 'second'))[0] == 1
            assert await sessions.release('test-user', 'second') >= 1
        finally: await sessions.close()
    asyncio.run(run())


def test_redis_failure_disables_stream_without_fake_success(media_client, monkeypatch):
    from redis.exceptions import ConnectionError
    async def unavailable(*args): raise ConnectionError('test outage')
    monkeypatch.setattr(app.state.live_sessions, 'claim', unavailable)
    with media_client[0].websocket_connect('/api/faces/stream') as socket:
        socket.send_json({'type': 'auth', 'token': 'rescuer'})
        response = socket.receive_json()
        assert response['status'] == 503 and 'Redis' in response['message']
