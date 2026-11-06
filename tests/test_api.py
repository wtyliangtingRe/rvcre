import time

from fastapi.testclient import TestClient

from rvcre.api import create_app


def test_api_import_generate_and_download(service, tone):
    with TestClient(create_app(service=service)) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        with tone.open("rb") as stream:
            response = client.post("/api/voices/import", data={"name": "API测试"},
                                   files={"audio": ("signal.wav", stream, "audio/wav")})
        assert response.status_code == 201, response.text
        voice = response.json()
        assert client.get(f"/api/voices/{voice['id']}/reference").status_code == 200
        response = client.post("/api/tts", json={"voice_id": voice["id"], "text": "你好"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        for _ in range(200):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert job["status"] == "succeeded", job
        audio = client.get(f"/api/jobs/{job_id}/audio/0")
        assert audio.status_code == 200 and audio.content.startswith(b"RIFF")
        assert client.get(f"/api/jobs/{job_id}/audio/-1").status_code == 404
        bad = client.post("/api/tts", json={"voice_id": voice["id"], "text": "hi", "mode": "hifi"})
        assert bad.status_code == 400


def test_api_rejects_unknown_controls_and_missing_voice(service):
    with TestClient(create_app(service=service)) as client:
        assert client.post("/api/design", json={"text": "hi", "spec": {"imaginary_dial": 1}}).status_code == 422
        assert client.get("/api/voices/aaaaaaaaaaaa").status_code == 404
