import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from . import __version__
from .schema import Performance, VoiceSpec
from .service import VoiceService


class DesignRequest(BaseModel):
    text: str
    spec: VoiceSpec = Field(default_factory=VoiceSpec)
    candidates: int = Field(default=3, ge=1, le=6)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    parent_voice_id: str | None = None


class CandidateRequest(BaseModel):
    job_id: str
    index: int = Field(ge=0)
    name: str


class TTSRequest(BaseModel):
    voice_id: str
    text: str
    performance: Performance = Field(default_factory=Performance)
    mode: str = "controllable"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


def create_app(workspace: Path = Path("data"), config_path: Path | None = None,
               *, service: VoiceService | None = None) -> FastAPI:
    owned = service is None
    service = service or VoiceService(workspace, config_path)

    @asynccontextmanager
    async def lifespan(app):
        yield
        if owned:
            service.close()

    app = FastAPI(title="RVCre · 声构", version=__version__, lifespan=lifespan)
    app.state.service = service

    @app.exception_handler(ValueError)
    async def invalid_request(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ValidationError)
    async def invalid_controls(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def missing_item(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/backends")
    def backends():
        return service.backend_status()

    @app.get("/api/voices")
    def voices():
        return service.list_voices()

    @app.post("/api/voices/import", status_code=201)
    def import_voice(audio: UploadFile = File(...), name: str = Form(...),
                     transcript: str = Form(""), notes: str = Form(""), source_note: str = Form("")):
        with tempfile.TemporaryDirectory(prefix="rvcre-upload-") as tmp:
            path = Path(tmp) / ("input" + Path(audio.filename or "audio.wav").suffix)
            with path.open("wb") as stream:
                shutil.copyfileobj(audio.file, stream)
            return service.import_voice(name, str(path), transcript, notes, source_note=source_note)

    @app.post("/api/voices/from-candidate", status_code=201)
    def save_candidate(body: CandidateRequest):
        return service.save_candidate(body.job_id, body.index, body.name)

    @app.get("/api/voices/{voice_id}")
    def voice(voice_id: str):
        return service.get_voice(voice_id)

    @app.get("/api/voices/{voice_id}/reference")
    def voice_reference(voice_id: str):
        return FileResponse(service.get_voice(voice_id)["reference_path"], media_type="audio/wav")

    @app.post("/api/design", status_code=202)
    def design(body: DesignRequest):
        return service.submit_design(body.text, body.spec.model_dump(), body.candidates,
                                     body.seed, body.parent_voice_id)

    @app.post("/api/tts", status_code=202)
    def tts(body: TTSRequest):
        return service.submit_tts(body.voice_id, body.text, body.performance.model_dump(), body.mode, body.seed)

    @app.post("/api/vc", status_code=202)
    def vc(audio: UploadFile = File(...), voice_id: str = Form(...),
           diffusion_steps: int = Form(30), similarity_cfg_rate: float = Form(0.7),
           intelligibility_cfg_rate: float = Form(0.7)):
        with tempfile.TemporaryDirectory(prefix="rvcre-vc-") as tmp:
            path = Path(tmp) / ("input" + Path(audio.filename or "audio.wav").suffix)
            with path.open("wb") as stream:
                shutil.copyfileobj(audio.file, stream)
            return service.submit_vc(voice_id, str(path), {
                "diffusion_steps": diffusion_steps, "similarity_cfg_rate": similarity_cfg_rate,
                "intelligibility_cfg_rate": intelligibility_cfg_rate,
            })

    @app.get("/api/jobs")
    def jobs():
        return service.jobs.list()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return service.get_job(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return service.cancel_job(job_id)

    @app.get("/api/jobs/{job_id}/audio/{index}")
    def audio(job_id: str, index: int):
        path = service.audio_path(job_id, index)
        if not path.is_file():
            raise HTTPException(404, "音频文件已被移动或删除。")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @app.get("/api/jobs/{job_id}/files/{index}")
    def download(job_id: str, index: int):
        path = service.file_path(job_id, index)
        return FileResponse(path, filename=path.name)

    return app
