"""FastAPI app: MJPEG stream + reactive stats + runtime prompt/class control."""

import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import SETTINGS
from .pipeline.engine import Engine
from .pipeline.source import VideoSource, resolve_source

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VisionStream")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Engine(SETTINGS)


class PromptsIn(BaseModel):
    prompts: list[str]


class ClassesIn(BaseModel):
    classes: list[int] | None = None


class SourceIn(BaseModel):
    source: str  # "0" for webcam, or a filename under data/


@app.on_event("startup")
def _startup() -> None:
    # Pay the ~2.5s graph-compile / MPS-init cost now, not on the user's first frame.
    engine.warmup()


@app.on_event("shutdown")
def _shutdown() -> None:
    engine.stop()


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "device": SETTINGS.device,
        "in_docker": SETTINGS.is_docker(),
        # Docker Desktop on Apple Silicon exposes no MPS and no host webcam.
        "webcam_supported": not SETTINGS.is_docker(),
    }


@app.get("/api/stats")
def stats() -> dict:
    return engine.stats()


@app.get("/api/classes")
def classes() -> dict:
    return {
        "names": {int(k): v for k, v in engine.detector.names.items()},
        "active": SETTINGS.class_filter,
    }


@app.post("/api/classes")
def set_classes(body: ClassesIn) -> dict:
    SETTINGS.class_filter = body.classes or None
    return {"active": SETTINGS.class_filter}


@app.get("/api/prompts")
def get_prompts() -> dict:
    return {"prompts": engine.clip.prompts, "clip_every_n": SETTINGS.clip_every_n}


@app.post("/api/prompts")
def set_prompts(body: PromptsIn) -> dict:
    """Swapping prompts re-encodes text once and changes behaviour with no retraining."""
    engine.clip.set_prompts(body.prompts)
    return {"prompts": engine.clip.prompts}


@app.get("/api/videos")
def list_videos() -> dict:
    vids = sorted(p.name for p in DATA_DIR.glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
    return {"videos": vids}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(400, f"unsupported video type: {suffix!r}")
    dest = DATA_DIR / Path(file.filename).name
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"saved": dest.name}


@app.post("/api/start")
def start(body: SourceIn) -> dict:
    if body.source.isdigit() and SETTINGS.is_docker():
        raise HTTPException(
            400,
            "Webcam is unavailable in Docker: Docker Desktop on Apple Silicon has no host "
            "camera passthrough. Use file input here, or run natively for webcam.",
        )
    try:
        source = resolve_source(body.source, DATA_DIR)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    engine.start(source)
    return {"started": True, "source": source.describe()}


@app.post("/api/stop")
def stop() -> dict:
    engine.stop()
    return {"started": False}


@app.get("/api/stream")
def stream() -> StreamingResponse:
    if not engine.running:
        raise HTTPException(409, "no source running — POST /api/start first")
    return StreamingResponse(
        engine.jpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
