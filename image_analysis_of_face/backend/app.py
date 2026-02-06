from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

import camera_worker
import state

app = FastAPI(
    title="Mental Health – Facial Signal Backend",
    description="Local backend for facial feature extraction (Module 2)",
    version="1.0.0"
)

# -------------------------------------------------
# API ROUTES (define FIRST)
# -------------------------------------------------

@app.post("/start")
def start_pipeline():
    if state.pipeline_running:
        return {"status": "already_running"}

    camera_worker.start_pipeline()
    return {"status": "started"}


@app.post("/stop")
def stop_pipeline():
    if not state.pipeline_running:
        return {"status": "already_stopped"}

    camera_worker.stop_pipeline()
    return {"status": "stopped"}


@app.get("/status")
def status():
    return {
        "pipeline_running": state.pipeline_running
    }

# -------------------------------------------------
# FRONTEND (mounted AFTER routes)
# -------------------------------------------------
app.mount(
    "/ui",
    StaticFiles(directory="../frontend", html=True),
    name="frontend"
)