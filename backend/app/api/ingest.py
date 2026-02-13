from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from datetime import datetime, timezone
import os
import tempfile

from app.services.ingest_service import ingest_file_bytes

router = APIRouter(tags=["ingest"])

ALLOWED_EXT = {".md", ".txt"}  # add ".pdf" later

@router.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    result = {
        "files_received": len(files),
        "files_ingested": 0,
        "chunks_created": 0,
        "embeddings_written": 0,
        "errors": [],
        "last_ingest_time": None,
    }

    for f in files:
        _, ext = os.path.splitext(f.filename.lower())
        if ext not in ALLOWED_EXT:
            result["errors"].append({"file": f.filename, "error": f"Unsupported file type: {ext}"})
            continue

        try:
            data = await f.read()
            # Wrapper should chunk, embed, and write to Supabase (or call your existing script functions)
            stats = ingest_file_bytes(filename=f.filename, data=data)

            result["files_ingested"] += 1
            result["chunks_created"] += int(stats.get("chunks_created", 0))
            result["embeddings_written"] += int(stats.get("embeddings_written", 0))

        except Exception as e:
            result["errors"].append({"file": f.filename, "error": str(e)})

    result["last_ingest_time"] = datetime.now(timezone.utc).isoformat()

    # (optional) persist last_ingest_time somewhere (DB table / local file) for /corpus/stats
    return result
