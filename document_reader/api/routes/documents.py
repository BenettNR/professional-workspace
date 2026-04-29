import asyncio
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from config.settings import settings
from extraction.models import ProcessingResult
from output.json_formatter import to_json
from output.report_formatter import to_markdown
from pipeline import process_document

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_SUFFIXES = {f".{fmt}" for fmt in settings.supported_formats}


def _validate_upload(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
        )
    if file.size and file.size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit.",
        )


@router.post("/process", summary="Upload and process a bank statement")
async def process(
    file: UploadFile,
    narrative: bool = Query(True, description="Generate AI narrative summary"),
    enrich: bool = Query(True, description="Use Claude to decode cryptic merchant names"),
    format: str = Query("json", description="Output format: json | markdown"),
) -> JSONResponse | PlainTextResponse:
    _validate_upload(file)

    suffix = Path(file.filename or "file").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result: ProcessingResult = await asyncio.to_thread(
            process_document,
            tmp_path,
            generate_narrative=narrative,
            use_claude_enrichment=enrich,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    if format.lower() == "markdown":
        return PlainTextResponse(to_markdown(result), media_type="text/markdown")

    return JSONResponse(content={"ok": True, "data": __import__("json").loads(to_json(result))})


@router.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}
