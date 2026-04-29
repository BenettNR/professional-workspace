import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.documents import router as documents_router
from config.settings import settings
from pii.engine import get_analyzer  # warm the NLP model at startup

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading spaCy NLP model…")
    get_analyzer()
    logger.info("NLP model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Document Reader",
    description="Enterprise bank statement reader with PII masking",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
