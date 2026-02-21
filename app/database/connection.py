import os
import sys
import asyncio
import ssl as ssl_module
import logging
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import text

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
logger = logging.getLogger(__name__)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "iot_db")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS") or os.getenv("DB_PASSWORD", "")

DATABASE_URL = f"mysql+aiomysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

CA_CERT_PATH = os.getenv("CA_CERT_PATH", "")

def _build_ssl_ctx() -> ssl_module.SSLContext:
    ctx = ssl_module.create_default_context()
    ctx.check_hostname = False
    if CA_CERT_PATH and os.path.exists(CA_CERT_PATH):
        ctx.load_verify_locations(CA_CERT_PATH)
        ctx.verify_mode = ssl_module.CERT_REQUIRED
        logger.info(f"SSL: using CA cert from {CA_CERT_PATH}")
    else:
        ctx.verify_mode = ssl_module.CERT_NONE
        logger.warning("SSL: CA_CERT_PATH not set or missing — peer verification disabled.")
    return ctx

engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
    connect_args={"ssl": _build_ssl_ctx()},
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def check_connection():
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("MySQL connection OK")
        return True
    except Exception as e:
        logger.error(f"MySQL connection failed: {e}")
        return False