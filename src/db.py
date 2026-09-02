"""
FerryFlow – TiDB Database Connection

Provides a MySQL-compatible connection to TiDB Cloud.
TiDB Cloud requires SSL; we use the DigiCert CA bundled with certifi.
"""

import ssl
import logging
from contextlib import contextmanager
from typing import Generator

import pymysql
import certifi
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import get_config

logger = logging.getLogger(__name__)


def _build_ssl_context() -> dict:
    """Build SSL configuration for TiDB Cloud connection."""
    config = get_config()
    if config.tidb_ssl_ca:
        ca_file = config.tidb_ssl_ca
    else:
        # Use certifi's CA bundle which includes DigiCert used by TiDB Cloud
        ca_file = certifi.where()
    return {"ca": ca_file}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(pymysql.OperationalError),
    reraise=True,
)
def get_connection() -> pymysql.Connection:
    """
    Open and return a new TiDB connection.
    Caller is responsible for closing it.
    Uses retry with exponential backoff for transient failures.
    """
    config = get_config()

    if not config.tidb_configured:
        raise ConnectionError(
            "TiDB is not configured. Please set TIDB_HOST, TIDB_USER, "
            "and TIDB_PASSWORD in your .env file."
        )

    ssl_params = _build_ssl_context()

    conn = pymysql.connect(
        host=config.tidb_host,
        port=config.tidb_port,
        user=config.tidb_user,
        password=config.tidb_password,
        database=config.tidb_database,
        ssl=ssl_params,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=30,
        write_timeout=30,
        autocommit=True,
    )
    return conn


@contextmanager
def db_cursor() -> Generator[pymysql.cursors.DictCursor, None, None]:
    """
    Context manager that yields a DictCursor and ensures connection is closed.

    Usage:
        with db_cursor() as cursor:
            cursor.execute("SELECT ...")
            rows = cursor.fetchall()
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            yield cursor
    finally:
        conn.close()


def test_connection() -> tuple[bool, str]:
    """
    Test the TiDB connection.
    Returns (success: bool, message: str).
    """
    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT VERSION() AS version")
            row = cursor.fetchone()
            version = row["version"] if row else "unknown"
            return True, f"Connected to TiDB. Version: {version}"
    except ConnectionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Connection failed: {e}"
