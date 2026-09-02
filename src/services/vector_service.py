"""
FerryFlow – TiDB Vector Service

Manages the airport_profiles table with VECTOR(1024) embeddings.
Provides semantic airport similarity search using VEC_COSINE_DISTANCE.

Embeddings are generated with Amazon Bedrock Cohere embed-multilingual-v3.
Profile text is generated programmatically from airport analytics.

If the vector table has not been populated yet, all similarity functions
return an explicit status indicating setup is required.
"""

import json
import logging
from typing import Any

from src.db import db_cursor

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = 1024


# ---------------------------------------------------------------------------
# Table management
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS airport_profiles (
    airport_id   INT PRIMARY KEY,
    iata         VARCHAR(10),
    profile_text TEXT,
    embedding    VECTOR({dim}),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_iata (iata)
)
""".format(dim=EMBEDDING_DIMENSIONS)


def create_vector_table() -> dict:
    """
    Create the airport_profiles vector table if it does not exist.
    Returns {"success": bool, "message": str}.
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(CREATE_TABLE_SQL)
        return {"success": True, "message": "airport_profiles table ready."}
    except Exception as e:
        logger.error("create_vector_table error: %s", e)
        return {"success": False, "message": str(e)}


def vector_table_exists() -> bool:
    """Check whether the airport_profiles table exists."""
    sql = """
        SELECT COUNT(*) AS cnt
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_name = 'airport_profiles'
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
        return int(row["cnt"]) > 0 if row else False
    except Exception as e:
        logger.error("vector_table_exists error: %s", e)
        return False


def vector_table_populated() -> dict:
    """
    Check whether the airport_profiles table has embeddings.
    Returns {"populated": bool, "count": int, "message": str}.
    """
    if not vector_table_exists():
        return {
            "populated": False,
            "count": 0,
            "message": "airport_profiles table does not exist. Run: python scripts/build_airport_profiles.py",
        }
    sql = "SELECT COUNT(*) AS cnt FROM airport_profiles WHERE embedding IS NOT NULL"
    try:
        with db_cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
        count = int(row["cnt"]) if row else 0
        if count == 0:
            return {
                "populated": False,
                "count": 0,
                "message": "airport_profiles table is empty. Run: python scripts/build_airport_profiles.py",
            }
        return {"populated": True, "count": count, "message": f"{count} airport embeddings available."}
    except Exception as e:
        logger.error("vector_table_populated error: %s", e)
        return {"populated": False, "count": 0, "message": str(e)}


# ---------------------------------------------------------------------------
# Storing embeddings
# ---------------------------------------------------------------------------

def store_airport_embedding(
    airport_id: int,
    iata: str,
    profile_text: str,
    embedding: list[float],
) -> dict:
    """
    Insert or update an airport profile and its embedding in airport_profiles.
    Returns {"success": bool, "message": str}.
    """
    if len(embedding) != EMBEDDING_DIMENSIONS:
        return {
            "success": False,
            "message": f"Embedding has {len(embedding)} dimensions, expected {EMBEDDING_DIMENSIONS}.",
        }

    # TiDB stores vectors as JSON arrays
    embedding_str = json.dumps(embedding)

    sql = """
        INSERT INTO airport_profiles (airport_id, iata, profile_text, embedding)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            iata         = VALUES(iata),
            profile_text = VALUES(profile_text),
            embedding    = VALUES(embedding),
            updated_at   = CURRENT_TIMESTAMP
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (airport_id, iata, profile_text, embedding_str))
        return {"success": True, "message": f"Stored embedding for {iata} (id={airport_id})."}
    except Exception as e:
        logger.error("store_airport_embedding error for %s: %s", iata, e)
        return {"success": False, "message": str(e)}


def get_stored_embedding(airport_id: int) -> list[float] | None:
    """Retrieve the stored embedding for an airport by ID."""
    sql = "SELECT embedding FROM airport_profiles WHERE airport_id = %s LIMIT 1"
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (airport_id,))
            row = cursor.fetchone()
        if row and row.get("embedding"):
            raw = row["embedding"]
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, list):
                return raw
        return None
    except Exception as e:
        logger.error("get_stored_embedding error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def find_similar_airports(
    airport_id: int,
    top_k: int = 5,
    exclude_ids: list[int] | None = None,
) -> dict:
    """
    Find the top_k most semantically similar airports to the given airport
    using VEC_COSINE_DISTANCE on stored 1024-dim Cohere embeddings.

    Returns:
    {
        "success": bool,
        "results": [
            {
                "airport_id": int,
                "iata": str,
                "name": str,
                "city": str,
                "country": str,
                "similarity_score": float,   # 1 = identical, 0 = orthogonal
                "distance": float,           # cosine distance (lower = more similar)
                "profile_text": str,
            },
            ...
        ],
        "message": str,
    }
    """
    status = vector_table_populated()
    if not status["populated"]:
        return {"success": False, "results": [], "message": status["message"]}

    # Check the target airport has an embedding
    embedding = get_stored_embedding(airport_id)
    if embedding is None:
        return {
            "success": False,
            "results": [],
            "message": f"No embedding stored for airport_id {airport_id}. Run build_airport_profiles.py.",
        }

    embedding_str = json.dumps(embedding)

    exclude_ids = exclude_ids or []
    exclude_ids_with_self = list(set(exclude_ids + [airport_id]))
    placeholders = ", ".join(["%s"] * len(exclude_ids_with_self))

    sql = f"""
        SELECT
            ap.airport_id,
            ap.iata,
            ap.profile_text,
            a.name,
            g.city,
            g.country,
            VEC_COSINE_DISTANCE(ap.embedding, %s) AS cosine_distance
        FROM airport_profiles ap
        INNER JOIN airport a ON ap.airport_id = a.airport_id
        LEFT JOIN airport_geo g ON a.airport_id = g.airport_id
        WHERE ap.airport_id NOT IN ({placeholders})
          AND ap.embedding IS NOT NULL
        ORDER BY cosine_distance ASC
        LIMIT %s
    """

    try:
        with db_cursor() as cursor:
            params = [embedding_str] + exclude_ids_with_self + [top_k]
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            dist = float(row["cosine_distance"]) if row["cosine_distance"] is not None else 1.0
            # Cosine distance ∈ [0, 2]; similarity = 1 - dist/2 maps to [0, 1]
            similarity = round(max(0.0, 1.0 - dist / 2.0), 4)
            results.append({
                "airport_id": int(row["airport_id"]),
                "iata": row.get("iata", ""),
                "name": row.get("name", ""),
                "city": row.get("city", ""),
                "country": row.get("country", ""),
                "similarity_score": similarity,
                "distance": round(dist, 6),
                "profile_text": row.get("profile_text", ""),
            })

        return {
            "success": True,
            "results": results,
            "message": f"Found {len(results)} similar airports.",
        }
    except Exception as e:
        logger.error("find_similar_airports error: %s", e)
        return {"success": False, "results": [], "message": str(e)}


def find_similar_airports_by_query(
    query_text: str,
    top_k: int = 5,
) -> dict:
    """
    Find airports similar to an arbitrary text query using a query embedding.
    Useful for 'Find an alternative stop in South America' type queries.
    """
    from src.services.bedrock_service import embed_query

    status = vector_table_populated()
    if not status["populated"]:
        return {"success": False, "results": [], "message": status["message"]}

    query_embedding = embed_query(query_text)
    if query_embedding is None:
        return {
            "success": False,
            "results": [],
            "message": "Could not generate query embedding. Check Bedrock configuration.",
        }

    embedding_str = json.dumps(query_embedding)

    sql = """
        SELECT
            ap.airport_id,
            ap.iata,
            ap.profile_text,
            a.name,
            g.city,
            g.country,
            VEC_COSINE_DISTANCE(ap.embedding, %s) AS cosine_distance
        FROM airport_profiles ap
        INNER JOIN airport a ON ap.airport_id = a.airport_id
        LEFT JOIN airport_geo g ON a.airport_id = g.airport_id
        WHERE ap.embedding IS NOT NULL
        ORDER BY cosine_distance ASC
        LIMIT %s
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (embedding_str, top_k))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            dist = float(row["cosine_distance"]) if row["cosine_distance"] is not None else 1.0
            similarity = round(max(0.0, 1.0 - dist / 2.0), 4)
            results.append({
                "airport_id": int(row["airport_id"]),
                "iata": row.get("iata", ""),
                "name": row.get("name", ""),
                "city": row.get("city", ""),
                "country": row.get("country", ""),
                "similarity_score": similarity,
                "distance": round(dist, 6),
                "profile_text": row.get("profile_text", ""),
            })

        return {
            "success": True,
            "results": results,
            "message": f"Found {len(results)} airports matching query.",
        }
    except Exception as e:
        logger.error("find_similar_airports_by_query error: %s", e)
        return {"success": False, "results": [], "message": str(e)}


# ---------------------------------------------------------------------------
# Batch similarity for all route airports
# ---------------------------------------------------------------------------

def find_similar_for_route(
    airport_ids: list[int],
    top_k: int = 3,
) -> dict[int, list[dict]]:
    """
    Run similarity search for each airport in a route.
    Returns {airport_id: [similar_airports]} dict.
    """
    results = {}
    for aid in airport_ids:
        sim = find_similar_airports(aid, top_k=top_k, exclude_ids=airport_ids)
        results[aid] = sim.get("results", [])
    return results
