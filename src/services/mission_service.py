"""
FerryFlow – Mission Orchestration Service

Orchestrates the full mission planning pipeline:
  1. Resolve origin and destination airports
  2. Load aircraft context from dataset
  3. Fetch airport profiles for all candidate airports
  4. Compute route candidates via route_service
  5. Enrich routes with weather data
  6. Call Bedrock for AI recommendation
  7. Return unified mission result

DISCLAIMER: This is decision support only. Not an operational flight plan.
"""

import logging
from datetime import datetime
from typing import Any

from src.services.airport_service import (
    get_airport_by_iata,
    get_airport_by_id,
    get_airport_profile,
    search_airports,
)
from src.services.route_service import compute_mission_routes
from src.services.bedrock_service import generate_mission_recommendation
from src.db import db_cursor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aircraft lookup
# ---------------------------------------------------------------------------

def get_aircraft_types(limit: int = 100) -> list[dict]:
    """
    Return available aircraft types from the dataset.
    Used to populate the aircraft selector in the UI.
    """
    sql = """
        SELECT DISTINCT
            at.type_id,
            at.identifier,
            at.description
        FROM airplane_type at
        ORDER BY at.identifier
        LIMIT %s
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
        return rows if rows else []
    except Exception as e:
        logger.error("get_aircraft_types error: %s", e)
        # Return a minimal fallback list so UI doesn't break
        return [
            {"airplane_type_id": 0, "identifier": "Generic", "description": "Generic aircraft type"},
        ]


def get_aircraft_info(aircraft_type_identifier: str) -> dict | None:
    """
    Retrieve aircraft type details by identifier string.
    """
    sql = """
        SELECT
            at.type_id,
            at.identifier,
            at.description,
            COUNT(a.airplane_id) AS fleet_count
        FROM airplane_type at
        LEFT JOIN airplane a ON a.type_id = at.type_id
        WHERE at.identifier = %s OR at.description LIKE %s
        GROUP BY at.type_id, at.identifier, at.description
        LIMIT 1
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (aircraft_type_identifier, f"%{aircraft_type_identifier}%"))
            return cursor.fetchone()
    except Exception as e:
        logger.error("get_aircraft_info error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Airport profiles cache (within a single mission computation)
# ---------------------------------------------------------------------------

def _load_profiles_for_ids(airport_ids: list[int]) -> dict[int, dict]:
    """Load profiles for a list of airport IDs, return as {airport_id: profile}."""
    profiles = {}
    for aid in airport_ids:
        try:
            profile = get_airport_profile(aid)
            if "error" not in profile:
                profiles[aid] = profile
        except Exception as e:
            logger.warning("Failed to load profile for airport %s: %s", aid, e)
    return profiles


# ---------------------------------------------------------------------------
# Main mission planner
# ---------------------------------------------------------------------------

def plan_mission(
    origin_iata: str,
    dest_iata: str,
    aircraft_type: str = "Not specified",
    departure_time: str | None = None,
    include_ai: bool = True,
) -> dict:
    """
    Plan a ferry/cargo mission from origin to destination.

    Returns a unified mission result dict:
    {
        "success": bool,
        "origin": dict,
        "destination": dict,
        "aircraft_type": str,
        "departure_time": str,
        "recommended_route": dict | None,
        "alternative_routes": list[dict],
        "airports_data": dict[str, dict],   # keyed by iata
        "ai_recommendation": dict,
        "graph_stats": dict,
        "error": str | None,
        "warnings": list[str],
        "disclaimer": str,
        "computed_at": str,
    }
    """
    warnings: list[str] = []
    result: dict[str, Any] = {
        "success": False,
        "origin": None,
        "destination": None,
        "aircraft_type": aircraft_type,
        "departure_time": departure_time or "Not specified",
        "recommended_route": None,
        "alternative_routes": [],
        "airports_data": {},
        "ai_recommendation": {"success": False, "text": "", "error": None},
        "graph_stats": {},
        "error": None,
        "warnings": warnings,
        "disclaimer": (
            "FerryFlow MVP provides historical/data-driven decision support "
            "and is not an operational flight plan."
        ),
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }

    # -- 1. Resolve airports --
    origin = get_airport_by_iata(origin_iata.strip().upper())
    if not origin:
        result["error"] = f"Origin airport '{origin_iata}' not found in dataset."
        return result

    destination = get_airport_by_iata(dest_iata.strip().upper())
    if not destination:
        result["error"] = f"Destination airport '{dest_iata}' not found in dataset."
        return result

    result["origin"] = origin
    result["destination"] = destination

    if origin["airport_id"] == destination["airport_id"]:
        result["error"] = "Origin and destination are the same airport."
        return result

    # -- 2. Aircraft context --
    aircraft_info = get_aircraft_info(aircraft_type)
    result["aircraft_info"] = aircraft_info

    # -- 3. Compute routes --
    # Pre-load minimal profiles for origin/destination to seed route scoring
    seed_profiles = _load_profiles_for_ids([origin["airport_id"], destination["airport_id"]])

    route_result = compute_mission_routes(
        origin_id=origin["airport_id"],
        dest_id=destination["airport_id"],
        airport_profiles=seed_profiles,
        max_routes=3,
    )

    if route_result.get("error"):
        warnings.append(f"Route engine warning: {route_result['error']}")

    routes = route_result.get("routes", [])
    result["graph_stats"] = route_result.get("graph_stats", {})

    if not routes:
        # Fallback: direct route with just origin → destination
        warnings.append(
            "No historical route found. Showing direct origin → destination."
        )
        routes = [_build_direct_route(origin, destination)]

    # -- 4. Enrich: load full profiles for all airports in routes --
    all_airport_ids: set[int] = set()
    for route in routes:
        for aid in route.get("airport_ids", []):
            all_airport_ids.add(aid)

    all_profiles = _load_profiles_for_ids(list(all_airport_ids))
    # Merge with seed profiles
    all_profiles.update(seed_profiles)

    # -- 5. Re-enrich route airport details with full profiles --
    for route in routes:
        for airport_entry in route.get("airports", []):
            aid = airport_entry.get("airport_id")
            if aid and aid in all_profiles:
                profile = all_profiles[aid]
                metrics = profile.get("metrics", {})
                score = profile.get("operational_score", {})
                airport_entry["total_movements"] = metrics.get("total_movements", 0)
                airport_entry["unique_connected"] = metrics.get("unique_connected_airports", 0)
                airport_entry["airlines"] = metrics.get("airlines", 0)
                airport_entry["operational_score"] = score.get("total", 0)
                airport_entry["score_components"] = score.get("components", {})
                airport_entry["historical_weather"] = profile.get("historical_weather", [])

    # Keyed by iata for easy lookup in AI prompts and UI
    airports_data_by_iata: dict[str, dict] = {}
    for aid, profile in all_profiles.items():
        iata_key = profile.get("airport", {}).get("iata", str(aid))
        airports_data_by_iata[iata_key] = profile

    result["airports_data"] = airports_data_by_iata

    # -- 6. Set recommended and alternatives --
    recommended = routes[0] if routes else None
    alternatives = routes[1:] if len(routes) > 1 else []

    result["recommended_route"] = recommended
    result["alternative_routes"] = alternatives
    result["success"] = True

    # -- 7. AI recommendation --
    if include_ai:
        mission_context_for_ai = {
            "origin": origin,
            "destination": destination,
            "aircraft_type": aircraft_type,
            "departure_time": departure_time or "Not specified",
            "recommended_route": recommended or {},
            "alternative_routes": alternatives,
            "airports_data": airports_data_by_iata,
        }
        ai_result = generate_mission_recommendation(mission_context_for_ai)
        result["ai_recommendation"] = ai_result
        if not ai_result["success"]:
            warnings.append(f"AI recommendation unavailable: {ai_result.get('error', 'unknown error')}")

    return result


# ---------------------------------------------------------------------------
# Direct route fallback
# ---------------------------------------------------------------------------

def _build_direct_route(origin: dict, destination: dict) -> dict:
    """
    Construct a minimal direct route when the route engine finds no path.
    Uses Haversine distance only.
    """
    from src.services.route_service import haversine_km, km_to_nm

    olat = origin.get("latitude") or 0.0
    olon = origin.get("longitude") or 0.0
    dlat = destination.get("latitude") or 0.0
    dlon = destination.get("longitude") or 0.0

    dist_km = haversine_km(float(olat), float(olon), float(dlat), float(dlon))

    return {
        "airports": [
            {
                "airport_id": origin["airport_id"],
                "iata": origin.get("iata", ""),
                "name": origin.get("name", ""),
                "city": origin.get("city", ""),
                "country": origin.get("country", ""),
                "lat": float(olat),
                "lon": float(olon),
                "leg_distance_km": 0.0,
                "leg_distance_nm": 0.0,
                "role": "origin",
                "operational_score": 0,
                "total_movements": 0,
                "unique_connected": 0,
                "airlines": 0,
                "historical_weather": [],
            },
            {
                "airport_id": destination["airport_id"],
                "iata": destination.get("iata", ""),
                "name": destination.get("name", ""),
                "city": destination.get("city", ""),
                "country": destination.get("country", ""),
                "lat": float(dlat),
                "lon": float(dlon),
                "leg_distance_km": round(dist_km, 1),
                "leg_distance_nm": km_to_nm(dist_km),
                "role": "destination",
                "operational_score": 0,
                "total_movements": 0,
                "unique_connected": 0,
                "airlines": 0,
                "historical_weather": [],
            },
        ],
        "airport_ids": [origin["airport_id"], destination["airport_id"]],
        "num_legs": 1,
        "total_distance_km": round(dist_km, 1),
        "total_distance_nm": km_to_nm(dist_km),
        "direct_distance_km": round(dist_km, 1),
        "route_score": 0.0,
        "note": "Direct route (no historical connectivity found in dataset).",
    }
