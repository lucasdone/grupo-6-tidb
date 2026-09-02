"""
FerryFlow – Airport Analytics Service

Provides airport lookup, search, and profile calculations derived
from the airportdb historical dataset. All metrics are dataset-derived
indicators and not real-time or official airport ratings.
"""

import logging
from typing import Any

from src.db import db_cursor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup / search
# ---------------------------------------------------------------------------

def search_airports(query: str, limit: int = 20) -> list[dict]:
    """
    Search airports by IATA code, name, or city.
    Returns list of dicts with basic airport info.
    """
    sql = """
        SELECT
            a.airport_id,
            a.iata,
            a.icao,
            a.name,
            g.city,
            g.country,
            g.latitude,
            g.longitude
        FROM airport a
        LEFT JOIN airport_geo g ON a.airport_id = g.airport_id
        WHERE
            a.iata LIKE %s
            OR a.name LIKE %s
            OR g.city LIKE %s
            OR a.icao LIKE %s
        ORDER BY
            CASE WHEN a.iata = %s THEN 0
                 WHEN a.iata LIKE %s THEN 1
                 ELSE 2 END,
            a.name
        LIMIT %s
    """
    pattern = f"%{query}%"
    exact = query.upper()
    starts = f"{query.upper()}%"
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (pattern, pattern, pattern, pattern, exact, starts, limit))
            return cursor.fetchall()
    except Exception as e:
        logger.error("search_airports error: %s", e)
        return []


def get_airport_by_iata(iata: str) -> dict | None:
    """
    Fetch a single airport record by IATA code.
    Returns None if not found.
    """
    sql = """
        SELECT
            a.airport_id,
            a.iata,
            a.icao,
            a.name,
            g.city,
            g.country,
            g.latitude,
            g.longitude
        FROM airport a
        LEFT JOIN airport_geo g ON a.airport_id = g.airport_id
        WHERE a.iata = %s
        LIMIT 1
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (iata.upper(),))
            return cursor.fetchone()
    except Exception as e:
        logger.error("get_airport_by_iata error: %s", e)
        return None


def get_airport_by_id(airport_id: int) -> dict | None:
    """Fetch a single airport record by internal airport_id."""
    sql = """
        SELECT
            a.airport_id,
            a.iata,
            a.icao,
            a.name,
            g.city,
            g.country,
            g.latitude,
            g.longitude
        FROM airport a
        LEFT JOIN airport_geo g ON a.airport_id = g.airport_id
        WHERE a.airport_id = %s
        LIMIT 1
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (airport_id,))
            return cursor.fetchone()
    except Exception as e:
        logger.error("get_airport_by_id error: %s", e)
        return None


def list_all_airports(limit: int = 2000) -> list[dict]:
    """
    Return a list of airports with coordinates for graph building.
    """
    sql = """
        SELECT
            a.airport_id,
            a.iata,
            a.icao,
            a.name,
            g.city,
            g.country,
            g.latitude,
            g.longitude
        FROM airport a
        INNER JOIN airport_geo g ON a.airport_id = g.airport_id
        WHERE a.iata IS NOT NULL AND a.iata != ''
            AND g.latitude IS NOT NULL AND g.longitude IS NOT NULL
        LIMIT %s
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (limit,))
            return cursor.fetchall()
    except Exception as e:
        logger.error("list_all_airports error: %s", e)
        return []


# ---------------------------------------------------------------------------
# Traffic metrics
# ---------------------------------------------------------------------------

def get_flight_metrics(airport_id: int) -> dict:
    """
    Compute historical flight movement metrics for an airport.
    Returns inbound count, outbound count, unique routes, airlines, aircraft types.
    All values are historical/dataset-derived indicators.
    """
    sql_out = """
        SELECT
            COUNT(*) AS outbound_flights,
            COUNT(DISTINCT f.to) AS unique_destinations,
            COUNT(DISTINCT f.airline_id) AS airlines_outbound,
            COUNT(DISTINCT f.airplane_id) AS aircraft_outbound
        FROM flight f
        INNER JOIN airplane ap ON f.airplane_id = ap.airplane_id
        WHERE f.`from` = %s
    """
    sql_in = """
        SELECT
            COUNT(*) AS inbound_flights,
            COUNT(DISTINCT f.`from`) AS unique_origins,
            COUNT(DISTINCT f.airline_id) AS airlines_inbound,
            COUNT(DISTINCT f.airplane_id) AS aircraft_inbound
        FROM flight f
        WHERE f.`to` = %s
    """
    sql_schedule = """
        SELECT COUNT(*) AS scheduled_routes
        FROM flightschedule fs
        WHERE fs.`from` = %s OR fs.`to` = %s
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql_out, (airport_id,))
            out = cursor.fetchone() or {}

            cursor.execute(sql_in, (airport_id,))
            inn = cursor.fetchone() or {}

            cursor.execute(sql_schedule, (airport_id, airport_id))
            sched = cursor.fetchone() or {}

        outbound = int(out.get("outbound_flights") or 0)
        inbound = int(inn.get("inbound_flights") or 0)
        unique_destinations = int(out.get("unique_destinations") or 0)
        unique_origins = int(inn.get("unique_origins") or 0)
        airlines = max(
            int(out.get("airlines_outbound") or 0),
            int(inn.get("airlines_inbound") or 0),
        )
        aircraft_types = max(
            int(out.get("aircraft_outbound") or 0),
            int(inn.get("aircraft_inbound") or 0),
        )
        scheduled_routes = int(sched.get("scheduled_routes") or 0)

        return {
            "outbound_flights": outbound,
            "inbound_flights": inbound,
            "total_movements": outbound + inbound,
            "unique_connected_airports": unique_destinations + unique_origins,
            "airlines": airlines,
            "aircraft_types": aircraft_types,
            "scheduled_routes": scheduled_routes,
        }
    except Exception as e:
        logger.error("get_flight_metrics error for airport %s: %s", airport_id, e)
        return {
            "outbound_flights": 0,
            "inbound_flights": 0,
            "total_movements": 0,
            "unique_connected_airports": 0,
            "airlines": 0,
            "aircraft_types": 0,
            "scheduled_routes": 0,
        }


def get_hourly_activity(airport_id: int) -> list[dict]:
    """
    Return approximate historical activity by hour of day (outbound departures).
    Returns list of {"hour": 0-23, "count": int}.
    """
    sql = """
        SELECT
            HOUR(f.departure) AS hour,
            COUNT(*) AS flight_count
        FROM flight f
        WHERE f.`from` = %s AND f.departure IS NOT NULL
        GROUP BY HOUR(f.departure)
        ORDER BY hour
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (airport_id,))
            rows = cursor.fetchall()
        return [{"hour": int(r["hour"]), "count": int(r["flight_count"])} for r in rows]
    except Exception as e:
        logger.error("get_hourly_activity error: %s", e)
        return []


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def get_airport_weather(airport_id: int, limit: int = 10) -> list[dict]:
    """
    Return historical weather observations for an airport if available.
    In the airportdb dataset, weatherdata.station maps to airport_id (stations 1-4 only).
    Data is historical (June 2015), not a forecast.
    """
    sql = """
        SELECT
            w.station,
            w.log_date AS date,
            w.time,
            w.temp AS mean_temperature,
            w.humidity,
            w.airpressure,
            w.wind AS mean_wind_speed,
            w.winddirection,
            w.weather AS events
        FROM weatherdata w
        WHERE w.station = %s
        ORDER BY w.log_date DESC, w.time DESC
        LIMIT %s
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql, (airport_id, limit))
            rows = cursor.fetchall()
        return _clean_weather_rows(rows)
    except Exception as e:
        logger.warning("get_airport_weather error: %s", e)
        return []


def _clean_weather_rows(rows: list[dict]) -> list[dict]:
    """Convert Decimal/datetime weather fields to plain Python types."""
    cleaned = []
    for r in rows:
        cleaned.append({
            k: (float(v) if hasattr(v, "__float__") and not isinstance(v, (int, str, type(None))) else v)
            for k, v in r.items()
        })
    return cleaned


# ---------------------------------------------------------------------------
# Operational score
# ---------------------------------------------------------------------------

# Normalization reference values (based on dataset exploration)
# Airports with very high values will score near 100; smaller ones proportionally less.
_NORM = {
    "total_movements": 5000,
    "unique_connected_airports": 80,
    "airlines": 30,
    "aircraft_types": 200,
}


def calculate_operational_score(metrics: dict) -> dict:
    """
    Calculate a normalized operational score (0–100) from airport metrics.

    Formula (transparent, dataset-derived indicator — NOT an official rating):
        connectivity_component  = unique_connected / NORM * 35
        activity_component      = total_movements  / NORM * 30
        airline_component       = airlines         / NORM * 20
        aircraft_component      = aircraft_types   / NORM * 10
        data_component          =                         5  (if data available)

    Returns dict with total score and component breakdown.
    """
    def clamp(v: float) -> float:
        return min(1.0, max(0.0, v))

    conn_raw = clamp(metrics.get("unique_connected_airports", 0) / _NORM["unique_connected_airports"])
    act_raw = clamp(metrics.get("total_movements", 0) / _NORM["total_movements"])
    air_raw = clamp(metrics.get("airlines", 0) / _NORM["airlines"])
    acft_raw = clamp(metrics.get("aircraft_types", 0) / _NORM["aircraft_types"])
    data_raw = 1.0 if metrics.get("total_movements", 0) > 0 else 0.0

    connectivity_score = round(conn_raw * 35, 1)
    activity_score = round(act_raw * 30, 1)
    airline_score = round(air_raw * 20, 1)
    aircraft_score = round(acft_raw * 10, 1)
    data_score = round(data_raw * 5, 1)

    total = round(
        connectivity_score + activity_score + airline_score + aircraft_score + data_score,
        1,
    )

    return {
        "total": total,
        "components": {
            "connectivity": connectivity_score,
            "activity": activity_score,
            "airline_diversity": airline_score,
            "aircraft_diversity": aircraft_score,
            "data_availability": data_score,
        },
        "note": "Dataset-derived indicator. Not an official airport rating.",
    }


# ---------------------------------------------------------------------------
# Full airport profile
# ---------------------------------------------------------------------------

def get_airport_profile(airport_id: int) -> dict:
    """
    Build a complete airport profile combining geographic, traffic, weather,
    and scoring data.
    """
    airport = get_airport_by_id(airport_id)
    if not airport:
        return {"error": f"Airport {airport_id} not found"}

    metrics = get_flight_metrics(airport_id)
    score_data = calculate_operational_score(metrics)
    weather = get_airport_weather(airport_id, limit=5)
    hourly = get_hourly_activity(airport_id)

    return {
        "airport": airport,
        "metrics": metrics,
        "operational_score": score_data,
        "historical_weather": weather,
        "hourly_activity": hourly,
    }


def build_profile_text(profile: dict) -> str:
    """
    Generate a human-readable profile text for an airport.
    Used for TiDB Vector embeddings.
    """
    a = profile.get("airport", {})
    m = profile.get("metrics", {})
    s = profile.get("operational_score", {})

    name = a.get("name", "Unknown")
    city = a.get("city", "")
    country = a.get("country", "")
    iata = a.get("iata", "")
    icao = a.get("icao", "")
    lat = a.get("latitude", "")
    lon = a.get("longitude", "")

    total_mvt = m.get("total_movements", 0)
    connected = m.get("unique_connected_airports", 0)
    airlines = m.get("airlines", 0)
    aircraft_types = m.get("aircraft_types", 0)
    score = s.get("total", 0)

    text = (
        f"{name} ({iata}/{icao}), located in {city}, {country}. "
        f"Coordinates: {lat}, {lon}. "
        f"Historical dataset-derived indicators: "
        f"{total_mvt} total historical movements, "
        f"connected to {connected} airports in dataset, "
        f"{airlines} airlines recorded, "
        f"{aircraft_types} aircraft types observed. "
        f"Operational score (dataset indicator): {score}/100. "
    )

    weather = profile.get("historical_weather", [])
    if weather:
        recent = weather[0]
        temp = recent.get("mean_temperature")
        wind = recent.get("mean_wind_speed")
        vis = recent.get("mean_visibility")
        text += "Historical weather sample: "
        if temp is not None:
            text += f"mean temperature {temp}°C, "
        if wind is not None:
            text += f"mean wind speed {wind} km/h, "
        if vis is not None:
            text += f"mean visibility {vis} km. "

    return text.strip()
