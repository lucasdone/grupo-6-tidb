"""
FerryFlow – Route Engine

Builds a route graph from historical flight data and computes candidate
routes between origin and destination using NetworkX + Haversine.

All distances are approximate great-circle distances.
Historical connectivity is derived from the airportdb dataset.

DISCLAIMER: This engine does NOT account for aircraft range, runway
compatibility, fuel availability, NOTAMs, or regulatory constraints.
"""

import logging
import math
from typing import Any

import networkx as nx

from src.db import db_cursor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0
KM_TO_NM = 0.539957


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance in kilometres between two points
    on the Earth using the Haversine formula.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def km_to_nm(km: float) -> float:
    return round(km * KM_TO_NM, 1)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _load_historical_edges() -> list[dict]:
    """
    Load all historical flight routes from TiDB as directed edges.
    Returns list of {from_id, to_id, flight_count}.
    """
    sql = """
        SELECT
            f.`from`       AS from_id,
            f.`to`         AS to_id,
            COUNT(*)       AS flight_count
        FROM flight f
        WHERE f.`from` IS NOT NULL AND f.`to` IS NOT NULL
        GROUP BY f.`from`, f.`to`
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()
    except Exception as e:
        logger.error("_load_historical_edges error: %s", e)
        return []


def _load_airport_coords() -> dict[int, dict]:
    """
    Load airport coordinates indexed by airport_id.
    Returns {airport_id: {lat, lon, iata, name, city, country}}.
    """
    sql = """
        SELECT
            a.airport_id,
            a.iata,
            a.name,
            g.city,
            g.country,
            g.latitude  AS lat,
            g.longitude AS lon
        FROM airport a
        INNER JOIN airport_geo g ON a.airport_id = g.airport_id
        WHERE g.latitude IS NOT NULL AND g.longitude IS NOT NULL
    """
    try:
        with db_cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return {
            int(r["airport_id"]): {
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "iata": r.get("iata", ""),
                "name": r.get("name", ""),
                "city": r.get("city", ""),
                "country": r.get("country", ""),
            }
            for r in rows
        }
    except Exception as e:
        logger.error("_load_airport_coords error: %s", e)
        return {}


def build_route_graph(
    coords: dict[int, dict],
    edges: list[dict],
) -> nx.DiGraph:
    """
    Construct a weighted directed graph where:
      - nodes are airport_ids with lat/lon attributes
      - edges represent historical flight links weighted by great-circle distance
    """
    G = nx.DiGraph()

    for aid, info in coords.items():
        G.add_node(aid, **info)

    for edge in edges:
        from_id = int(edge["from_id"])
        to_id = int(edge["to_id"])
        flight_count = int(edge.get("flight_count", 1))

        if from_id not in coords or to_id not in coords:
            continue

        dist_km = haversine_km(
            coords[from_id]["lat"], coords[from_id]["lon"],
            coords[to_id]["lat"], coords[to_id]["lon"],
        )
        G.add_edge(
            from_id,
            to_id,
            distance_km=dist_km,
            flight_count=flight_count,
            weight=dist_km,  # used by shortest-path algorithms
        )

    return G


# ---------------------------------------------------------------------------
# Route scoring
# ---------------------------------------------------------------------------

# Explicit weights for route candidate scoring
ROUTE_SCORE_WEIGHTS = {
    "distance_efficiency": 0.40,
    "avg_connectivity": 0.30,
    "avg_activity": 0.20,
    "leg_penalty": 0.10,
}

# Normalization references
_MAX_REASONABLE_DISTANCE_KM = 20_000
_MAX_CONNECTIVITY = 80
_MAX_MOVEMENTS = 5000


def score_route(
    route_airport_ids: list[int],
    coords: dict[int, dict],
    airport_profiles: dict[int, dict],
    direct_distance_km: float,
) -> float:
    """
    Score a candidate route on a 0–1 scale.

    Higher is better.

    Components:
      distance_efficiency:  direct_dist / actual_dist  (penalises detours)
      avg_connectivity:     mean(connected_airports / NORM) across stops
      avg_activity:         mean(total_movements / NORM) across stops
      leg_penalty:          1 - (legs - 1) / MAX_LEGS  (penalises extra legs)
    """
    if len(route_airport_ids) < 2:
        return 0.0

    # Actual route distance
    actual_dist_km = 0.0
    for i in range(len(route_airport_ids) - 1):
        a = route_airport_ids[i]
        b = route_airport_ids[i + 1]
        if a in coords and b in coords:
            actual_dist_km += haversine_km(
                coords[a]["lat"], coords[a]["lon"],
                coords[b]["lat"], coords[b]["lon"],
            )

    if actual_dist_km == 0:
        return 0.0

    distance_efficiency = min(1.0, direct_distance_km / actual_dist_km)

    # Connectivity and activity scores from profiles
    connectivity_scores = []
    activity_scores = []
    for aid in route_airport_ids:
        profile = airport_profiles.get(aid, {})
        metrics = profile.get("metrics", {})
        connectivity_scores.append(
            min(1.0, metrics.get("unique_connected_airports", 0) / _MAX_CONNECTIVITY)
        )
        activity_scores.append(
            min(1.0, metrics.get("total_movements", 0) / _MAX_MOVEMENTS)
        )

    avg_connectivity = sum(connectivity_scores) / len(connectivity_scores)
    avg_activity = sum(activity_scores) / len(activity_scores)

    # Leg penalty: 1 leg = no penalty, each extra leg reduces score
    num_legs = len(route_airport_ids) - 1
    max_legs = 6
    leg_penalty = max(0.0, 1.0 - (num_legs - 1) / max_legs)

    score = (
        distance_efficiency * ROUTE_SCORE_WEIGHTS["distance_efficiency"]
        + avg_connectivity * ROUTE_SCORE_WEIGHTS["avg_connectivity"]
        + avg_activity * ROUTE_SCORE_WEIGHTS["avg_activity"]
        + leg_penalty * ROUTE_SCORE_WEIGHTS["leg_penalty"]
    )

    return round(score, 4)


# ---------------------------------------------------------------------------
# Route finder
# ---------------------------------------------------------------------------

def find_routes(
    origin_id: int,
    dest_id: int,
    G: nx.DiGraph,
    coords: dict[int, dict],
    airport_profiles: dict[int, dict],
    max_routes: int = 3,
    max_hops: int = 4,
) -> list[dict]:
    """
    Find up to max_routes candidate routes from origin to destination.

    Strategy:
      1. Try shortest path by distance (NetworkX dijkstra on weight=distance_km)
      2. Try simple paths (up to max_hops) and pick diverse candidates
      3. Deduplicate and score all candidates

    Returns list of route dicts sorted by score descending.
    """
    if origin_id not in G or dest_id not in G:
        logger.warning(
            "find_routes: origin %s or dest %s not in graph", origin_id, dest_id
        )
        return []

    direct_distance_km = 0.0
    if origin_id in coords and dest_id in coords:
        direct_distance_km = haversine_km(
            coords[origin_id]["lat"], coords[origin_id]["lon"],
            coords[dest_id]["lat"], coords[dest_id]["lon"],
        )

    candidate_paths: list[list[int]] = []

    # 1. Dijkstra shortest distance path
    try:
        path = nx.dijkstra_path(G, origin_id, dest_id, weight="distance_km")
        candidate_paths.append(path)
    except nx.NetworkXNoPath:
        logger.info("No direct Dijkstra path from %s to %s", origin_id, dest_id)
    except Exception as e:
        logger.warning("Dijkstra error: %s", e)

    # 2. Simple paths up to cutoff – collect several and rank
    try:
        simple = nx.all_simple_paths(G, origin_id, dest_id, cutoff=max_hops)
        count = 0
        for path in simple:
            if path not in candidate_paths:
                candidate_paths.append(path)
            count += 1
            if count >= 50:  # cap exploration
                break
    except Exception as e:
        logger.warning("simple_paths error: %s", e)

    # 3. If still no routes, try geographic greedy fallback
    if not candidate_paths:
        fallback = _geographic_greedy_route(origin_id, dest_id, G, coords, max_hops)
        if fallback:
            candidate_paths.append(fallback)

    if not candidate_paths:
        return []

    # Score all candidates
    scored: list[dict] = []
    seen: set[tuple] = set()

    for path in candidate_paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)

        route_score = score_route(path, coords, airport_profiles, direct_distance_km)
        route_dist = _route_distance_km(path, coords)
        enriched = _enrich_path(path, coords, airport_profiles)

        scored.append({
            "airports": enriched,
            "airport_ids": path,
            "num_legs": len(path) - 1,
            "total_distance_km": round(route_dist, 1),
            "total_distance_nm": km_to_nm(route_dist),
            "direct_distance_km": round(direct_distance_km, 1),
            "route_score": route_score,
            "score_weights": ROUTE_SCORE_WEIGHTS,
        })

    # Sort by score descending and return top N
    scored.sort(key=lambda r: r["route_score"], reverse=True)

    # Ensure diversity: if top routes are identical except minor variation, keep variety
    return scored[:max_routes]


def _route_distance_km(path: list[int], coords: dict[int, dict]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if a in coords and b in coords:
            total += haversine_km(
                coords[a]["lat"], coords[a]["lon"],
                coords[b]["lat"], coords[b]["lon"],
            )
    return total


def _enrich_path(
    path: list[int],
    coords: dict[int, dict],
    airport_profiles: dict[int, dict],
) -> list[dict]:
    """Add readable metadata to each airport in the path."""
    enriched = []
    for i, aid in enumerate(path):
        info = coords.get(aid, {})
        profile = airport_profiles.get(aid, {})
        metrics = profile.get("metrics", {})
        score = profile.get("operational_score", {})

        leg_dist_km = 0.0
        if i > 0:
            prev = path[i - 1]
            if prev in coords and aid in coords:
                leg_dist_km = haversine_km(
                    coords[prev]["lat"], coords[prev]["lon"],
                    info.get("lat", 0), info.get("lon", 0),
                )

        enriched.append({
            "airport_id": aid,
            "iata": info.get("iata", ""),
            "name": info.get("name", ""),
            "city": info.get("city", ""),
            "country": info.get("country", ""),
            "lat": info.get("lat"),
            "lon": info.get("lon"),
            "leg_distance_km": round(leg_dist_km, 1),
            "leg_distance_nm": km_to_nm(leg_dist_km),
            "total_movements": metrics.get("total_movements", 0),
            "unique_connected": metrics.get("unique_connected_airports", 0),
            "airlines": metrics.get("airlines", 0),
            "operational_score": score.get("total", 0),
            "role": "origin" if i == 0 else ("destination" if i == len(path) - 1 else "intermediate"),
        })
    return enriched


def _geographic_greedy_route(
    origin_id: int,
    dest_id: int,
    G: nx.DiGraph,
    coords: dict[int, dict],
    max_hops: int,
) -> list[int] | None:
    """
    Greedy fallback: repeatedly pick the neighbour that is closest
    to the destination, until destination is reached or max_hops exceeded.
    Only traverses edges present in graph (historical connectivity).
    """
    if origin_id not in coords or dest_id not in coords:
        return None

    dest_lat = coords[dest_id]["lat"]
    dest_lon = coords[dest_id]["lon"]

    path = [origin_id]
    visited = {origin_id}

    for _ in range(max_hops):
        current = path[-1]
        if current == dest_id:
            break

        neighbors = list(G.successors(current))
        if not neighbors:
            break

        # Pick neighbour minimising distance to destination
        best = None
        best_dist = float("inf")
        for nb in neighbors:
            if nb in visited:
                continue
            if nb not in coords:
                continue
            d = haversine_km(
                coords[nb]["lat"], coords[nb]["lon"],
                dest_lat, dest_lon,
            )
            if d < best_dist:
                best_dist = d
                best = nb

        if best is None:
            break

        path.append(best)
        visited.add(best)

        if best == dest_id:
            break

    # Ensure destination is appended if we got close but graph has edge
    if path[-1] != dest_id:
        if G.has_edge(path[-1], dest_id):
            path.append(dest_id)
        else:
            return None  # Could not connect

    return path if len(path) >= 2 else None


# ---------------------------------------------------------------------------
# Top-level: load data and find routes
# ---------------------------------------------------------------------------

def compute_mission_routes(
    origin_id: int,
    dest_id: int,
    airport_profiles: dict[int, dict],
    max_routes: int = 3,
) -> dict:
    """
    Main entry point called by mission_service.

    Loads historical edges and coordinates, builds the graph,
    and returns candidate routes.

    Returns:
    {
        "routes": [...],
        "graph_stats": {"nodes": int, "edges": int},
        "error": str | None,
    }
    """
    try:
        coords = _load_airport_coords()
        edges = _load_historical_edges()

        if not coords:
            return {"routes": [], "graph_stats": {}, "error": "No airport coordinate data available."}

        G = build_route_graph(coords, edges)

        graph_stats = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }

        routes = find_routes(
            origin_id, dest_id, G, coords, airport_profiles, max_routes=max_routes
        )

        return {
            "routes": routes,
            "graph_stats": graph_stats,
            "error": None,
        }

    except Exception as e:
        logger.error("compute_mission_routes error: %s", e)
        return {
            "routes": [],
            "graph_stats": {},
            "error": str(e),
        }
