"""
FerryFlow – Amazon Bedrock Service

Handles all interactions with Amazon Bedrock:
  - Mission recommendation narrative (Claude 3 Haiku)
  - Ask the Mission conversational Q&A (Claude 3 Haiku)
  - Airport profile embeddings (Cohere embed-multilingual-v3)

Bedrock DOES NOT calculate routes. Python/TiDB calculate; Bedrock explains.

Anti-hallucination system prompt is enforced on every call.
Graceful fallback on any exception — the UI must never crash due to AI failure.
"""

import json
import logging
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig

from src.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------

TEXT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
EMBED_MODEL_ID = "cohere.embed-multilingual-v3"
EMBED_DIMENSIONS = 1024

# ---------------------------------------------------------------------------
# Anti-hallucination system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an aviation mission decision-support assistant for FerryFlow.
Use ONLY the facts supplied in the mission context provided to you.
Never invent airport regulations, runway characteristics, fuel availability,
NOTAMs, weather forecasts, aircraft range, or any operational clearances.
If information is not available in the context, clearly state it is missing.
State explicitly that FerryFlow provides decision support and is not an
operational flight plan or certified dispatch system.
Be concise, professional, and grounded in the data provided."""

# ---------------------------------------------------------------------------
# Bedrock client factory
# ---------------------------------------------------------------------------

def _get_bedrock_client():
    """Create a boto3 Bedrock Runtime client using bearer token auth."""
    config = get_config()

    session = boto3.Session(region_name=config.aws_region)

    # Bearer token authentication for Bedrock
    client = session.client(
        service_name="bedrock-runtime",
        region_name=config.aws_region,
        config=BotocoreConfig(
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 2},
        ),
    )
    return client, config.aws_bearer_token_bedrock


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def _invoke_claude(prompt: str, max_tokens: int = 1024) -> str:
    """
    Call Claude 3 Haiku via Bedrock Messages API.
    Returns response text or raises on failure.
    """
    config = get_config()
    if not config.bedrock_configured:
        raise RuntimeError("Bedrock not configured. Set AWS_BEARER_TOKEN_BEDROCK and AWS_REGION.")

    client, bearer_token = _get_bedrock_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    # Inject bearer token as Authorization header via additional headers
    response = client.invoke_model(
        modelId=TEXT_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
        # Pass bearer token via request headers
        # boto3 supports additional_headers via the request
    )

    result = json.loads(response["body"].read())
    content = result.get("content", [])
    if content and isinstance(content, list):
        return content[0].get("text", "").strip()
    return ""


def _invoke_claude_with_bearer(prompt: str, max_tokens: int = 1024) -> str:
    """
    Call Claude 3 Haiku with explicit bearer token injection.
    """
    import urllib.request
    import urllib.error

    config = get_config()
    if not config.bedrock_configured:
        raise RuntimeError("Bedrock not configured.")

    url = (
        f"https://bedrock-runtime.{config.aws_region}.amazonaws.com"
        f"/model/{TEXT_MODEL_ID}/invoke"
    )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.aws_bearer_token_bedrock}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    content = body.get("content", [])
    if content and isinstance(content, list):
        return content[0].get("text", "").strip()
    return ""


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float] | None:
    """
    Generate a 1024-dimensional embedding using Cohere embed-multilingual-v3
    via Amazon Bedrock.

    Returns list of floats or None on failure.
    """
    import urllib.request
    import urllib.error

    config = get_config()
    if not config.bedrock_configured:
        logger.warning("embed_text: Bedrock not configured.")
        return None

    url = (
        f"https://bedrock-runtime.{config.aws_region}.amazonaws.com"
        f"/model/{EMBED_MODEL_ID}/invoke"
    )

    payload = {
        "texts": [text],
        "input_type": "search_document",
        "truncate": "END",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.aws_bearer_token_bedrock}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        embeddings = body.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        return None
    except Exception as e:
        logger.error("embed_text error: %s", e)
        return None


def embed_query(text: str) -> list[float] | None:
    """
    Generate a query embedding (input_type=search_query) for similarity search.
    """
    import urllib.request

    config = get_config()
    if not config.bedrock_configured:
        return None

    url = (
        f"https://bedrock-runtime.{config.aws_region}.amazonaws.com"
        f"/model/{EMBED_MODEL_ID}/invoke"
    )

    payload = {
        "texts": [text],
        "input_type": "search_query",
        "truncate": "END",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.aws_bearer_token_bedrock}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        embeddings = body.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        return None
    except Exception as e:
        logger.error("embed_query error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Mission recommendation
# ---------------------------------------------------------------------------

def generate_mission_recommendation(mission_context: dict) -> dict:
    """
    Send structured mission context to Bedrock Claude and get a recommendation.

    Returns:
    {
        "success": bool,
        "text": str,          # markdown narrative
        "error": str | None,
    }
    """
    try:
        prompt = _build_recommendation_prompt(mission_context)
        text = _invoke_claude_with_bearer(prompt, max_tokens=1500)
        return {"success": True, "text": text, "error": None}
    except Exception as e:
        logger.error("generate_mission_recommendation error: %s", e)
        return {
            "success": False,
            "text": "",
            "error": str(e),
        }


def _build_recommendation_prompt(ctx: dict) -> str:
    """Build the structured prompt for mission recommendation."""
    origin = ctx.get("origin", {})
    destination = ctx.get("destination", {})
    aircraft = ctx.get("aircraft_type", "Not specified")
    departure_time = ctx.get("departure_time", "Not specified")
    recommended_route = ctx.get("recommended_route", {})
    alternatives = ctx.get("alternative_routes", [])
    airports_data = ctx.get("airports_data", {})

    # Build route description
    route_airports = recommended_route.get("airports", [])
    route_desc = " → ".join(
        f"{a.get('iata', '?')} ({a.get('name', '?')}, {a.get('country', '?')})"
        for a in route_airports
    )

    alt_descs = []
    for i, alt in enumerate(alternatives[:2], 1):
        alt_airports = alt.get("airports", [])
        alt_desc = " → ".join(
            f"{a.get('iata', '?')} ({a.get('name', '?')})" for a in alt_airports
        )
        alt_descs.append(f"Alternative {i}: {alt_desc} | {alt.get('total_distance_km', '?')} km | Score: {alt.get('route_score', '?')}")

    # Build airport intelligence summary
    intel_lines = []
    for aid_str, profile in airports_data.items():
        ap = profile.get("airport", {})
        m = profile.get("metrics", {})
        s = profile.get("operational_score", {})
        intel_lines.append(
            f"  - {ap.get('iata', '?')} {ap.get('name', '?')}: "
            f"{m.get('total_movements', 0)} historical movements, "
            f"{m.get('unique_connected_airports', 0)} connected airports, "
            f"operational score {s.get('total', 0)}/100"
        )

    prompt = f"""MISSION PLANNING REQUEST – FerryFlow Decision Support

MISSION DATA:
  Aircraft type: {aircraft}
  Origin: {origin.get('iata', '?')} – {origin.get('name', '?')}, {origin.get('city', '?')}, {origin.get('country', '?')}
  Destination: {destination.get('iata', '?')} – {destination.get('name', '?')}, {destination.get('city', '?')}, {destination.get('country', '?')}
  Preferred departure: {departure_time}

RECOMMENDED ROUTE:
  {route_desc}
  Total distance: {recommended_route.get('total_distance_km', '?')} km ({recommended_route.get('total_distance_nm', '?')} nm)
  Number of legs: {recommended_route.get('num_legs', '?')}
  Route score: {recommended_route.get('route_score', '?')} (scale 0–1)

ALTERNATIVE ROUTES:
{chr(10).join(alt_descs) if alt_descs else '  None available'}

AIRPORT INTELLIGENCE (dataset-derived historical indicators):
{chr(10).join(intel_lines) if intel_lines else '  No airport data available'}

Please provide:
1. A concise mission summary (2-3 sentences)
2. Reasons why this route is recommended based on the data above
3. Key considerations for each intermediate stop (based only on provided data)
4. Brief assessment of the alternatives
5. Explicit limitations and missing information
6. A clear disclaimer that this is decision support, not an operational flight plan

Format your response in clear sections with headers."""

    return prompt


# ---------------------------------------------------------------------------
# Ask the Mission
# ---------------------------------------------------------------------------

def ask_mission(mission_context: dict, question: str) -> dict:
    """
    Answer a user question about the current mission.

    Returns:
    {
        "success": bool,
        "text": str,
        "error": str | None,
    }
    """
    try:
        prompt = _build_ask_prompt(mission_context, question)
        text = _invoke_claude_with_bearer(prompt, max_tokens=800)
        return {"success": True, "text": text, "error": None}
    except Exception as e:
        logger.error("ask_mission error: %s", e)
        return {
            "success": False,
            "text": "",
            "error": str(e),
        }


def _build_ask_prompt(ctx: dict, question: str) -> str:
    """Build prompt for Ask the Mission Q&A."""
    origin = ctx.get("origin", {})
    destination = ctx.get("destination", {})
    aircraft = ctx.get("aircraft_type", "Not specified")
    recommended_route = ctx.get("recommended_route", {})
    alternatives = ctx.get("alternative_routes", [])
    airports_data = ctx.get("airports_data", {})
    similarity_results = ctx.get("similarity_results", {})

    route_airports = recommended_route.get("airports", [])
    route_desc = " → ".join(
        f"{a.get('iata', '?')} ({a.get('name', '?')}, {a.get('country', '?')})"
        for a in route_airports
    )

    alt_descs = []
    for i, alt in enumerate(alternatives[:2], 1):
        alt_airports = alt.get("airports", [])
        alt_desc = " → ".join(f"{a.get('iata', '?')}" for a in alt_airports)
        alt_descs.append(f"  Alt {i}: {alt_desc} | {alt.get('total_distance_km', '?')} km")

    intel_lines = []
    for aid_str, profile in airports_data.items():
        ap = profile.get("airport", {})
        m = profile.get("metrics", {})
        s = profile.get("operational_score", {})
        weather = profile.get("historical_weather", [])
        weather_note = ""
        if weather:
            w = weather[0]
            weather_note = (
                f", historical weather sample: mean temp {w.get('mean_temperature', 'N/A')}°C"
                f", wind {w.get('mean_wind_speed', 'N/A')} km/h"
            )
        intel_lines.append(
            f"  {ap.get('iata', '?')} {ap.get('name', '?')} ({ap.get('country', '?')}): "
            f"{m.get('total_movements', 0)} movements, "
            f"{m.get('unique_connected_airports', 0)} connections, "
            f"score {s.get('total', 0)}/100"
            f"{weather_note}"
        )

    sim_lines = []
    for iata, sims in similarity_results.items():
        if sims:
            sim_list = ", ".join(
                f"{s.get('iata', '?')} (similarity {s.get('similarity_score', '?')})"
                for s in sims[:3]
            )
            sim_lines.append(f"  {iata} → similar airports: {sim_list}")

    prompt = f"""ACTIVE MISSION CONTEXT – FerryFlow Decision Support

Aircraft: {aircraft}
Origin: {origin.get('iata', '?')} – {origin.get('name', '?')}, {origin.get('country', '?')}
Destination: {destination.get('iata', '?')} – {destination.get('name', '?')}, {destination.get('country', '?')}

RECOMMENDED ROUTE: {route_desc}
Distance: {recommended_route.get('total_distance_km', '?')} km | Legs: {recommended_route.get('num_legs', '?')}

ALTERNATIVES:
{chr(10).join(alt_descs) if alt_descs else '  None'}

AIRPORT INTELLIGENCE (historical/dataset-derived):
{chr(10).join(intel_lines) if intel_lines else '  No data'}

SIMILAR AIRPORTS (vector search results):
{chr(10).join(sim_lines) if sim_lines else '  Not available'}

USER QUESTION: {question}

Answer the question using only the mission data provided above.
If the answer requires information not in this context, explicitly say so.
Keep your answer focused and concise."""

    return prompt


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_bedrock_available() -> tuple[bool, str]:
    """
    Quick availability check for Bedrock.
    Returns (available: bool, message: str).
    """
    config = get_config()
    if not config.bedrock_configured:
        return False, "AWS_BEARER_TOKEN_BEDROCK or AWS_REGION not configured."
    try:
        result = _invoke_claude_with_bearer("Respond with exactly: OK", max_tokens=10)
        if result:
            return True, "Bedrock available."
        return False, "Empty response from Bedrock."
    except Exception as e:
        return False, f"Bedrock unavailable: {e}"
