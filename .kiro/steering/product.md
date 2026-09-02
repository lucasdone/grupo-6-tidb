---
inclusion: always
---

# FerryFlow – Product Steering

## What is FerryFlow?

FerryFlow is an AI-powered decision-support platform for ferry and cargo mission planning,
built for the TiDB × AWS Hackathon. It combines historical airport data from TiDB Cloud,
graph-based route analytics, TiDB Vector semantic search, and Amazon Bedrock AI to help
operators plan aircraft delivery and cargo missions across countries.

## Target Users

- Aircraft manufacturers
- Aircraft owners, lessors, and brokers
- Ferry flight operators
- Air cargo operators
- Logistics companies
- Ferry pilots

## Decision-Support Positioning

FerryFlow is explicitly a **decision-support tool**, not a certified or operational system.

- It uses **historical and dataset-derived data** from the airportdb database.
- It does NOT provide real-time data, official dispatch, or regulatory clearance.
- All route recommendations are based on historical connectivity patterns and
  approximate great-circle distances.
- Operational scores are dataset-derived indicators, not official airport ratings.
- Weather information shown is historical, not a forecast or METAR/TAF.

## Safety and Accuracy Constraints

The following information is **NOT available** in the airportdb dataset and must NEVER be
invented, assumed, or implied by FerryFlow or any AI component:

- Aircraft performance, range, or fuel burn
- Runway length, surface, or compatibility
- Fuel availability at specific airports
- NOTAMs or airspace restrictions
- Customs, immigration, or overflight authorizations
- Real-time or forecast weather
- Current airport operational status
- Any regulatory compliance determination

When these topics arise, the system must:
1. Explicitly state the information is not available in the dataset
2. Recommend the user consult official sources (e.g., Jeppesen, official aeronautical charts)
3. Not substitute a guess or hallucinated value

## Required Disclaimer

Every page of the FerryFlow application must display:

> **FerryFlow MVP provides historical/data-driven decision support and is not an
> operational flight plan.**

## AI Behavior Guidelines

- Bedrock must only use facts supplied in the structured mission context
- Bedrock must explicitly call out missing information rather than filling gaps
- Responses must be grounded, concise, and aviation-professional in tone
- The "Ask the Mission" feature must stay scoped to the active mission context
- Bedrock must not be used to calculate routes — Python/TiDB calculate, Bedrock explains

## Data Language Standards

Always use dataset-qualified language:

| Instead of | Use |
|-----------|-----|
| "airport traffic" | "historical airport activity" |
| "current conditions" | "dataset-derived indicator" |
| "weather forecast" | "historical weather observations" |
| "runway available" | "airport present in historical dataset" |
| "aircraft compatible" | "aircraft type present in dataset" |

## Technical Stack Context

- **Frontend:** Streamlit (Python)
- **Database:** TiDB Cloud – airportdb (MySQL-compatible, AWS São Paulo)
- **Vector Search:** TiDB Vector VECTOR(1024) with VEC_COSINE_DISTANCE
- **AI:** Amazon Bedrock, Claude 3 Haiku, region ap-southeast-1
- **Embeddings:** Cohere embed-multilingual-v3 (1024-dim) via Bedrock
- **Deployment:** AWS EC2, port 8000
