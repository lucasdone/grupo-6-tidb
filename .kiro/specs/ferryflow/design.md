# FerryFlow – Architecture & Design

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User (Browser)                        │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────────┐
│              Streamlit App (app.py)                      │
│         Running on AWS EC2 :8000                         │
├──────────────┬──────────────────┬───────────────────────┤
│  Mission     │  Airport Intel   │  Ask the Mission       │
│  Planner     │  + Vector Search │  Chat Q&A              │
└──────┬───────┴────────┬─────────┴────────┬──────────────┘
       │                │                  │
┌──────▼───────┐ ┌──────▼──────┐  ┌────────▼──────────────┐
│ Route Engine │ │ Airport     │  │ Bedrock Service        │
│ (NetworkX +  │ │ Service     │  │ (Claude Haiku)         │
│  Haversine)  │ │             │  │ ap-southeast-1         │
└──────┬───────┘ └──────┬──────┘  └────────────────────────┘
       │                │
┌──────▼────────────────▼──────────────────────────────────┐
│                  TiDB Cloud (AWS São Paulo)                │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  airportdb  │  │ airport_geo  │  │ airport_profiles │ │
│  │  (existing) │  │ flight/sched │  │ VECTOR(1024)     │ │
│  └─────────────┘  └──────────────┘  └──────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

## Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Frontend | Streamlit | Single-page app, polished UI |
| Language | Python 3.11+ | Modular service layer |
| Database | TiDB Cloud Starter (AWS São Paulo) | MySQL-compatible |
| Vector DB | TiDB Vector (VECTOR(1024)) | Same TiDB instance |
| AI | Amazon Bedrock Claude 3 Haiku | ap-southeast-1 |
| Embeddings | Amazon Bedrock Cohere multilingual v3 | 1024-dim |
| Graph | NetworkX | Route graph computation |
| Maps | Plotly | Route visualization |
| Deployment | AWS EC2 | Port 8000, 0.0.0.0 |

## Module Design

### src/config.py
- Load all environment variables
- Provide a single Config dataclass
- Never log credentials

### src/db.py
- `get_connection()` — returns a pymysql connection to TiDB
- `get_pool()` — optional connection pool
- Connection uses SSL (TiDB Cloud requires it)
- Retry logic for cold starts

### src/services/airport_service.py
- `get_airport_by_iata(iata)` — lookup single airport
- `search_airports(query)` — search by name/IATA/city
- `get_airport_profile(airport_id)` — full analytics profile
- `calculate_operational_score(metrics)` — normalized 0–100 score
- `get_airport_weather(airport_id)` — historical weather from weatherdata

#### Operational Score Formula
```
operational_score = (
    connectivity_score * 0.35     # unique connected airports (normalized)
  + activity_score    * 0.30     # total movements (normalized)
  + airline_score     * 0.20     # airline diversity
  + aircraft_score    * 0.10     # aircraft type diversity
  + data_score        * 0.05     # data availability bonus
) * 100
```

### src/services/route_service.py
- `build_route_graph(airport_ids)` — NetworkX DiGraph from flight history
- `find_routes(origin_id, dest_id, max_routes=3)` — candidate route search
- `haversine(lat1, lon1, lat2, lon2)` — great-circle distance in km
- `score_route(route, airport_profiles)` — rank candidates
- Route scoring:
  ```
  route_score = (
      distance_efficiency * 0.40
    + avg_connectivity    * 0.30
    + avg_activity        * 0.20
    - leg_penalty         * 0.10
  )
  ```

### src/services/mission_service.py
- `plan_mission(origin_iata, dest_iata, aircraft_type, departure_time)`
- Orchestrates: airport lookup → route finding → profile enrichment → Bedrock

### src/services/bedrock_service.py
- `generate_mission_recommendation(mission_context)` → str
- `ask_mission(mission_context, question)` → str
- Uses `anthropic.claude-3-haiku-20240307-v1:0`
- System prompt enforces no hallucination
- Graceful fallback on any exception

### src/services/vector_service.py
- `create_vector_table()` — DDL for airport_profiles with VECTOR(1024)
- `embed_text(text)` — calls Cohere embed-multilingual-v3 via Bedrock
- `store_airport_embedding(airport_id, profile_text, embedding)`
- `find_similar_airports(airport_id, top_k=5)` — VEC_COSINE_DISTANCE query
- Status check: `vector_table_populated()` → bool

## TiDB Schema Usage

### Existing Tables Used
```sql
airport          -- id, iata, icao, name, city, country
airport_geo      -- airport_id, latitude, longitude
flight           -- flight_id, from, to, departure, arrival, airline_id, airplane_id
flightschedule   -- route scheduling and operating days
airplane         -- airplane_id, type_id
airplane_type    -- type_id, name, capacity
weatherdata      -- weather observations (linked to airport)
airline          -- airline_id, name, iata_code
```

### New Table
```sql
CREATE TABLE airport_profiles (
    airport_id    INT PRIMARY KEY,
    iata          VARCHAR(10),
    profile_text  TEXT,
    embedding     VECTOR(1024),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Amazon Bedrock Integration

### Text Generation
- Model: `anthropic.claude-3-haiku-20240307-v1:0`
- Region: `ap-southeast-1`
- Auth: AWS Bearer Token via `AWS_BEARER_TOKEN_BEDROCK`
- Input: structured JSON mission context
- Output: markdown narrative

### Embeddings
- Model: `cohere.embed-multilingual-v3`
- Dimensions: 1024
- Input type: `search_document` for indexing, `search_query` for queries
- Used for: airport profile semantic similarity

### Anti-Hallucination System Prompt
```
You are an aviation mission decision-support assistant.
Use only the facts supplied in the mission context.
Never invent airport regulations, runway characteristics,
fuel availability, NOTAMs, weather forecasts or aircraft range.
Clearly identify missing information.
State explicitly that this is decision-support, not an operational flight plan.
```

## Deployment – AWS EC2

1. Launch EC2 (Amazon Linux 2 or Ubuntu 22.04), t3.small or larger
2. Open port 8000 in security group
3. SSH, clone repo, create .env
4. `pip install -r requirements.txt`
5. `streamlit run app.py --server.address 0.0.0.0 --server.port 8000`
6. Access via `http://<EC2_PUBLIC_IP>:8000`

No Nginx required for hackathon demo.

## Security Considerations
- All credentials in environment variables
- `.env` in `.gitignore`
- TiDB connection uses SSL
- Bedrock uses Bearer Token (not long-term IAM keys in code)
- No credentials logged

## Graceful Degradation
| Failure | Behavior |
|---------|----------|
| TiDB unavailable | Show error message, no crash |
| Bedrock unavailable | Show calculated route, AI section shows friendly message |
| Vector table empty | Show "Run build_airport_profiles.py first" message |
| Airport not found | Friendly validation message |
