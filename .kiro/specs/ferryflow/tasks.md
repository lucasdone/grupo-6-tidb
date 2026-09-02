# FerryFlow – Implementation Tasks

## Stage 1 – Repository & Scaffolding
- [x] Inspect existing repository structure
- [x] Check git status, branch, remote
- [x] Create .kiro/specs/ferryflow/requirements.md
- [x] Create .kiro/specs/ferryflow/design.md
- [x] Create .kiro/specs/ferryflow/tasks.md
- [x] Create .kiro/steering/product.md
- [x] Create .gitignore
- [x] Create .env.example
- [x] Create requirements.txt

## Stage 2 – Core Configuration & Database
- [x] Create src/config.py – environment variable loading
- [x] Create src/db.py – TiDB connection with SSL and retry
- [x] Create src/__init__.py
- [x] Create src/services/__init__.py
- [x] Create src/models/__init__.py
- [x] Create src/utils/__init__.py

## Stage 3 – Airport Analytics
- [x] Create src/services/airport_service.py
  - [x] search_airports(query)
  - [x] get_airport_by_iata(iata)
  - [x] get_airport_profile(airport_id)
  - [x] calculate_operational_score(metrics)
  - [x] get_airport_weather(airport_id)

## Stage 4 – Route Engine
- [x] Create src/services/route_service.py
  - [x] haversine(lat1, lon1, lat2, lon2)
  - [x] build_route_graph()
  - [x] find_routes(origin_id, dest_id)
  - [x] score_route(route, profiles)
  - [x] enrich_route(route)

## Stage 5 – Mission Orchestration
- [x] Create src/services/mission_service.py
  - [x] plan_mission(origin_iata, dest_iata, aircraft_type, departure_time)
  - [x] Combine route + profiles + weather + Bedrock

## Stage 6 – Amazon Bedrock
- [x] Create src/services/bedrock_service.py
  - [x] generate_mission_recommendation(mission_context)
  - [x] ask_mission(mission_context, question)
  - [x] Graceful fallback on exception

## Stage 7 – TiDB Vector
- [x] Create src/services/vector_service.py
  - [x] create_vector_table()
  - [x] embed_text(text)
  - [x] store_airport_embedding(airport_id, profile_text, embedding)
  - [x] find_similar_airports(airport_id, top_k)
  - [x] vector_table_populated()
- [x] Create sql/vector_schema.sql

## Stage 8 – Streamlit UI
- [x] Create app.py
  - [x] Sidebar navigation
  - [x] Mission Planner form
  - [x] Route results with map (Plotly)
  - [x] Airport Intelligence cards
  - [x] AI Recommendation section
  - [x] Alternative Routes section
  - [x] Similar Airports section
  - [x] Ask the Mission chat
  - [x] Disclaimer banner

## Stage 9 – Scripts
- [x] Create scripts/inspect_database.py
- [x] Create scripts/build_airport_profiles.py
- [x] Create scripts/test_connections.py

## Stage 10 – SQL
- [x] Create sql/analytics.sql
- [x] Create sql/vector_schema.sql

## Stage 11 – Documentation
- [x] Create README.md with Mermaid diagram
- [x] Create SUBMISSION.md

## Stage 12 – Quality & Delivery
- [x] Run Python syntax checks (py_compile)
- [x] Verify no secrets in tracked files
- [x] git add, commit, push
