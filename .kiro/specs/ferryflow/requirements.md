# FerryFlow – Product Requirements

## Overview

FerryFlow is an AI-powered decision-support platform for ferry and cargo mission planning.
It helps ferry pilots, aircraft operators, and logistics companies plan aircraft delivery and
cargo missions across countries by combining historical airport data, route analytics, and
AI-generated recommendations.

**Tagline:** AI Mission Planner for Ferry & Cargo Aviation

## Target Users

- Aircraft manufacturers
- Aircraft owners / lessors / brokers
- Ferry flight operators
- Air cargo operators
- Logistics companies
- Ferry pilots

## Core Problem Statement

Ferry pilots and cargo aviation operators need to plan aircraft delivery and cargo missions
across countries. Airport characteristics, route complexity, traffic, aircraft characteristics
and local operational constraints make manual planning difficult.

## MVP Acceptance Criteria

### AC-1: Mission Planner Form
- [ ] User can select an aircraft type from available dataset
- [ ] User can select an origin airport (IATA/name search)
- [ ] User can select a destination airport (IATA/name search)
- [ ] User can optionally enter a preferred departure time
- [ ] User can trigger mission planning with a "Plan Mission" button

### AC-2: Route Results
- [ ] System displays a recommended route (list of airports, leg by leg)
- [ ] System displays up to 2 alternative routes
- [ ] System shows number of legs per route
- [ ] System shows airports involved with names and IATA codes
- [ ] System shows approximate total great-circle distance in km/nm
- [ ] System shows historical traffic/activity indicators per airport
- [ ] System shows an airport operational score (0–100, dataset-derived)
- [ ] System shows available historical weather information
- [ ] System shows an AI-generated recommendation narrative
- [ ] System shows explicit risks, limitations, and assumptions

### AC-3: Route Visualization
- [ ] Route is plotted on a world map (Plotly)
- [ ] Intermediate airports are visible as waypoints

### AC-4: Airport Intelligence
- [ ] Each airport in the route shows: name, IATA, country, city, coordinates
- [ ] Historical movement counts (inbound + outbound)
- [ ] Number of unique connected airports
- [ ] Number of airlines serving
- [ ] Aircraft diversity index
- [ ] Operational score with component breakdown

### AC-5: Similar Airports (TiDB Vector)
- [ ] For any airport in the route, user can request semantically similar airports
- [ ] Similarity uses VEC_COSINE_DISTANCE on 1024-dim Cohere embeddings
- [ ] Similarity score is displayed
- [ ] User can substitute a similar airport as an alternative stop
- [ ] If vector table not populated, system shows setup status message

### AC-6: Ask the Mission (Conversational Q&A)
- [ ] After planning, user can ask free-text questions about the mission
- [ ] System provides answers grounded in current mission context
- [ ] Supported question types include route rationale, airport comparison,
      alternative stops, operational explanations

### AC-7: AI Recommendation (Amazon Bedrock)
- [ ] Bedrock generates mission summary and recommendation narrative
- [ ] Bedrock receives structured mission context (not raw free text)
- [ ] Bedrock explicitly states limitations and missing information
- [ ] If Bedrock fails, UI shows calculated data with a friendly fallback message
- [ ] Bedrock never invents airport regulations, fuel availability, NOTAMs,
      weather forecasts, or aircraft range

### AC-8: Disclaimer
- [ ] Prominent UI disclaimer: "FerryFlow MVP provides historical/data-driven
      decision support and is not an operational flight plan."

### AC-9: Non-Functional
- [ ] Application runs on `streamlit run app.py --server.address 0.0.0.0 --server.port 8000`
- [ ] All credentials via environment variables, never hardcoded
- [ ] .env never committed
- [ ] Graceful error handling for DB and AI failures
- [ ] Startup completes even without Bedrock or Vector credentials

## Out of Scope for MVP

- Real-time weather or NOTAM integration
- Actual aircraft performance / range calculation
- Regulatory compliance checking
- Flight dispatch or official operational planning
- Real-time ATC data
