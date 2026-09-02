"""
Airlog.ai (FerryFlow) – Streamlit UI

Decision-support planner for ferry and cargo missions.
Data: TiDB Cloud (airportdb) · Vector search: TiDB VECTOR · AI: Amazon Bedrock.

DISCLAIMER: Historical, dataset-derived decision support. Not an operational flight plan.
"""

import logging
import os
import sys
from datetime import datetime, time as dtime

# Ensure the project root is importable regardless of the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_config
from src.db import test_connection
from src.services.mission_service import get_aircraft_types, plan_mission
from src.services.airport_service import search_airports
from src.services.bedrock_service import ask_mission, check_bedrock_available
from src.services.vector_service import (
    find_similar_airports_by_query,
    find_similar_for_route,
    vector_table_populated,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Airlog.ai – Ferry & Cargo Mission Planner",
    page_icon="✈️",
    layout="wide",
)

DISCLAIMER = (
    "**Aviso:** o Airlog.ai fornece apoio à decisão baseado em dados históricos do dataset "
    "airportdb. Não é um plano de voo operacional, não substitui despacho, briefing "
    "meteorológico (METAR/TAF) nem autorização regulatória."
)


# ---------------------------------------------------------------------------
# Cached status checks
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _tidb_status() -> tuple[bool, str]:
    return test_connection()


@st.cache_data(ttl=300, show_spinner=False)
def _bedrock_status() -> tuple[bool, str]:
    return check_bedrock_available()


@st.cache_data(ttl=300, show_spinner=False)
def _vector_status() -> dict:
    return vector_table_populated()


@st.cache_data(ttl=3600, show_spinner=False)
def _aircraft_options() -> list[str]:
    rows = get_aircraft_types(limit=400)
    return sorted({r["identifier"] for r in rows if r.get("identifier")})


@st.cache_data(ttl=600, show_spinner=False)
def _search(query: str) -> list[dict]:
    if not query or len(query) < 2:
        return []
    return search_airports(query, limit=15)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.title("✈️ Airlog.ai")
    st.sidebar.caption("Planejamento de missões ferry & cargo")

    st.sidebar.subheader("Status dos serviços")
    ok, msg = _tidb_status()
    st.sidebar.markdown(f"{'🟢' if ok else '🔴'} **TiDB Cloud** – {'conectado' if ok else 'indisponível'}")
    if not ok:
        st.sidebar.caption(msg)

    vec = _vector_status()
    st.sidebar.markdown(
        f"{'🟢' if vec.get('populated') else '🟡'} **TiDB Vector** – "
        f"{vec.get('count', 0)} perfis com embedding"
    )
    if not vec.get("populated"):
        st.sidebar.caption(vec.get("message", ""))

    ok_b, msg_b = _bedrock_status()
    st.sidebar.markdown(f"{'🟢' if ok_b else '🔴'} **Amazon Bedrock** – {'disponível' if ok_b else 'indisponível'}")
    if not ok_b:
        st.sidebar.caption(msg_b)

    cfg = get_config()
    st.sidebar.caption(f"Região Bedrock: `{cfg.aws_region}` · DB: `{cfg.tidb_database}`")

    st.sidebar.divider()
    st.sidebar.subheader("Busca rápida de aeroporto")
    q = st.sidebar.text_input("IATA, ICAO, nome ou cidade", placeholder="ex.: GRU, Campinas, Miami")
    if q:
        rows = _search(q)
        if rows:
            for r in rows:
                st.sidebar.markdown(
                    f"**{r.get('iata') or '—'}** / {r.get('icao') or '—'} · {r.get('name', '')}  \n"
                    f"<span style='color:gray'>{r.get('city', '')}, {r.get('country', '')}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.sidebar.info("Nenhum aeroporto encontrado.")

    st.sidebar.divider()
    st.sidebar.caption("Time Rainbow 6 · TiDB × AWS Hackathon")


# ---------------------------------------------------------------------------
# Mission form
# ---------------------------------------------------------------------------

def render_form() -> dict | None:
    st.title("Planejador de missão")
    st.markdown(
        "Informe origem, destino, aeronave e horário. O Airlog.ai cruza a base histórica de "
        "aviação no TiDB para sugerir rota, janelas de operação e aeroportos alternativos, "
        "e gera uma recomendação com Amazon Bedrock."
    )

    with st.form("mission_form"):
        c1, c2, c3 = st.columns(3)
        origin = c1.text_input("Origem (IATA)", value="GRU", max_chars=3).upper()
        dest = c2.text_input("Destino (IATA)", value="MIA", max_chars=3).upper()
        aircraft = c3.selectbox("Tipo de aeronave", ["Não especificado"] + _aircraft_options())

        c4, c5, c6 = st.columns(3)
        dep_date = c4.date_input("Data de partida", value=datetime.utcnow().date())
        dep_time = c5.time_input("Hora de partida (UTC)", value=dtime(9, 0))
        payload = c6.selectbox(
            "Tipo de missão",
            ["Ferry (entrega da própria aeronave)", "Carga geral", "Carga perecível", "Carga perigosa (DG)"],
        )

        include_ai = st.checkbox("Gerar recomendação com IA (Amazon Bedrock)", value=True)
        submitted = st.form_submit_button("Planejar missão", type="primary", use_container_width=True)

    if not submitted:
        return None
    if not origin or not dest:
        st.error("Informe os códigos IATA de origem e destino.")
        return None

    return {
        "origin": origin,
        "dest": dest,
        "aircraft": aircraft,
        "departure": f"{dep_date.isoformat()} {dep_time.strftime('%H:%M')} UTC",
        "payload": payload,
        "include_ai": include_ai,
    }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def render_route_map(routes: list[dict]) -> None:
    fig = go.Figure()
    for idx, route in enumerate(routes):
        aps = route.get("airports", [])
        lats = [a["lat"] for a in aps]
        lons = [a["lon"] for a in aps]
        labels = [f"{a.get('iata') or a.get('name')}" for a in aps]
        is_main = idx == 0
        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="lines+markers+text",
                text=labels,
                textposition="top center",
                name="Rota recomendada" if is_main else f"Alternativa {idx}",
                line=dict(width=3 if is_main else 1.5, color="#e4572e" if is_main else "#5c7cfa"),
                marker=dict(size=9 if is_main else 6),
                opacity=1.0 if is_main else 0.6,
            )
        )
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        landcolor="#f2f2ef",
        countrycolor="#c9c9c4",
        fitbounds="locations",
    )
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def render_route_table(route: dict) -> None:
    rows = []
    for a in route.get("airports", []):
        rows.append(
            {
                "Papel": a.get("role", ""),
                "IATA": a.get("iata") or "—",
                "Aeroporto": a.get("name", ""),
                "Cidade": f"{a.get('city', '')}, {a.get('country', '')}",
                "Trecho (km)": a.get("leg_distance_km", 0),
                "Trecho (NM)": a.get("leg_distance_nm", 0),
                "Movimentos hist.": a.get("total_movements", 0),
                "Score operacional": a.get("operational_score", 0),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_airport_cards(mission: dict) -> None:
    st.subheader("Inteligência de aeroportos")
    route = mission.get("recommended_route") or {}
    airports_data = mission.get("airports_data", {})
    aps = route.get("airports", [])
    if not aps:
        st.info("Sem aeroportos para exibir.")
        return

    cols = st.columns(min(len(aps), 4))
    for i, a in enumerate(aps):
        profile = airports_data.get(a.get("iata"), {})
        metrics = profile.get("metrics", {})
        score = profile.get("operational_score", {})
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"### {a.get('iata') or '—'} · {a.get('role', '')}")
                st.caption(f"{a.get('name', '')} — {a.get('city', '')}, {a.get('country', '')}")
                st.metric("Score operacional (indicador)", f"{score.get('total', a.get('operational_score', 0))}/100")
                m1, m2 = st.columns(2)
                m1.metric("Movimentos", metrics.get("total_movements", a.get("total_movements", 0)))
                m2.metric("Conexões", metrics.get("unique_connected_airports", a.get("unique_connected", 0)))
                m1.metric("Cias aéreas", metrics.get("airlines", a.get("airlines", 0)))
                m2.metric("Tipos de aeronave", metrics.get("aircraft_types", 0))

                hourly = profile.get("hourly_activity", [])
                if hourly:
                    df = pd.DataFrame(hourly).set_index("hour")
                    quiet = min(hourly, key=lambda h: h["count"])
                    st.caption(f"Partidas por hora (UTC). Hora mais tranquila: {quiet['hour']:02d}h")
                    st.bar_chart(df, height=140)
                else:
                    st.caption("Sem histórico horário de partidas no dataset.")

                weather = profile.get("historical_weather", [])
                if weather:
                    w = weather[0]
                    st.caption(
                        f"Clima histórico ({w.get('date')}): {w.get('mean_temperature')}°C, "
                        f"vento {w.get('mean_wind_speed')} km/h, {w.get('events') or 'sem eventos'}"
                    )


def render_ai_section(mission: dict, form: dict) -> None:
    st.subheader("Recomendação da IA (Amazon Bedrock)")
    ai = mission.get("ai_recommendation", {})
    if ai.get("success") and ai.get("text"):
        st.markdown(ai["text"])
    elif not form["include_ai"]:
        st.info("Recomendação por IA desativada no formulário.")
    else:
        st.warning(
            "Recomendação por IA indisponível no momento. A rota calculada acima continua válida "
            "como apoio à decisão."
        )
        if ai.get("error"):
            with st.expander("Detalhes do erro"):
                st.code(str(ai["error"]))


def render_alternatives(mission: dict) -> None:
    alts = mission.get("alternative_routes", [])
    st.subheader("Rotas alternativas")
    if not alts:
        st.caption("Nenhuma rota alternativa encontrada na conectividade histórica do dataset.")
        return
    for i, r in enumerate(alts, start=1):
        path = " → ".join(a.get("iata") or a.get("name", "?") for a in r.get("airports", []))
        with st.expander(f"Alternativa {i}: {path} · {r.get('total_distance_km', 0)} km · score {r.get('route_score', 0)}"):
            render_route_table(r)


def render_similar(mission: dict) -> None:
    st.subheader("Aeroportos semelhantes (TiDB Vector)")
    vec = _vector_status()
    if not vec.get("populated"):
        st.info(
            "A tabela de embeddings ainda não foi populada. Execute "
            "`python scripts/build_airport_profiles.py` para gerar os perfis com Bedrock "
            "e habilitar a busca vetorial."
        )
        return

    route = mission.get("recommended_route") or {}
    ids = route.get("airport_ids", [])
    by_id = {a["airport_id"]: a for a in route.get("airports", [])}
    results = find_similar_for_route(ids, top_k=3)
    for aid, sims in results.items():
        label = by_id.get(aid, {}).get("iata") or str(aid)
        st.markdown(f"**Alternativas semelhantes a {label}:**")
        if not sims:
            st.caption("Sem resultados.")
            continue
        df = pd.DataFrame(
            [
                {
                    "IATA": s.get("iata"),
                    "Aeroporto": s.get("name"),
                    "Cidade": f"{s.get('city', '')}, {s.get('country', '')}",
                    "Similaridade": round(s.get("similarity_score", 0), 3),
                }
                for s in sims
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Buscar aeroportos por descrição livre"):
        q = st.text_input("Ex.: aeroporto internacional na costa com muitas conexões", key="vec_query")
        if q:
            res = find_similar_airports_by_query(q, top_k=5)
            if res.get("success"):
                st.dataframe(pd.DataFrame(res["results"]), use_container_width=True, hide_index=True)
            else:
                st.warning(res.get("message", "Busca indisponível."))


def render_chat(mission: dict, form: dict) -> None:
    st.subheader("Pergunte sobre a missão")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)

    question = st.chat_input("Ex.: qual a melhor janela de chegada no destino?")
    if question:
        st.session_state.chat.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        ctx = {
            "origin": mission.get("origin"),
            "destination": mission.get("destination"),
            "aircraft_type": form["aircraft"],
            "departure_time": form["departure"],
            "recommended_route": mission.get("recommended_route") or {},
            "alternative_routes": mission.get("alternative_routes", []),
            "airports_data": mission.get("airports_data", {}),
        }
        with st.chat_message("assistant"):
            with st.spinner("Consultando Bedrock..."):
                ans = ask_mission(ctx, question)
            text = ans["text"] if ans.get("success") else f"IA indisponível: {ans.get('error')}"
            st.markdown(text)
        st.session_state.chat.append(("assistant", text))


def render_results(mission: dict, form: dict) -> None:
    if not mission.get("success"):
        st.error(mission.get("error") or "Falha ao planejar a missão.")
        return

    for w in mission.get("warnings", []):
        st.warning(w)

    route = mission["recommended_route"]
    o, d = mission["origin"], mission["destination"]
    st.success(
        f"Missão {o.get('iata')} → {d.get('iata')} planejada · "
        f"{form['payload']} · {form['aircraft']} · partida {form['departure']}"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Distância total", f"{route.get('total_distance_km', 0):,.0f} km")
    k2.metric("Distância (NM)", f"{route.get('total_distance_nm', 0):,.0f}")
    k3.metric("Trechos", route.get("num_legs", 1))
    k4.metric("Score da rota", route.get("route_score", 0))

    st.subheader("Rota recomendada")
    render_route_map([route] + mission.get("alternative_routes", []))
    render_route_table(route)
    if route.get("note"):
        st.caption(route["note"])

    st.divider()
    render_airport_cards(mission)
    st.divider()
    render_ai_section(mission, form)
    st.divider()
    render_alternatives(mission)
    st.divider()
    render_similar(mission)
    st.divider()
    render_chat(mission, form)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    render_sidebar()
    st.info(DISCLAIMER)

    form = render_form()
    if form:
        with st.spinner("Cruzando dados históricos no TiDB e calculando rotas..."):
            mission = plan_mission(
                origin_iata=form["origin"],
                dest_iata=form["dest"],
                aircraft_type=form["aircraft"],
                departure_time=form["departure"],
                include_ai=form["include_ai"],
            )
        st.session_state.mission = mission
        st.session_state.form = form
        st.session_state.chat = []

    if "mission" in st.session_state:
        render_results(st.session_state.mission, st.session_state.form)
    else:
        st.caption("Preencha o formulário e clique em **Planejar missão**.")


if __name__ == "__main__":
    main()
