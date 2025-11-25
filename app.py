import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path

# =========================
# Configuración básica
# =========================
st.set_page_config(
    page_title="Dashboard Telemetría",
    layout="wide"
)

DATA_DIR = Path("data")
FILE_CURRENT = DATA_DIR / "master_status_inner_qs_ready.csv"
FILE_HIST = DATA_DIR / "historico_conectividad.xlsx"

# =========================
# Helpers
# =========================

ORDER5 = ["Conectado 0-2", "Intermitente 3-14", "Limitado 15-30+", "Desconectado 31+", "Nunca"]

def no_accents_upper(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().upper()

def clasificar_5rangos(ts, dias):
    """
    ts   : Series datetime (NaT -> Nunca)
    dias : Series numérica (días desde última conexión)
    """
    out = pd.Series(index=dias.index, dtype="object")
    na = ts.isna()
    out[na] = "Nunca"
    m = ~na
    out[m & (dias <= 2)] = "Conectado 0-2"
    out[m & dias.between(3, 14, inclusive="both")] = "Intermitente 3-14"
    out[m & dias.between(15, 30, inclusive="both")] = "Limitado 15-30+"
    out[m & (dias >= 31)] = "Desconectado 31+"
    return pd.Categorical(out, categories=ORDER5, ordered=True)

def safe_pct(num, den):
    num = num.astype(float)
    den = den.astype(float).replace(0, np.nan)
    return (num / den * 100).round(2).fillna(0.0)

@st.cache_data
def load_current():
    if not FILE_CURRENT.exists():
        st.error(f"No encuentro el archivo {FILE_CURRENT}")
        return None

    df = pd.read_csv(FILE_CURRENT)

    # Normaliza columnas clave
    if "VIN" not in df.columns:
        st.error("El CSV no tiene columna 'VIN'.")
        return None

    # regla_norm
    if "regla_norm" not in df.columns and "REGLA GENERAL DE REPORTABILIDAD" in df.columns:
        df["regla_norm"] = df["REGLA GENERAL DE REPORTABILIDAD"].map(no_accents_upper)
    elif "regla_norm" in df.columns:
        df["regla_norm"] = df["regla_norm"].map(no_accents_upper)
    else:
        df["regla_norm"] = ""

    # Timestamps
    for c in ["gps_timestamp", "can_timestamp"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Días: acepta days_can/dias_sin_can, days_gps/dias_sin_gps
    if "days_can" in df.columns:
        df["days_can"] = pd.to_numeric(df["days_can"], errors="coerce")
    elif "dias_sin_can" in df.columns:
        df["days_can"] = pd.to_numeric(df["dias_sin_can"], errors="coerce")
    else:
        if "can_timestamp" in df.columns:
            today = pd.Timestamp.utcnow().normalize()
            can_date = pd.to_datetime(df["can_timestamp"].dt.date, errors="coerce")
            df["days_can"] = (today - can_date).dt.days
        else:
            df["days_can"] = np.nan

    if "days_gps" in df.columns:
        df["days_gps"] = pd.to_numeric(df["days_gps"], errors="coerce")
    elif "dias_sin_gps" in df.columns:
        df["days_gps"] = pd.to_numeric(df["dias_sin_gps"], errors="coerce")
    else:
        if "gps_timestamp" in df.columns:
            today = pd.Timestamp.utcnow().normalize()
            gps_date = pd.to_datetime(df["gps_timestamp"].dt.date, errors="coerce")
            df["days_gps"] = (today - gps_date).dt.days
        else:
            df["days_gps"] = np.nan

    # Empresa
    if "Empresa" not in df.columns:
        df["Empresa"] = "SIN_EMPRESA"

    # Estados 5 rangos
    # Telemetría: sólo regla_norm == 'TELEMETRIA'
    mask_tlm = df["regla_norm"] == "TELEMETRIA"
    df["estado_telemetria"] = pd.Categorical(["No aplica"] * len(df),
                                             categories=ORDER5 + ["No aplica"],
                                             ordered=True)
    if "can_timestamp" in df.columns:
        can_cat = clasificar_5rangos(df["can_timestamp"], df["days_can"])
        df.loc[mask_tlm, "estado_telemetria"] = can_cat[mask_tlm].astype(object)

    # GPS según REGLA (regla_norm != TELEMETRIA)
    mask_gps_regla = ~mask_tlm
    df["gps_status_regla"] = pd.Categorical(["No aplica"] * len(df),
                                            categories=ORDER5 + ["No aplica"],
                                            ordered=True)
    if "gps_timestamp" in df.columns:
        gps_cat = clasificar_5rangos(df["gps_timestamp"], df["days_gps"])
        df.loc[mask_gps_regla, "gps_status_regla"] = gps_cat[mask_gps_regla].astype(object)

    # GPS global (any)
    df["gps_status_any"] = clasificar_5rangos(df["gps_timestamp"], df["days_gps"]) \
        if "gps_timestamp" in df.columns else pd.Categorical(["Nunca"] * len(df),
                                                             categories=ORDER5,
                                                             ordered=True)

    # sacamos duplicados por VIN (nos quedamos con 1 fila por VIN)
    df = df.sort_values("VIN").drop_duplicates(subset=["VIN"], keep="first")

    return df


@st.cache_data
def load_historico():
    if not FILE_HIST.exists():
        return None

    try:
        hist = pd.read_excel(FILE_HIST)
    except Exception:
        hist = pd.read_excel(FILE_HIST, sheet_name=0)

    if "snapshot_date" in hist.columns:
        hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
    return hist


# =========================
#  UI
# =========================

st.title("📊 Dashboard Telemetría & GPS – CoPiloto")

df_current = load_current()
hist = load_historico()

if df_current is None:
    st.stop()

st.sidebar.header("Opciones")
show_raw = st.sidebar.checkbox("Mostrar tabla raw (foto actual)", False)

# =========================
# 1) KPIs actuales
# =========================
st.subheader("1. KPIs de conectividad (foto actual)")

# --- Telemetría: regla = TELEMETRIA ---
df_tlm = df_current[df_current["regla_norm"] == "TELEMETRIA"].copy()
tele_total = len(df_tlm)

tele_0_30 = len(df_tlm[df_tlm["days_can"] <= 30])
tele_31p  = len(df_tlm[df_tlm["days_can"] >= 31])
tele_nunca = len(df_tlm[df_tlm["can_timestamp"].isna()])

pct_tele_0_30  = safe_pct(pd.Series([tele_0_30]), pd.Series([tele_total]))[0] if tele_total else 0
pct_tele_31p   = safe_pct(pd.Series([tele_31p]),   pd.Series([tele_total]))[0] if tele_total else 0
pct_tele_nunca = safe_pct(pd.Series([tele_nunca]), pd.Series([tele_total]))[0] if tele_total else 0

# --- GPS por REGLA: regla != TELEMETRIA ---
df_gps_regla = df_current[df_current["regla_norm"] != "TELEMETRIA"].copy()
gps_regla_total = len(df_gps_regla)

gps_regla_0_15 = len(df_gps_regla[df_gps_regla["days_gps"] <= 15])
gps_regla_16p  = len(df_gps_regla[df_gps_regla["days_gps"] >= 16])
gps_regla_nunca = len(df_gps_regla[df_gps_regla["gps_timestamp"].isna()])

pct_gpsr_0_15  = safe_pct(pd.Series([gps_regla_0_15]), pd.Series([gps_regla_total]))[0] if gps_regla_total else 0
pct_gpsr_16p   = safe_pct(pd.Series([gps_regla_16p]),  pd.Series([gps_regla_total]))[0] if gps_regla_total else 0
pct_gpsr_nunca = safe_pct(pd.Series([gps_regla_nunca]),pd.Series([gps_regla_total]))[0] if gps_regla_total else 0

# --- GPS GLOBAL (todas las unidades con timestamp GPS) ---
gps_all = df_current.copy()
gps_all_total = len(gps_all)
gps_all_0_15  = len(gps_all[gps_all["days_gps"] <= 15])
gps_all_16p   = len(gps_all[gps_all["days_gps"] >= 16])
gps_all_nunca = len(gps_all[gps_all["gps_timestamp"].isna()])

pct_gps_all_0_15  = safe_pct(pd.Series([gps_all_0_15]), pd.Series([gps_all_total]))[0] if gps_all_total else 0
pct_gps_all_16p   = safe_pct(pd.Series([gps_all_16p]),  pd.Series([gps_all_total]))[0] if gps_all_total else 0
pct_gps_all_nunca = safe_pct(pd.Series([gps_all_nunca]),pd.Series([gps_all_total]))[0] if gps_all_total else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Telemetría – % conectados (0–30 días)**")
    st.metric("Telemetría 0–30 días", f"{pct_tele_0_30:.1f} %", help=f"Base: {tele_total} VIN con regla Telemetría")
with col2:
    st.markdown("**GPS (regla) – % conectados (0–15 días)**")
    st.metric("GPS regla 0–15 días", f"{pct_gpsr_0_15:.1f} %", help=f"Base: {gps_regla_total} VIN (regla ≠ Telemetría)")
with col3:
    st.markdown("**GPS global – % conectados (0–15 días)**")
    st.metric("GPS global 0–15 días", f"{pct_gps_all_0_15:.1f} %", help=f"Base: {gps_all_total} VIN totales")


# =========================
# 2) Barras por rangos 5 segmentos
# =========================
st.subheader("2. Distribución por rangos de días sin conexión (5 segmentos)")

colA, colB = st.columns(2)

# Telemetría
with colA:
    st.markdown("**Telemetría (regla = Telemetría)**")
    if not df_tlm.empty:
        counts_tlm = (
            df_tlm["estado_telemetria"]
            .value_counts()
            .reindex(ORDER5)
            .fillna(0)
            .astype(int)
            .reset_index()
            .rename(columns={"index":"estado","estado_telemetria":"VIN"})
        )
        counts_tlm["%"] = safe_pct(counts_tlm["VIN"], counts_tlm["VIN"].sum())
        st.bar_chart(counts_tlm.set_index("estado")["VIN"])
        st.dataframe(counts_tlm, use_container_width=True)
    else:
        st.info("No hay unidades con regla Telemetría.")

# GPS (según REGLA)
with colB:
    st.markdown("**GPS (según REGLA ≠ Telemetría)**")
    if not df_gps_regla.empty:
        counts_gpsr = (
            df_gps_regla["gps_status_regla"]
            .value_counts()
            .reindex(ORDER5)
            .fillna(0)
            .astype(int)
            .reset_index()
            .rename(columns={"index":"estado","gps_status_regla":"VIN"})
        )
        counts_gpsr["%"] = safe_pct(counts_gpsr["VIN"], counts_gpsr["VIN"].sum())
        st.bar_chart(counts_gpsr.set_index("estado")["VIN"])
        st.dataframe(counts_gpsr, use_container_width=True)
    else:
        st.info("No hay unidades con GPS según REGLA.")

# =========================
# 3) Top 10 empresas con más problemas
# =========================
st.subheader("3. Top 10 empresas con más unidades con problemas")

problem_labels = {"Intermitente 3-14", "Limitado 15-30+", "Desconectado 31+", "Nunca"}

# Telemetría
tele_problem = df_tlm[df_tlm["estado_telemetria"].isin(problem_labels)]
top10_tele = (
    tele_problem.groupby("Empresa")["VIN"].nunique()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
    .rename(columns={"VIN":"VIN_con_problema"})
)

# GPS regla
gpsr_problem = df_gps_regla[df_gps_regla["gps_status_regla"].isin(problem_labels)]
top10_gpsr = (
    gpsr_problem.groupby("Empresa")["VIN"].nunique()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
    .rename(columns={"VIN":"VIN_con_problema"})
)

colT1, colT2 = st.columns(2)
with colT1:
    st.markdown("**Top 10 empresas con problemas en Telemetría**")
    st.dataframe(top10_tele, use_container_width=True)
with colT2:
    st.markdown("**Top 10 empresas con problemas en GPS (según REGLA)**")
    st.dataframe(top10_gpsr, use_container_width=True)


# =========================
# 4) Histórico de conectividad
# =========================
st.subheader("4. Evolución histórica de la conectividad")

if hist is None or hist.empty:
    st.info("No se encontró `data/historico_conectividad.xlsx` o está vacío.")
else:
    hist = hist.sort_values("snapshot_date")
    st.markdown("**Telemetría 0–30 días vs desconectado 31+**")
    cols_disp = [c for c in hist.columns if c.startswith("tele_")]
    st.line_chart(
        hist.set_index("snapshot_date")[["tele_0_30_pct","tele_desconectado_31p_pct"]]
    )

    st.markdown("**GPS (regla) 0–15 días vs desconectado**")
    if {"gps_regla_0_15_pct","gps_regla_desconectado_16p_pct"}.issubset(hist.columns):
        st.line_chart(
            hist.set_index("snapshot_date")[["gps_regla_0_15_pct","gps_regla_desconectado_16p_pct"]]
        )

    st.markdown("**GPS global 0–15 días vs desconectado**")
    if {"gps_global_0_15_pct","gps_global_desconectado_16p_pct"}.issubset(hist.columns):
        st.line_chart(
            hist.set_index("snapshot_date")[["gps_global_0_15_pct","gps_global_desconectado_16p_pct"]]
        )

    st.markdown("**Tabla histórico de KPIs**")
    st.dataframe(hist, use_container_width=True)

# =========================
# 5) Tabla raw opcional
# =========================
if show_raw:
    st.subheader("5. Tabla raw – foto actual (1 fila por VIN)")
    st.dataframe(df_current, use_container_width=True)
