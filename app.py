# -*- coding: utf-8 -*-
import streamlit as st  # pyright: ignore[reportMissingImports]
import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path

# =========================
# CONFIGURACIÓN BÁSICA
# =========================
st.set_page_config(page_title="Dashboard Telemetría & GPS", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

PATH_STATUS = DATA_DIR / "master_status_inner_qs_ready.csv"
PATH_HIST   = DATA_DIR / "historico_conectividad.xlsx"

ORDER5 = ["Conectado 0-2", "Intermitente 3-14", "Limitado 15-30+", "Desconectado 31+", "Nunca"]
PROBLEM5 = ["Intermitente 3-14", "Limitado 15-30+", "Desconectado 31+", "Nunca"]

# =========================
# HELPERS
# =========================
def no_accents_upper(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().upper()

def safe_pct(num, den):
    den = float(den) if den else 0.0
    return round(float(num) / den * 100, 2) if den > 0 else 0.0

def clasificar_5rangos(ts: pd.Series, dias: pd.Series) -> pd.Categorical:
    out = pd.Series(index=dias.index, dtype="object")

    is_na_ts = ts.isna()
    out[is_na_ts] = "Nunca"

    m = ~is_na_ts
    out[m & (dias <= 2)]                              = "Conectado 0-2"
    out[m & dias.between(3, 14, inclusive="both")]    = "Intermitente 3-14"
    out[m & dias.between(15, 30, inclusive="both")]   = "Limitado 15-30+"
    out[m & (dias >= 31)]                             = "Desconectado 31+"

    return pd.Categorical(out, categories=ORDER5, ordered=True)

def estado_4rangos_tele(dias_can: pd.Series, can_ts: pd.Series) -> pd.Series:
    """
    Telemetría (CAN): 4 rangos con 'Nunca' dentro de 'Desconectado'
    - 0-2 Conectado
    - 3-14 Intermitente
    - 15-30 Limitado
    - >30 o null -> Desconectado
    """
    out = pd.Series(index=dias_can.index, dtype="object")
    out[:] = "Desconectado"
    m = can_ts.notna() & dias_can.notna()
    out[m & (dias_can <= 2)] = "Conectado"
    out[m & dias_can.between(3, 14, inclusive="both")] = "Intermitente"
    out[m & dias_can.between(15, 30, inclusive="both")] = "Limitado"
    return out

def estado_4rangos_gps(dias_gps: pd.Series, gps_ts: pd.Series) -> pd.Series:
    """
    GPS: 4 rangos con 'Nunca' dentro de 'Desconectado'
    - 0-2 Conectado
    - 3-7 Intermitente
    - 8-15 Limitado
    - >15 o null -> Desconectado
    """
    out = pd.Series(index=dias_gps.index, dtype="object")
    out[:] = "Desconectado"
    m = gps_ts.notna() & dias_gps.notna()
    out[m & (dias_gps <= 2)] = "Conectado"
    out[m & dias_gps.between(3, 7, inclusive="both")] = "Intermitente"
    out[m & dias_gps.between(8, 15, inclusive="both")] = "Limitado"
    return out

# =========================
# CARGA DE DATOS
# =========================
@st.cache_data
def load_status_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Parse timestamps (vienen UTC) → naive
    for c in ["gps_timestamp", "can_timestamp", "last_update_utc", "t_ref"]:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = ts.dt.tz_localize(None)

    today = pd.Timestamp.now().normalize()

    # Días desde último dato
    if "gps_timestamp" in df.columns:
        df["days_gps"] = (today - df["gps_timestamp"].dt.normalize()).dt.days
    else:
        df["days_gps"] = np.nan

    if "can_timestamp" in df.columns:
        df["days_can"] = (today - df["can_timestamp"].dt.normalize()).dt.days
    else:
        df["days_can"] = np.nan

    df["days_gps"] = pd.to_numeric(df["days_gps"], errors="coerce").astype("Int64")
    df["days_can"] = pd.to_numeric(df["days_can"], errors="coerce").astype("Int64")

    # Regla
    regla_col = "REGLA GENERAL DE REPORTABILIDAD"
    if regla_col in df.columns:
        df["regla_norm"] = df[regla_col].map(no_accents_upper)
    else:
        df["regla_norm"] = ""

    # Estados 5 rangos
    if "can_timestamp" in df.columns:
        df["estado_telemetria_5"] = clasificar_5rangos(df["can_timestamp"], df["days_can"])
    else:
        df["estado_telemetria_5"] = pd.Categorical(["Nunca"] * len(df), categories=ORDER5, ordered=True)

    if "gps_timestamp" in df.columns:
        df["estado_gps_5"] = clasificar_5rangos(df["gps_timestamp"], df["days_gps"])
    else:
        df["estado_gps_5"] = pd.Categorical(["Nunca"] * len(df), categories=ORDER5, ordered=True)

    # Estados 4 rangos (ejecutivo)
    df["estado_telemetria_4"] = estado_4rangos_tele(df["days_can"], df.get("can_timestamp", pd.Series(pd.NaT, index=df.index)))
    df["estado_gps_4"] = estado_4rangos_gps(df["days_gps"], df.get("gps_timestamp", pd.Series(pd.NaT, index=df.index)))

    return df

@st.cache_data
def load_historico_df(path: Path):
    if not path.exists():
        return None, None

    try:
        hist = pd.read_excel(path, sheet_name="historico")
    except Exception:
        return None, None

    if hist.empty:
        return None, None

    if "fecha" in hist.columns:
        hist["fecha"] = pd.to_datetime(hist["fecha"], errors="coerce")

    # pct 0–30 desde columnas si existen
    for c in ["pct_Conectado 0-2", "pct_Intermitente 3-14", "pct_Limitado 15-30+"]:
        if c not in hist.columns:
            hist[c] = 0.0

    hist["pct_0_30"] = (
        hist["pct_Conectado 0-2"].fillna(0)
        + hist["pct_Intermitente 3-14"].fillna(0)
        + hist["pct_Limitado 15-30+"].fillna(0)
    )

    ultima_fecha = hist["fecha"].max() if "fecha" in hist.columns else None
    return hist, ultima_fecha

# =========================
# APP
# =========================
df_status = load_status_df(PATH_STATUS)
hist_df, ultima_fecha = load_historico_df(PATH_HIST)

st.title("Dashboard de Conectividad Telemetría & GPS")

if ultima_fecha is not None and pd.notna(ultima_fecha):
    st.markdown(f"**Fecha de la Data (Última Actualización):** **`{ultima_fecha.strftime('%d-%m-%Y')}`**")
else:
    st.info("No se pudo determinar la fecha de la última actualización del histórico.")

# =========================
# SIDEBAR FILTROS
# =========================
st.sidebar.title("Filtros")

empresa_col = "Empresa" if "Empresa" in df_status.columns else None
device_col = "device_model" if "device_model" in df_status.columns else None

if empresa_col:
    empresas = sorted(df_status[empresa_col].fillna("SIN_EMPRESA").unique())
    empresa_sel = st.sidebar.multiselect("Empresa", options=empresas, default=empresas)
else:
    empresa_sel = None

if device_col:
    modelos = sorted(df_status[device_col].fillna("SIN_MODELO").unique())
    modelo_sel = st.sidebar.multiselect("Modelo dispositivo", options=modelos, default=modelos)
else:
    modelo_sel = None

mask = pd.Series(True, index=df_status.index)
if empresa_sel is not None:
    mask &= df_status[empresa_col].fillna("SIN_EMPRESA").isin(empresa_sel)
if modelo_sel is not None:
    mask &= df_status[device_col].fillna("SIN_MODELO").isin(modelo_sel)

df_f = df_status.loc[mask].copy()

st.markdown(
    f"**Unidades en muestra (filtradas):** {len(df_f):,}"
)

# =========================
# SUBSETS (Telemetría vs GPS regla)
# =========================
tele = df_f[df_f["regla_norm"] == "TELEMETRIA"].copy()
gps_regla = df_f[df_f["regla_norm"] != "TELEMETRIA"].copy()

tele_total = len(tele)
gps_total = len(gps_regla)

# KPIs Telemetría (30 días)
tele_0_30 = len(tele[tele["days_can"].notna() & (tele["days_can"] <= 30)])
tele_31p  = len(tele[tele["days_can"].notna() & (tele["days_can"] >= 31)])
tele_nunca = len(tele[tele.get("can_timestamp").isna()]) if "can_timestamp" in tele.columns else tele_total

tele_pct_0_30 = safe_pct(tele_0_30, tele_total)

# KPIs GPS (15 días)
gps_0_15 = len(gps_regla[gps_regla["days_gps"].notna() & (gps_regla["days_gps"] <= 15)])
gps_16p  = len(gps_regla[gps_regla["days_gps"].notna() & (gps_regla["days_gps"] >= 16)])
gps_nunca = len(gps_regla[gps_regla.get("gps_timestamp").isna()]) if "gps_timestamp" in gps_regla.columns else gps_total

gps_pct_0_15 = safe_pct(gps_0_15, gps_total)

# =========================
# KPIs PRINCIPALES
# =========================
st.subheader("KPIs Principales")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Telemetría (0–30 días)", f"{tele_pct_0_30:.2f}%")
    st.caption(f"{tele_0_30:,} / {tele_total:,}")

with c2:
    st.metric("GPS Copiloto (0–15 días)", f"{gps_pct_0_15:.2f}%")
    st.caption(f"{gps_0_15:,} / {gps_total:,}")

with c3:
    st.metric("Telemetría Desconectado 31+", f"{safe_pct(tele_31p, tele_total):.2f}%")
    st.caption(f"{tele_31p:,}")

with c4:
    st.metric("GPS Desconectado 16+", f"{safe_pct(gps_16p, gps_total):.2f}%")
    st.caption(f"{gps_16p:,}")

st.markdown("---")

# =========================
# DISTRIBUCIÓN 5 RANGOS
# =========================
st.subheader("Distribución (5 rangos)")

colA, colB = st.columns(2)

with colA:
    st.markdown("**Telemetría – Estado (5 rangos)**")
    if tele_total:
        tele_counts = tele["estado_telemetria_5"].value_counts().reindex(ORDER5).fillna(0).astype(int)
        st.bar_chart(tele_counts)
        st.dataframe(tele_counts.to_frame("count"))
    else:
        st.info("Sin datos de Telemetría con los filtros actuales.")

with colB:
    st.markdown("**GPS (regla ≠ Telemetría) – Estado (5 rangos)**")
    if gps_total:
        gps_counts = gps_regla["estado_gps_5"].value_counts().reindex(ORDER5).fillna(0).astype(int)
        st.bar_chart(gps_counts)
        st.dataframe(gps_counts.to_frame("count"))
    else:
        st.info("Sin datos de GPS con los filtros actuales.")

st.markdown("---")

# =========================
# TOP 10 PROBLEMAS
# =========================
st.subheader("Top 10 Empresas con más problemas")

if empresa_col:
    colT, colG = st.columns(2)

    with colT:
        st.markdown("**Telemetría – Top 10 problemas**")
        if tele_total:
            tele_prob = tele[tele["estado_telemetria_5"].isin(PROBLEM5)]
            top10 = tele_prob.groupby(empresa_col)["VIN"].nunique().sort_values(ascending=False).head(10)
            st.bar_chart(top10)
            st.dataframe(top10.to_frame("vin_con_problema"))
        else:
            st.info("Sin datos de Telemetría.")

    with colG:
        st.markdown("**GPS (regla) – Top 10 problemas**")
        if gps_total:
            gps_prob = gps_regla[gps_regla["estado_gps_5"].isin(PROBLEM5)]
            top10g = gps_prob.groupby(empresa_col)["VIN"].nunique().sort_values(ascending=False).head(10)
            st.bar_chart(top10g)
            st.dataframe(top10g.to_frame("vin_con_problema"))
        else:
            st.info("Sin datos de GPS.")
else:
    st.info("No existe columna 'Empresa' en el dataset para Top 10.")

st.markdown("---")

# =========================
# HISTÓRICO
# =========================
st.subheader("Histórico de conectividad (pct 0–30)")

if hist_df is None or hist_df.empty:
    st.info("No se encontró historico_conectividad.xlsx (hoja 'historico') o está vacío.")
else:
    h = hist_df.dropna(subset=["fecha"]).sort_values("fecha")
    pivot = h.pivot(index="fecha", columns="resumen", values="pct_0_30").sort_index()

    # Renombrar para leyenda (opcional)
    pivot = pivot.rename(columns={
        "Telemetría": "Telemetría",
        "GPS (según REGLA)": "GPS_Copiloto",
        "GPS (todas con gps_timestamp)": "wicar_gps",
    })

    cols_exist = [c for c in ["Telemetría", "GPS_Copiloto", "wicar_gps"] if c in pivot.columns]
    pivot = pivot[cols_exist]

    if pivot.empty:
        st.info("Histórico sin columnas válidas para graficar.")
    else:
        st.line_chart(pivot)
        st.dataframe(pivot.tail(20))

st.caption("Dashboard local (Streamlit)")