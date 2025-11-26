import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path
from datetime import datetime

# =========================
# CONFIGURACIÓN BÁSICA
# =========================
st.set_page_config(
    page_title="Dashboard Telemetría & GPS",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
PATH_STATUS = DATA_DIR / "master_status_inner_qs_ready.csv"
PATH_HIST   = DATA_DIR / "historico_conectividad.xlsx"

ORDER5 = ["Conectado 0-2", "Intermitente 3-14", "Limitado 15-30+",
          "Desconectado 31+", "Nunca"]

# =========================
# HELPERS
# =========================
def no_accents_upper(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().upper()

def clasificar_5rangos(ts, dias):
    """
    5 rangos usando:
      - ts: timestamp (NaT => 'Nunca')
      - dias: días desde última conexión (Int)
    """
    out = pd.Series(index=dias.index, dtype="object")
    is_na_ts = ts.isna()
    out[is_na_ts] = "Nunca"
    m = ~is_na_ts
    out[m & (dias <= 2)]                             = "Conectado 0-2"
    out[m & dias.between(3, 14, inclusive="both")]  = "Intermitente 3-14"
    out[m & dias.between(15, 30, inclusive="both")] = "Limitado 15-30+"
    out[m & (dias >= 31)]                           = "Desconectado 31+"
    return pd.Categorical(out, categories=ORDER5, ordered=True)

def safe_pct(num, den):
    num = float(num)
    den = float(den) if den else 0.0
    return round(num / den * 100, 2) if den > 0 else 0.0

# =========================
# CARGA DE DATOS
# =========================

@st.cache_data
def load_status_df():
    # Ajusta la ruta si es distinta
    path = "data/master_status_inner_qs_ready.csv"
    df = pd.read_csv(path)

    # --- Normalizar timestamps ---
    for c in ["gps_timestamp", "can_timestamp", "last_update_utc"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)  # siempre UTC

    # ---- Definir "hoy" también en UTC y luego quitar tz para que todo sea naive ----
    today_utc = pd.Timestamp.utcnow().normalize().tz_localize("UTC")
    # hacemos las fechas de los timestamps también UTC pero luego las pasamos a naive

    # GPS
    if "gps_timestamp" in df.columns:
        gps_local = df["gps_timestamp"].dt.tz_convert("UTC")  # ya es UTC, pero por si acaso
        gps_date = gps_local.dt.tz_localize(None).dt.normalize()  # quitar tz, dejar YYYY-MM-DD 00:00
    else:
        gps_date = pd.NaT

    # CAN
    if "can_timestamp" in df.columns:
        can_local = df["can_timestamp"].dt.tz_convert("UTC")
        can_date = can_local.dt.tz_localize(None).dt.normalize()
    else:
        can_date = pd.NaT

    # Y dejamos today también naive para que no choque
    today_naive = today_utc.tz_localize(None)  # quitar tz

    # --- Calcular días SIN mezclar aware/naive ---
    if "gps_timestamp" in df.columns:
        df["days_gps"] = (today_naive - gps_date).dt.days
    else:
        df["days_gps"] = np.nan

    if "can_timestamp" in df.columns:
        df["days_can"] = (today_naive - can_date).dt.days
    else:
        df["days_can"] = np.nan

    # Asegurar tipo Int64 con NA
    df["days_gps"] = pd.to_numeric(df["days_gps"], errors="coerce").astype("Int64")
    df["days_can"] = pd.to_numeric(df["days_can"], errors="coerce").astype("Int64")

    # --- Normalizar regla de reportabilidad (si la usas en filtros/KPIs) ---
    def no_accents_upper(s):
        if pd.isna(s): 
            return ""
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return s.strip().upper()

    if "REGLA GENERAL DE REPORTABILIDAD" in df.columns:
        df["regla_norm"] = df["REGLA GENERAL DE REPORTABILIDAD"].map(no_accents_upper)
    else:
        df["regla_norm"] = ""

    return df

@st.cache_data
def load_historico_df():
    try:
        hist = pd.read_excel(PATH_HIST)
    except FileNotFoundError:
        return None
    if "snapshot_date" in hist.columns:
        hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
    return hist

df_status = load_status_df()
hist_df   = load_historico_df()

# =========================
# SIDEBAR FILTROS
# =========================
st.sidebar.title("Filtros")

# Filtro por Empresa
empresas = sorted(df_status.get("Empresa", pd.Series(["SIN_EMPRESA"])).fillna("SIN_EMPRESA").unique())
empresa_sel = st.sidebar.multiselect(
    "Empresa",
    options=empresas,
    default=empresas  # todas por defecto
)

# Filtro por modelo de dispositivo (opcional)
if "device_model" in df_status.columns:
    modelos = sorted(df_status["device_model"].fillna("SIN_MODELO").unique())
    modelo_sel = st.sidebar.multiselect(
        "Modelo dispositivo",
        options=modelos,
        default=modelos
    )
else:
    modelo_sel = None

# Aplicar filtros
mask_emp = df_status.get("Empresa", "SIN_EMPRESA").fillna("SIN_EMPRESA").isin(empresa_sel)
if modelo_sel is not None:
    mask_mod = df_status["device_model"].fillna("SIN_MODELO").isin(modelo_sel)
else:
    mask_mod = True

df_f = df_status[mask_emp & mask_mod].copy()

# =========================
# CÁLCULO DE KPIs (subset filtrado)
# =========================

# --- Telemetría: regla = TELEMETRIA ---
tele = df_f[df_f["regla_norm"] == "TELEMETRIA"].copy()
tele_total = len(tele)
tele_0_30  = len(tele[tele["days_can"] <= 30])
tele_31p   = len(tele[tele["days_can"] >= 31])
tele_nunca = len(tele[tele["can_timestamp"].isna()])

tele_0_30_pct  = safe_pct(tele_0_30, tele_total)
tele_31p_pct   = safe_pct(tele_31p,  tele_total)
tele_nunca_pct = safe_pct(tele_nunca, tele_total)

# --- GPS (según REGLA ≠ Telemetría) ---
gps_regla = df_f[df_f["regla_norm"] != "TELEMETRIA"].copy()
gpsr_total = len(gps_regla)
gpsr_0_15  = len(gps_regla[gps_regla["days_gps"] <= 15])
gpsr_16p   = len(gps_regla[gps_regla["days_gps"] >= 16])
gpsr_nunca = len(gps_regla[gps_regla["gps_timestamp"].isna()])

gpsr_0_15_pct  = safe_pct(gpsr_0_15, gpsr_total)
gpsr_16p_pct   = safe_pct(gpsr_16p,  gpsr_total)
gpsr_nunca_pct = safe_pct(gpsr_nunca, gpsr_total)

# --- GPS GLOBAL (todas las filas filtradas) ---
gps_all = df_f.copy()
gpsa_total = len(gps_all)
gpsa_0_15  = len(gps_all[gps_all["days_gps"] <= 15])
gpsa_16p   = len(gps_all[gps_all["days_gps"] >= 16])
gpsa_nunca = len(gps_all[gps_all["gps_timestamp"].isna()])

gpsa_0_15_pct  = safe_pct(gpsa_0_15, gpsa_total)
gpsa_16p_pct   = safe_pct(gpsa_16p,  gpsa_total)
gpsa_nunca_pct = safe_pct(gpsa_nunca, gpsa_total)

# =========================
# LAYOUT PRINCIPAL
# =========================
st.title("Dashboard de Conectividad Telemetría & GPS")

st.markdown(
    f"**Unidades en muestra (filtradas):** {len(df_f):,}  "
    f"| **Telemetría (regla=TELEMETRIA):** {tele_total:,}  "
    f"| **GPS (regla≠Telemetría):** {gpsr_total:,}"
)

# ---------- KPIs ----------
st.subheader("KPIs Principales")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Telemetría 0–30 días", f"{tele_0_30_pct:.2f} %", help="% de unidades con REGLA=Telemetría con CAN en 0–30 días")
    st.metric("Telemetría Desconectado 31+ días", f"{tele_31p_pct:.2f} %")
    st.metric("Telemetría Nunca conectado", f"{tele_nunca_pct:.2f} %")

with col2:
    st.metric("GPS (regla) 0–15 días", f"{gpsr_0_15_pct:.2f} %", help="Solo unidades con REGLA≠Telemetría")
    st.metric("GPS (regla) 16+ días", f"{gpsr_16p_pct:.2f} %")
    st.metric("GPS (regla) Nunca", f"{gpsr_nunca_pct:.2f} %")

with col3:
    st.metric("GPS Global 0–15 días", f"{gpsa_0_15_pct:.2f} %")
    st.metric("GPS Global 16+ días", f"{gpsa_16p_pct:.2f} %")
    st.metric("GPS Global Nunca", f"{gpsa_nunca_pct:.2f} %")

st.markdown("---")

# ---------- Distribución por rangos ----------
st.subheader("Distribución por rangos de días (5 categorías)")

# Telemetría
tele_estado_counts = (
    tele["estado_telemetria"]
    .value_counts()
    .reindex(ORDER5)
    .fillna(0)
    .astype(int)
    .to_frame("VIN_unicos")
)
tele_estado_counts["%"] = (tele_estado_counts["VIN_unicos"] /
                           tele_estado_counts["VIN_unicos"].sum() * 100).round(2)

# GPS según REGLA
gpsr_estado_counts = (
    gps_regla["gps_status_regla"]
    .replace("No aplica", np.nan)
    .dropna()
    .value_counts()
    .reindex(ORDER5)
    .fillna(0)
    .astype(int)
    .to_frame("VIN_unicos")
)
if gpsr_estado_counts["VIN_unicos"].sum() > 0:
    gpsr_estado_counts["%"] = (gpsr_estado_counts["VIN_unicos"] /
                               gpsr_estado_counts["VIN_unicos"].sum() * 100).round(2)
else:
    gpsr_estado_counts["%"] = 0.0

col_tlm, col_gps = st.columns(2)

with col_tlm:
    st.markdown("**Telemetría – Estado (5 rangos)**")
    st.bar_chart(tele_estado_counts["VIN_unicos"])
    st.dataframe(tele_estado_counts)

with col_gps:
    st.markdown("**GPS (según REGLA≠Telemetría) – Estado (5 rangos)**")
    st.bar_chart(gpsr_estado_counts["VIN_unicos"])
    st.dataframe(gpsr_estado_counts)

st.markdown("---")

# ---------- Top 10 empresas con problemas ----------
st.subheader("Top 10 Empresas con más problemas de conectividad")

problem_labels = ["Intermitente 3-14", "Limitado 15-30+", "Desconectado 31+", "Nunca"]

# Telemetría problemas
tele_problem = tele[tele["estado_telemetria"].isin(problem_labels)]
top10_tele = (
    tele_problem.groupby("Empresa")["VIN"].nunique()
    .sort_values(ascending=False)
    .head(10)
    .to_frame("VIN_con_problema")
)
if len(top10_tele):
    st.markdown("**Top 10 – Telemetría**")
    st.bar_chart(top10_tele["VIN_con_problema"])
    st.dataframe(top10_tele)
else:
    st.info("No hay datos de telemetría con problemas bajo los filtros actuales.")

# GPS problemas
gpsr_problem = gps_regla[gps_regla["gps_status_regla"].isin(problem_labels)]
top10_gps = (
    gpsr_problem.groupby("Empresa")["VIN"].nunique()
    .sort_values(ascending=False)
    .head(10)
    .to_frame("VIN_con_problema")
)

if len(top10_gps):
    st.markdown("**Top 10 – GPS (según REGLA≠Telemetría)**")
    st.bar_chart(top10_gps["VIN_con_problema"])
    st.dataframe(top10_gps)
else:
    st.info("No hay datos de GPS con problemas bajo los filtros actuales.")

st.markdown("---")

# ---------- Histórico (si existe) ----------
st.subheader("Histórico de conectividad (snapshot diario)")

if hist_df is None or hist_df.empty:
    st.info("No se encontró `historico_conectividad.xlsx` en la carpeta data/ o está vacío.")
else:
    # Ordenar por fecha
    hist_df = hist_df.sort_values("snapshot_date")

    # Selección de métricas a mostrar
    metricas_hist = {
        "Telemetría 0–30 %": "tele_0_30_pct",
        "Telemetría desconectado 31+ %": "tele_desconectado_31p_pct",
        "Telemetría nunca %": "tele_nunca_pct",
        "GPS (regla) 0–15 %": "gps_regla_0_15_pct",
        "GPS (regla) desconectado 16+ %": "gps_regla_desconectado_16p_pct",
        "GPS (regla) nunca %": "gps_regla_nunca_pct",
        "GPS global 0–15 %": "gps_global_0_15_pct",
        "GPS global desconectado 16+ %": "gps_global_desconectado_16p_pct",
        "GPS global nunca %": "gps_global_nunca_pct",
    }

    metricas_sel = st.multiselect(
        "Métricas históricas a mostrar:",
        options=list(metricas_hist.keys()),
        default=["Telemetría 0–30 %", "GPS (regla) 0–15 %"]
    )

    cols_sel = [metricas_hist[k] for k in metricas_sel if metricas_hist[k] in hist_df.columns]

    if cols_sel:
        hist_plot = hist_df.set_index("snapshot_date")[cols_sel]
        st.line_chart(hist_plot)
        st.dataframe(hist_plot.tail(10))
    else:
        st.info("Selecciona al menos una métrica para graficar.")

st.markdown("---")
st.caption("Dashboard local – basado en master_status_inner_qs_ready.csv e historico_conectividad.xlsx")
