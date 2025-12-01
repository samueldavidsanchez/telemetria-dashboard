import streamlit as st  # pyright: ignore[reportMissingImports]
import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path

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
    # Usamos la ruta relativa dentro del repo
    path = DATA_DIR / "master_status_inner_qs_ready.csv"
    df = pd.read_csv(path)

    # --- Normalizar timestamps: parsear como UTC y luego quitar tz (dejarlos naive) ---
    for c in ["gps_timestamp", "can_timestamp", "last_update_utc"]:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            # Quitamos la zona horaria para trabajar todo en naive
            df[c] = ts.dt.tz_localize(None)

    # "Hoy" como fecha naive (sin tz)
    today = pd.Timestamp.now().normalize()

    # --- Calcular días desde última conexión ---
    if "gps_timestamp" in df.columns:
        gps_date = df["gps_timestamp"].dt.normalize()
        df["days_gps"] = (today - gps_date).dt.days
    else:
        df["days_gps"] = np.nan

    if "can_timestamp" in df.columns:
        can_date = df["can_timestamp"].dt.normalize()
        df["days_can"] = (today - can_date).dt.days
    else:
        df["days_can"] = np.nan

    # Asegurar tipo Int64 con NA
    df["days_gps"] = pd.to_numeric(df["days_gps"], errors="coerce").astype("Int64")
    df["days_can"] = pd.to_numeric(df["days_can"], errors="coerce").astype("Int64")

    # --- Normalizar regla de reportabilidad ---
    if "REGLA GENERAL DE REPORTABILIDAD" in df.columns:
        df["regla_norm"] = df["REGLA GENERAL DE REPORTABILIDAD"].map(no_accents_upper)
    else:
        df["regla_norm"] = ""

    # Si no existen las columnas de estado, las calculamos aquí (por si no lo hacías antes)
    # Telemetría (solo regla = TELEMETRIA)
    mask_tlm = df["regla_norm"] == "TELEMETRIA"
    df["estado_telemetria"] = pd.Categorical(["Nunca"] * len(df), categories=ORDER5, ordered=True)
    if "can_timestamp" in df.columns:
        df.loc[mask_tlm, "estado_telemetria"] = clasificar_5rangos(
            df.loc[mask_tlm, "can_timestamp"],
            df.loc[mask_tlm, "days_can"]
        )

    # GPS según REGLA (regla != TELEMETRIA)
    mask_gps_regla = df["regla_norm"] != "TELEMETRIA"
    df["gps_status_regla"] = pd.Categorical(["No aplica"] * len(df),
                                            categories=ORDER5 + ["No aplica"],
                                            ordered=True)
    if "gps_timestamp" in df.columns:
        gps_cat = clasificar_5rangos(df["gps_timestamp"], df["days_gps"]).astype(object)
        tmp = pd.Series(["No aplica"] * len(df), index=df.index, dtype="object")
        tmp[mask_gps_regla] = gps_cat[mask_gps_regla]
        df["gps_status_regla"] = pd.Categorical(tmp, categories=ORDER5 + ["No aplica"], ordered=True)

    return df

    # Lee el CSV de estado actual
    df = pd.read_csv(PATH_STATUS)

    # --- Normalizar timestamps como NAIVE (sin tz) ---
    for c in ["gps_timestamp", "can_timestamp", "last_update_utc"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")  # sin utc=True

    # ---- Calcular días desde hoy (todo naive) ----
    today = pd.Timestamp.today().normalize()  # naive, sin tz

    if "gps_timestamp" in df.columns:
        gps_date = pd.to_datetime(df["gps_timestamp"].dt.date, errors="coerce")
        df["days_gps"] = (today - gps_date).dt.days
    else:
        df["days_gps"] = np.nan

    if "can_timestamp" in df.columns:
        can_date = pd.to_datetime(df["can_timestamp"].dt.date, errors="coerce")
        df["days_can"] = (today - can_date).dt.days
    else:
        df["days_can"] = np.nan

    # Asegurar tipo Int64 con NA
    df["days_gps"] = pd.to_numeric(df["days_gps"], errors="coerce").astype("Int64")
    df["days_can"] = pd.to_numeric(df["days_can"], errors="coerce").astype("Int64")

    # --- Normalizar regla de reportabilidad ---
    if "REGLA GENERAL DE REPORTABILIDAD" in df.columns:
        df["regla_norm"] = df["REGLA GENERAL DE REPORTABILIDAD"].map(no_accents_upper)
    else:
        df["regla_norm"] = ""

    # --- Estados 5 rangos para Telemetría y GPS ---
    # Telemetría sólo para regla = TELEMETRIA
    df["estado_telemetria"] = clasificar_5rangos(
        df["can_timestamp"] if "can_timestamp" in df.columns else pd.Series(pd.NaT, index=df.index),
        df["days_can"]
    )

    # GPS (ANY) para todas las filas con gps_timestamp
    df["gps_status_any"] = clasificar_5rangos(
        df["gps_timestamp"] if "gps_timestamp" in df.columns else pd.Series(pd.NaT, index=df.index),
        df["days_gps"]
    )

    # GPS (REGRA): sólo cuando regla_norm != TELEMETRIA, resto "No aplica"
    gps_regla_raw = clasificar_5rangos(
        df["gps_timestamp"] if "gps_timestamp" in df.columns else pd.Series(pd.NaT, index=df.index),
        df["days_gps"]
    ).astype(object)

    gps_regla = pd.Series("No aplica", index=df.index, dtype="object")
    mask_gps_regla = df["regla_norm"] != "TELEMETRIA"
    gps_regla[mask_gps_regla] = gps_regla_raw[mask_gps_regla]
    df["gps_status_regla"] = pd.Categorical(gps_regla, categories=ORDER5 + ["No aplica"], ordered=True)

    return df

@st.cache_data
def load_historico_df():
    """
    Lee historico_conectividad.xlsx, hoja 'historico',
    y calcula pct_0_30 = Conectado 0-2 + Intermitente 3-14 + Limitado 15-30+
    """
    try:
        hist = pd.read_excel(PATH_HIST, sheet_name="historico")
    except FileNotFoundError:
        return None

    # Asegurar fecha en datetime
    if "fecha" in hist.columns:
        hist["fecha"] = pd.to_datetime(hist["fecha"], errors="coerce")

    # Asegurar columnas necesarias (si alguna faltara, la ponemos en 0)
    for c in ["pct_Conectado 0-2", "pct_Intermitente 3-14", "pct_Limitado 15-30+"]:
        if c not in hist.columns:
            hist[c] = 0.0

    # % conectado 0–30 días
    hist["pct_0_30"] = (
        hist["pct_Conectado 0-2"].fillna(0)
        + hist["pct_Intermitente 3-14"].fillna(0)
        + hist["pct_Limitado 15-30+"].fillna(0)
    )

    return hist


    """
    Carga historico_conectividad.xlsx, hoja 'historico'.
    Si no existe esa hoja, usa la primera disponible.
    Normaliza la columna de fecha a 'snapshot_date'.
    """
    try:
        xls = pd.ExcelFile(PATH_HIST)
    except FileNotFoundError:
        return None

    sheet_name = "historico"
    if sheet_name not in xls.sheet_names:
        # Usa la primera hoja si no existe 'historico'
        sheet_name = xls.sheet_names[0]

    hist = xls.parse(sheet_name)

    # Intentar detectar la columna de fecha
    date_col = None
    for candidate in ["snapshot_date", "fecha", "date", "Fecha"]:
        if candidate in hist.columns:
            date_col = candidate
            break

    if date_col is None:
        # No hay columna clara de fecha, devolvemos tal cual
        return hist

    # Renombrar a snapshot_date si hace falta
    if date_col != "snapshot_date":
        hist = hist.rename(columns={date_col: "snapshot_date"})

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
if len(tele):
    tele_estado_counts = (
        tele["estado_telemetria"]
        .value_counts()
        .reindex(ORDER5)
        .fillna(0)
        .astype(int)
        .to_frame("VIN_unicos")
    )
    tele_estado_counts["%"] = (
        tele_estado_counts["VIN_unicos"] /
        tele_estado_counts["VIN_unicos"].sum() * 100
    ).round(2)
else:
    tele_estado_counts = pd.DataFrame(
        {"VIN_unicos": [0]*len(ORDER5), "%": [0]*len(ORDER5)},
        index=ORDER5
    )

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
    gpsr_estado_counts["%"] = (
        gpsr_estado_counts["VIN_unicos"] /
        gpsr_estado_counts["VIN_unicos"].sum() * 100
    ).round(2)
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
    st.info("No se encontró historico_conectividad.xlsx (hoja 'historico') o está vacío.")
else:
    # Copia y ordena
    h = hist_df.copy()
    h = h.dropna(subset=["fecha"])
    h = h.sort_values("fecha")

    # Pivot: filas = fecha, columnas = resumen, valores = pct_0_30
    pivot = (
        h.pivot(index="fecha", columns="resumen", values="pct_0_30")
         .sort_index()
    )

    # Renombrar columnas para la leyenda (ajusta los textos si quieres)
    col_rename = {
        "Telemetría": "Telemetría",
        "GPS (según REGLA)": "GPS_Copiloto",
        "GPS (todas con gps_timestamp)": "wicar_gps",
    }
    pivot = pivot.rename(columns=col_rename)

    # Solo mantener las columnas que existan
    cols_exist = [c for c in ["Telemetría", "GPS_Copiloto", "wicar_gps"] if c in pivot.columns]
    pivot = pivot[cols_exist]

    if pivot.empty:
        st.info("No hay columnas válidas para graficar en el histórico.")
    else:
        st.line_chart(pivot)          # Histograma tipo líneas (como tu gráfico)
        st.dataframe(pivot.tail(20))  # Tabla con las últimas filas

st.markdown("---")
st.caption("Dashboard local – basado en master_status_inner_qs_ready.csv e historico_conectividad.xlsx")
