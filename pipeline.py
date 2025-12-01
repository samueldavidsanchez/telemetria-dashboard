# -*- coding: utf-8 -*-

from pathlib import Path
from dotenv import load_dotenv  # pip install python-dotenv

# Cargar .env si existe (para credenciales y configs)
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

import os
import re
import sys
import shutil
import logging
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- S3 / QuickSight (opcional) ---
import boto3
import botocore

# =========================
# Logging
# =========================
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "outQS" / "job.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")

# =========================
# Config básica y paths (RELATIVOS AL REPO)
# =========================
DATA_DIR = BASE_DIR / "data"      # carpeta donde están/irán los archivos del dashboard
OUT_DIR = DATA_DIR                # usamos data/ también como salida

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Entradas
# master de flota (df_master_flota)
PATH_MASTER = os.getenv("PATH_MASTER", str(DATA_DIR / "master_Flota.xlsx"))

# Histórico semilla (opcional). Si ya tienes data/historico_conectividad.xlsx, se usará directamente.
SNAPSHOT_SEED_IN = DATA_DIR / "historico_conectividad.xlsx"

# Salidas (los dos que usa el dashboard + inner completo)
OUT_INNER_XLSX = str(DATA_DIR / "master_status_inner.xlsx")
OUT_QS_CSV     = str(DATA_DIR / "master_status_inner_qs_ready.csv")
SNAPSHOT_PATH  = DATA_DIR / "historico_conectividad.xlsx"  # mismo nombre que el que lee app.py

# Endpoints Copiloto
COPILOTO_ENDPOINT    = "https://api.copiloto.ai/wicar-report/report-files/vehicle-records"
COPILOTO_SIGNIN_URL  = os.getenv("COPILOTO_SIGNIN_URL", "https://accounts.copiloto.ai/v1/sign-in").strip()

# Credenciales (idealmente vía .env o variables de entorno)
COPILOTO_EMAIL    = os.getenv("COPILOTO_EMAIL", "").strip()
COPILOTO_PASSWORD = os.getenv("COPILOTO_PASSWORD", "").strip()

# Token directo por entorno (si ya lo tienes y quieres forzarlo)
COPILOTO_TOKEN_ENV = os.getenv("COPILOTO_TOKEN", "").strip()

# === S3 / QuickSight (opcionales) ===
AWS_REGION     = os.getenv("AWS_REGION", "sa-east-1")
S3_BUCKET      = os.getenv("S3_BUCKET", "").strip()
S3_PREFIX      = os.getenv("S3_PREFIX", "quicksight").strip()
QS_ACCOUNT_ID  = os.getenv("QS_ACCOUNT_ID", "").strip()
QS_DATASET_IDS = [x.strip() for x in os.getenv("QS_DATASET_IDS", "").split(",") if x.strip()]

# Constantes
ORDER5   = ["Conectado 0-2", "Intermitente 3-14", "Limitado 15-30+", "Desconectado 31+", "Nunca"]
TZ_CL    = pytz.timezone("America/Santiago")
RUN_DATE = pd.Timestamp.now(tz=TZ_CL).normalize().date()

# Descargar CSV de Dataflota en data/
DL_DIR = DATA_DIR

# =========================
# Helpers generales
# =========================
def no_accents_upper(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().upper()

def norm_vin(s):
    if pd.isna(s):
        return pd.NA
    s = str(s).upper().strip()
    return re.sub(r"\s+", "", s)

def find_regla_column(df: pd.DataFrame) -> Optional[str]:
    def _n(x):
        x = unicodedata.normalize("NFKD", str(x))
        x = "".join(ch for ch in x if not unicodedata.combining(ch))
        return x.strip().upper()
    names = {c: _n(c) for c in df.columns}
    cands = [c for c, n in names.items() if "REGLA" in n and ("REPORT" in n or "REPORTABILIDAD" in n or "REPORTA" in n)]
    if cands:
        return cands[0]
    for c, n in names.items():
        if n == "REGLA":
            return c
    return None

def clasificar_5rangos(ts: pd.Series, dias: pd.Series) -> pd.Categorical:
    out = pd.Series(index=dias.index, dtype="object")
    is_na = ts.isna()
    out[is_na] = "Nunca"
    m = ~is_na
    out[m & (dias <= 2)]                              = "Conectado 0-2"
    out[m & dias.between(3, 14, inclusive="both")]   = "Intermitente 3-14"
    out[m & dias.between(15, 30, inclusive="both")]  = "Limitado 15-30+"
    out[m & (dias >= 31)]                            = "Desconectado 31+"
    return pd.Categorical(out, categories=ORDER5, ordered=True)

def row_from_counts(label, counts, total, fecha) -> pd.DataFrame:
    counts = counts.reindex(ORDER5).fillna(0).astype(int)
    pct = (counts / (counts.sum() if counts.sum() else 1) * 100).round(2)
    return pd.DataFrame([{
        "fecha": fecha, "resumen": label, "total_vin": int(total),
        "cnt_Conectado 0-2": int(counts.get("Conectado 0-2", 0)),
        "cnt_Intermitente 3-14": int(counts.get("Intermitente 3-14", 0)),
        "cnt_Limitado 15-30+": int(counts.get("Limitado 15-30+", 0)),
        "cnt_Desconectado 31+": int(counts.get("Desconectado 31+", 0)),
        "cnt_Nunca": int(counts.get("Nunca", 0)),
        "pct_Conectado 0-2": float(pct.get("Conectado 0-2", 0.0)),
        "pct_Intermitente 3-14": float(pct.get("Intermitente 3-14", 0.0)),
        "pct_Limitado 15-30+": float(pct.get("Limitado 15-30+", 0.0)),
        "pct_Desconectado 31+": float(pct.get("Desconectado 31+", 0.0)),
        "pct_Nunca": float(pct.get("Nunca", 0.0)),
    }])

def seed_historico_if_needed():
    """
    Si no existe data/historico_conectividad.xlsx pero sí hay un archivo
    de semilla (por defecto mismo path), lo copia.
    """
    if (not SNAPSHOT_PATH.exists()) and SNAPSHOT_SEED_IN.exists():
        shutil.copy2(SNAPSHOT_SEED_IN, SNAPSHOT_PATH)
        log.info(f"Histórico inicial copiado: {SNAPSHOT_SEED_IN} → {SNAPSHOT_PATH}")

# =========================
# Login para obtener token
# =========================
def fetch_copiloto_token(email: str, password: str, signin_url: str = COPILOTO_SIGNIN_URL, timeout: int = 45) -> str:
    """POST /v1/sign-in y devuelve el JWT."""
    if not (email and password):
        raise RuntimeError("Faltan COPILOTO_EMAIL/COPILOTO_PASSWORD para login.")

    sess = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))

    payload = {"email": email, "password": password}
    r = sess.post(signin_url, json=payload, timeout=timeout)
    if r.status_code in (401, 403):
        raise RuntimeError("Credenciales Copiloto inválidas (401/403).")
    r.raise_for_status()

    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Respuesta no-JSON del login: {r.text[:300]}")

    token = (
        data.get("accessToken")
        or data.get("access_token")
        or data.get("token")
        or (data.get("data") or {}).get("token")
        or (data.get("data") or {}).get("accessToken")
        or ""
    )

    if not isinstance(token, str) or not token.strip():
        raise RuntimeError(f"No encontré token en la respuesta de login. Cuerpo: {data}")

    token = token.strip()
    log.info("Token obtenido vía sign-in (longitud %d).", len(token))
    return token

def resolve_token() -> str:
    """Si hay COPILOTO_TOKEN en entorno, úsalo. Si no, login con email/pass."""
    if COPILOTO_TOKEN_ENV:
        log.info("Usando COPILOTO_TOKEN desde entorno.")
        return COPILOTO_TOKEN_ENV
    return fetch_copiloto_token(COPILOTO_EMAIL, COPILOTO_PASSWORD, COPILOTO_SIGNIN_URL)

# =========================
# Descarga CSV Dataflota
# =========================
def download_dataflota_csv(endpoint: str, bearer_token: str, out_dir: Path) -> Path:
    if not bearer_token:
        raise RuntimeError("Falta token Bearer para descargar Dataflota.")
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))

    headers = {"Authorization": f"Bearer {bearer_token}"}
    r = sess.get(endpoint, headers=headers, timeout=60, stream=True)
    if r.status_code == 401:
        raise RuntimeError("Token inválido o expirado (401) al descargar CSV.")
    r.raise_for_status()

    cd = r.headers.get("Content-Disposition", "")
    m = re.search(r'filename="?([^"]+)"?', cd)
    fname = m.group(1) if m else f"Dataflota_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    out_path = out_dir / fname

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if out_path.suffix.lower() != ".csv":
        text_head = out_path.read_text(errors="ignore")[:300]
        raise RuntimeError(f"El endpoint no devolvió CSV. Cabecera:\n{text_head}")

    log.info(f"CSV descargado: {out_path}")
    return out_path

# ========= Helpers S3 / QuickSight =========
def upload_file_to_s3(local_path: str, key: str, *, content_type: Optional[str] = None):
    if not S3_BUCKET:
        return
    s3 = boto3.client("s3", region_name=AWS_REGION)
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    elif key.lower().endswith(".csv"):
        extra["ContentType"] = "text/csv"
    elif key.lower().endswith(".xlsx"):
        extra["ContentType"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    log.info(f"Subiendo {local_path} → s3://{S3_BUCKET}/{key}")
    try:
        s3.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs=extra)
    except botocore.exceptions.ClientError as e:
        log.error(f"Error subiendo a S3 ({key}): {e}")
        raise
    log.info("Subida OK")

def trigger_quicksight_ingestion():
    if not (QS_ACCOUNT_ID and QS_DATASET_IDS):
        return
    qs = boto3.client("quicksight", region_name=AWS_REGION)
    for ds in QS_DATASET_IDS:
        ing_id = f"ing-{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}"
        try:
            resp = qs.create_ingestion(AwsAccountId=QS_ACCOUNT_ID, DataSetId=ds, IngestionId=ing_id)
            log.info(f"QuickSight ingestion solicitada para {ds} → {resp.get('IngestionArn')}")
        except botocore.exceptions.ClientError as e:
            log.error(f"Error al crear ingestion para {ds}: {e}")

# =========================
# Pipeline
# =========================
def main() -> int:
    try:
        seed_historico_if_needed()

        # === 0) Resolver token (env directo o login) ===
        token = resolve_token()

        # === 1) Descargar snapshot del día ===
        csv_path = download_dataflota_csv(COPILOTO_ENDPOINT, token, DL_DIR)
        try:
            df_status_flota = pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df_status_flota = pd.read_csv(csv_path, encoding="cp1252")
        log.info(f"Status flota: {len(df_status_flota)} filas")

        # === 2) Cargar master ===
        if not Path(PATH_MASTER).exists():
            raise FileNotFoundError(f"No existe PATH_MASTER: {PATH_MASTER}")
        df_master_flota = pd.read_excel(PATH_MASTER)
        log.info(f"Master flota:  {len(df_master_flota)} filas")

        # === 3) Prep master (dedup VIN) ===
        dfm = df_master_flota.copy()
        vin_col_m = "VIN" if "VIN" in dfm.columns else next((c for c in dfm.columns if c.lower() == "vin"), None)
        if vin_col_m is None:
            raise KeyError("df_master_flota: no encontré columna VIN")
        dfm["VIN_norm"] = dfm[vin_col_m].map(norm_vin)
        dfm = dfm.sort_values(by=["VIN_norm"]).drop_duplicates(subset=["VIN_norm"], keep="first")

        # === 4) Prep status (dedup VIN + días) ===
        dfs = df_status_flota.copy()
        vin_col_s = "VIN" if "VIN" in dfs.columns else next((c for c in dfs.columns if c.lower() == "vin"), None)
        if vin_col_s is None:
            raise KeyError("df_status_flota: no encontré columna VIN")
        dfs["VIN_norm"] = dfs[vin_col_s].map(norm_vin)

        for c in ["gps_timestamp", "can_timestamp", "last_update_utc", "last_header_gps", "last_header_can"]:
            if c in dfs.columns and not pd.api.types.is_datetime64_any_dtype(dfs[c]):
                dfs[c] = pd.to_datetime(dfs[c], utc=True, errors="coerce")

        today_utc = pd.Timestamp.now(tz="UTC").normalize()
        if "gps_timestamp" in dfs.columns:
            gps_date = pd.to_datetime(dfs["gps_timestamp"].dt.date, errors="coerce")
            dfs["dias_sin_gps"] = (pd.to_datetime(pd.Series([today_utc.date()] * len(dfs))) - gps_date).dt.days.astype("Int64")
            dfs["dias_sin_gps"] = dfs["dias_sin_gps"].where(dfs["dias_sin_gps"].isna() | (dfs["dias_sin_gps"] >= 0), 0)
        else:
            dfs["dias_sin_gps"] = pd.Series(pd.NA, index=dfs.index, dtype="Int64")

        if "can_timestamp" in dfs.columns:
            can_date = pd.to_datetime(dfs["can_timestamp"].dt.date, errors="coerce")
            dfs["dias_sin_can"] = (pd.to_datetime(pd.Series([today_utc.date()] * len(dfs))) - can_date).dt.days.astype("Int64")
            dfs["dias_sin_can"] = dfs["dias_sin_can"].where(dfs["dias_sin_can"].isna() | (dfs["dias_sin_can"] >= 0), 0)
        else:
            dfs["dias_sin_can"] = pd.Series(pd.NA, index=dfs.index, dtype="Int64")

        dfs["t_ref"] = pd.concat(
            [dfs.get("last_update_utc"), dfs.get("gps_timestamp"), dfs.get("can_timestamp")],
            axis=1
        ).max(axis=1)
        dfs = dfs.sort_values(["VIN_norm", "t_ref"], ascending=[True, False]).drop_duplicates(subset=["VIN_norm"], keep="first")

        # === 5) INNER JOIN ===
        df_inner = pd.merge(dfm, dfs, on="VIN_norm", how="inner", suffixes=("_master", "_status"))
        df_inner["VIN_final"] = df_inner.get("VIN_master", df_inner.get("VIN_status", df_inner["VIN_norm"]))
        log.info(f"INNER: {len(df_inner)} filas — VIN únicos: {df_inner['VIN_norm'].nunique(dropna=True)}")

        # === 6) Normalizar regla y clasificar ===
        regla_col = find_regla_column(df_inner)
        if regla_col:
            df_inner["regla_norm"] = df_inner[regla_col].map(no_accents_upper)
            mask_tlm = df_inner["regla_norm"].str.contains("TELEMETRIA", na=False)
        else:
            mask_tlm = df_inner["can_timestamp"].notna() if "can_timestamp" in df_inner.columns else pd.Series(False, index=df_inner.index)
            df_inner["regla_norm"] = np.where(mask_tlm, "TELEMETRIA (inferida)", "GPS (inferida)")
        mask_gps = ~mask_tlm

        for c in ["dias_sin_gps", "dias_sin_can"]:
            if c in df_inner.columns:
                df_inner[c] = pd.to_numeric(df_inner[c], errors="coerce").astype("Int64")

        df_inner["estado_can"] = clasificar_5rangos(
            df_inner.get("can_timestamp", pd.Series(pd.NaT, index=df_inner.index)),
            df_inner["dias_sin_can"]
        )
        df_inner["estado_gps"] = clasificar_5rangos(
            df_inner.get("gps_timestamp", pd.Series(pd.NaT, index=df_inner.index)),
            df_inner["dias_sin_gps"]
        )

        # === 7) Resúmenes ===
        df_tlm = df_inner[mask_tlm].copy()
        cnt_tlm = df_tlm["estado_can"].value_counts()
        tot_tlm = int(cnt_tlm.sum())

        df_gps_reg = df_inner[mask_gps].copy()
        cnt_gps_reg = df_gps_reg["estado_gps"].value_counts()
        tot_gps_reg = int(cnt_gps_reg.sum())

        has_gps_ts = df_inner["gps_timestamp"].notna() if "gps_timestamp" in df_inner.columns else pd.Series(False, index=df_inner.index)
        df_gps_any = df_inner[has_gps_ts].copy()
        cnt_gps_any = df_gps_any["estado_gps"].value_counts()
        tot_gps_any = int(cnt_gps_any.sum())

        snapshot_today = pd.concat(
            [
                row_from_counts("Telemetría", cnt_tlm, tot_tlm, RUN_DATE),
                row_from_counts("GPS (según REGLA)", cnt_gps_reg, tot_gps_reg, RUN_DATE),
                row_from_counts("GPS (todas con gps_timestamp)", cnt_gps_any, tot_gps_any, RUN_DATE),
            ],
            ignore_index=True
        )

        # === 8) Guardar INNER (Excel) + CSV QS (columnas exactas) ===
        # Datetime tz-aware → naive para Excel
        for col in df_inner.select_dtypes(include=['datetime64[ns, UTC]']).columns:
            df_inner[col] = df_inner[col].dt.tz_convert(None)

        with pd.ExcelWriter(OUT_INNER_XLSX, engine="xlsxwriter") as xw:
            df_inner.to_excel(xw, "inner_master_status", index=False)

        df_qs = df_inner.copy()
        if "VIN" not in df_qs.columns and "VIN_final" in df_qs.columns:
            df_qs = df_qs.rename(columns={"VIN_final": "VIN"})

        cols_finales = [
            'Nombre', 'Patente', 'VIN_master', 'IMEI_master', 'Empresa', 'Marca', 'Modelo', 'Activo',
            'PLAN', 'Telemetría según plan', 'DISPOSITIVO', 'Telemetría según dispositivo',
            'REGLA GENERAL DE REPORTABILIDAD', 'Unnamed: 13', 'Unnamed: 14', 'VIN_norm',
            'is_active_master', 'IMEI_status', 'VIN_status', 'license_plate', 'source', 'device_model',
            'gps_odometer', 'latitude', 'longitude', 'gps_timestamp', 'last_header_gps', 'can_odometer',
            'can_horometer', 'can_odoliter', 'can_timestamp', 'last_header_can', 'has_gps_data',
            'has_can_data', 'last_update_utc', 'is_active_status', 'dias_sin_gps', 'dias_sin_can',
            't_ref', 'VIN'
        ]
        cols_validas = [c for c in cols_finales if c in df_qs.columns]
        cols_faltantes = [c for c in cols_finales if c not in df_qs.columns]
        if cols_faltantes:
            log.warning(f"Columnas faltantes en df_inner (no incluidas en CSV): {cols_faltantes}")

        df_qs = df_qs[cols_validas]
        df_qs.to_csv(OUT_QS_CSV, index=False, encoding="utf-8-sig")
        log.info(f"CSV QS generado con columnas filtradas: {len(cols_validas)} columnas, {len(df_qs)} filas")
        log.info(f"INNER y CSV QS generados: {OUT_INNER_XLSX} | {OUT_QS_CSV}")

        # === 9) Actualizar histórico (sobrescribe solo el día) ===
        if SNAPSHOT_PATH.exists():
            try:
                hist = pd.read_excel(SNAPSHOT_PATH, sheet_name="historico")
            except Exception:
                hist = pd.DataFrame()
        else:
            hist = pd.DataFrame()

        if not hist.empty and "fecha" in hist.columns:
            hist["fecha"] = pd.to_datetime(hist["fecha"], errors="coerce").dt.date

        if not hist.empty:
            if set(hist.columns) == set(snapshot_today.columns):
                snapshot_today = snapshot_today[hist.columns]
            else:
                common = [c for c in hist.columns if c in snapshot_today.columns]
                snapshot_today = snapshot_today[common]
                hist = hist[common]

        if not hist.empty:
            hist = hist[hist["fecha"] != RUN_DATE]
        hist_new = pd.concat([hist, snapshot_today], ignore_index=True)

        if "resumen" in hist_new.columns:
            orden_res = ["Telemetría", "GPS (según REGLA)", "GPS (todas con gps_timestamp)"]
            try:
                hist_new["resumen"] = pd.Categorical(hist_new["resumen"], categories=orden_res, ordered=True)
            except Exception:
                pass
            hist_new = hist_new.sort_values(["fecha", "resumen"]).reset_index(drop=True)
        else:
            hist_new = hist_new.sort_values(["fecha"]).reset_index(drop=True)

        with pd.ExcelWriter(SNAPSHOT_PATH, engine="openpyxl", mode="w") as w:
            hist_new.to_excel(w, sheet_name="historico", index=False)
            snapshot_today.to_excel(w, sheet_name="ultimo_snapshot", index=False)

        log.info(f"Día {RUN_DATE} guardado en {SNAPSHOT_PATH} (solo se reemplazó ese día)")

        # === 10) SUBIDA A S3 + QuickSight (opcional) ===
        if S3_BUCKET:
            key_hist_fijo = f"{S3_PREFIX}/historico/historico_conectividad.xlsx"
            key_csv_fijo  = f"{S3_PREFIX}/inner/master_status_inner_qs_ready.csv"

            upload_file_to_s3(SNAPSHOT_PATH, key_hist_fijo)
            upload_file_to_s3(OUT_QS_CSV,   key_csv_fijo)

            today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
            upload_file_to_s3(SNAPSHOT_PATH, f"{S3_PREFIX}/historico/daily/historico_conectividad_{today_str}.xlsx")
            upload_file_to_s3(OUT_QS_CSV,   f"{S3_PREFIX}/inner/daily/master_status_inner_qs_ready_{today_str}.csv")

            trigger_quicksight_ingestion()
        else:
            log.info("S3_BUCKET no configurado: salto subida a S3 y refresh de QuickSight.")

        return 0

    except Exception as e:
        log.error(f"Fallo en pipeline: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
