from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import pandas as pd

@dataclass(frozen=True)
class DataReport:
    file_path: str
    n_rows: int
    n_cols: int
    columns: Tuple[str, ...]
    dropped_unnamed_columns: Tuple[str, ...]

    # Missing / duplicates
    missing_by_col: Dict[str, int]
    duplicate_rows_count: int

    # GPS sanity
    lat_col: Optional[str]
    lon_col: Optional[str]
    gps_missing_rows: int
    gps_invalid_rows: int
    lat_min: Optional[float]
    lat_max: Optional[float]
    lon_min: Optional[float]
    lon_max: Optional[float]

    # Date sanity (built from split columns)
    taken_dt_success_rate: Optional[float]
    upload_dt_success_rate: Optional[float]


def _resolve_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path

    if os.path.exists(path):
        return os.path.abspath(path)

    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(here, path))
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        "CSV file not found.\n"
        f"Tried:\n- {os.path.abspath(path)}\n- {candidate}\n"
        "Tip: put the CSV in project/data/ and call load_data('../data/your.csv') from src/main.py."
    )


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _drop_unnamed_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    unnamed = tuple([c for c in df.columns if str(c).startswith("unnamed:")])
    if unnamed:
        df = df.drop(columns=list(unnamed))
    return df, unnamed


def _pick_lat_lon_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = set(df.columns)
    lat_candidates = ["lat", "latitude"]
    lon_candidates = ["long", "lon", "lng", "longitude"]

    lat_col = next((c for c in lat_candidates if c in cols), None)
    lon_col = next((c for c in lon_candidates if c in cols), None)
    return lat_col, lon_col


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _gps_stats(df: pd.DataFrame, lat_col: str, lon_col: str) -> Tuple[int, int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    lat = _to_numeric(df[lat_col])
    lon = _to_numeric(df[lon_col])

    # missing rows: either lat or lon is missing
    missing_rows = int((lat.isna() | lon.isna()).sum())

    # invalid rows: out-of-bounds coords
    invalid_mask = (lat < -90) | (lat > 90) | (lon < -180) | (lon > 180)
    invalid_rows = int(invalid_mask.fillna(False).sum())

    lat_min = float(lat.min()) if lat.notna().any() else None
    lat_max = float(lat.max()) if lat.notna().any() else None
    lon_min = float(lon.min()) if lon.notna().any() else None
    lon_max = float(lon.max()) if lon.notna().any() else None

    return missing_rows, invalid_rows, lat_min, lat_max, lon_min, lon_max


def _build_datetime_from_split_fields(df: pd.DataFrame, prefix: str) -> pd.Series:
    required = [f"{prefix}_year", f"{prefix}_month", f"{prefix}_day", f"{prefix}_hour", f"{prefix}_minute"]
    for c in required:
        if c not in df.columns:
            # missing columns -> all NaT
            return pd.to_datetime(pd.Series([pd.NA] * len(df)), errors="coerce", utc=True)

    dt_df = pd.DataFrame(
        {
            "year": _to_numeric(df[f"{prefix}_year"]),
            "month": _to_numeric(df[f"{prefix}_month"]),
            "day": _to_numeric(df[f"{prefix}_day"]),
            "hour": _to_numeric(df[f"{prefix}_hour"]),
            "minute": _to_numeric(df[f"{prefix}_minute"]),
        }
    )
    return pd.to_datetime(dt_df, errors="coerce", utc=True)


def _success_rate(dt: pd.Series) -> Optional[float]:
    if len(dt) == 0:
        return None
    return float(dt.notna().sum() / len(dt))


def _recommend_dtypes(columns: List[str]) -> Dict[str, str]:
    dtypes: Dict[str, str] = {}
    for c in columns:
        if c in ("tags", "title", "user", "id"):
            dtypes[c] = "string"
        elif c.startswith("date_taken_") or c.startswith("date_upload_"):
            # minute/hour/day/month/year: integers but may have missing -> pandas nullable Int64
            dtypes[c] = "Int64"
        elif c in ("lat", "latitude", "long", "lon", "lng", "longitude"):
            dtypes[c] = "float64"
        else:
            pass
    return dtypes

def load_data(
    csv_path: str,
    *,
    sep: str = ",",
    encoding: Optional[str] = None,
    low_memory: bool = True,
    drop_unnamed: bool = True,
    normalize_columns: bool = True,
) -> Tuple[pd.DataFrame, DataReport]:

    resolved = _resolve_path(csv_path)

    header_df = pd.read_csv(resolved, sep=sep, encoding=encoding, nrows=0)
    header_cols = [str(c).strip().lower() for c in header_df.columns]
    dtypes = _recommend_dtypes(header_cols)

    df = pd.read_csv(
        resolved,
        sep=sep,
        encoding=encoding,
        low_memory=low_memory,
        dtype=dtypes if dtypes else None,
    )

    if normalize_columns:
        df = _normalize_column_names(df)

    dropped_unnamed_cols: Tuple[str, ...] = ()
    if drop_unnamed:
        df, dropped_unnamed_cols = _drop_unnamed_columns(df)

    n_rows, n_cols = df.shape
    columns = tuple(df.columns.tolist())
    missing_by_col = {c: int(df[c].isna().sum()) for c in df.columns}
    duplicate_rows_count = int(df.duplicated().sum())

    lat_col, lon_col = _pick_lat_lon_columns(df)
    if lat_col and lon_col:
        gps_missing_rows, gps_invalid_rows, lat_min, lat_max, lon_min, lon_max = _gps_stats(df, lat_col, lon_col)
    else:
        gps_missing_rows, gps_invalid_rows, lat_min, lat_max, lon_min, lon_max = 0, 0, None, None, None, None

    taken_dt = _build_datetime_from_split_fields(df, "date_taken")
    upload_dt = _build_datetime_from_split_fields(df, "date_upload")
    taken_rate = _success_rate(taken_dt)
    upload_rate = _success_rate(upload_dt)

    report = DataReport(
        file_path=resolved,
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        dropped_unnamed_columns=dropped_unnamed_cols,
        missing_by_col=missing_by_col,
        duplicate_rows_count=duplicate_rows_count,
        lat_col=lat_col,
        lon_col=lon_col,
        gps_missing_rows=gps_missing_rows,
        gps_invalid_rows=gps_invalid_rows,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        taken_dt_success_rate=taken_rate,
        upload_dt_success_rate=upload_rate,
    )

    return df, report


def print_report(report: DataReport, top_missing: int = 12) -> None:
    """
    Print a clean console report (perfect for Session 1 milestone demo).
    """
    print("\n" + "=" * 92)
    print("SESSION 1 — DATA EXPLORATION REPORT")
    print("=" * 92)
    print(f"File:   {report.file_path}")
    print(f"Shape:  {report.n_rows:,} rows × {report.n_cols:,} columns")
    print(f"Cols:   {', '.join(report.columns)}")

    if report.dropped_unnamed_columns:
        print("\n[Info] Dropped trailing empty columns (from ',,,' in CSV):")
        print("       " + ", ".join(report.dropped_unnamed_columns))

    print("\n--- Missing values (top columns) ---")
    missing_sorted = sorted(report.missing_by_col.items(), key=lambda kv: kv[1], reverse=True)

    shown = 0
    for col, miss in missing_sorted:
        if miss <= 0:
            continue
        pct = (miss / report.n_rows) * 100 if report.n_rows else 0.0
        print(f"{shown + 1:>2}. {col:<25} {miss:>10,}  ({pct:>6.2f}%)")
        shown += 1
        if shown >= top_missing:
            break
    if shown == 0:
        print("No missing values detected.")

    print("\n--- Duplicates ---")
    print(f"Duplicate rows: {report.duplicate_rows_count:,}")

    print("\n--- GPS sanity ---")
    if not report.lat_col or not report.lon_col:
        print("No lat/lon columns detected.")
    else:
        print(f"Columns used: lat='{report.lat_col}', lon='{report.lon_col}'")
        print(f"Latitude  min/max: {report.lat_min} / {report.lat_max}")
        print(f"Longitude min/max: {report.lon_min} / {report.lon_max}")
        print(f"Rows with missing GPS (lat OR lon missing): {report.gps_missing_rows:,}")
        print(f"Rows with invalid GPS (out of bounds):      {report.gps_invalid_rows:,}")

    print("\n--- Date sanity (built from split columns) ---")
    if report.taken_dt_success_rate is None:
        print("date_taken_* fields: N/A")
    else:
        print(f"date_taken_* → datetime parse success:  {report.taken_dt_success_rate * 100:.2f}%")

    if report.upload_dt_success_rate is None:
        print("date_upload_* fields: N/A")
    else:
        print(f"date_upload_* → datetime parse success: {report.upload_dt_success_rate * 100:.2f}%")

    print("=" * 92 + "\n")


if __name__ == "__main__":

    try:
        df_, rep_ = load_data("../data/flickr_data2.csv")
        print_report(rep_)
        print("Head(3):")
        print(df_.head(3))
    except Exception as e:
        print(f"[ERROR] {e}")
