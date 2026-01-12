from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import pandas as pd

GRAND_LYON_BBOX = {
    "lat_min": 45.60,
    "lat_max": 45.90,
    "lon_min": 4.70,
    "lon_max": 5.05,
}

@dataclass(frozen=True)
class CleaningReport:
    n_rows_before: int
    n_rows_after: int
    dropped_unnamed_columns: Tuple[str, ...]
    renamed_columns: Dict[str, str]
    coerced_numeric_cells: Dict[str, int]
    removed_missing_gps_rows: int
    removed_invalid_gps_rows: int
    removed_outside_bbox_rows: int
    removed_duplicate_coords_rows: int  
    taken_dt_success_rate: Optional[float]
    upload_dt_success_rate: Optional[float]
    removed_invalid_time_rows: int  
    filled_missing_tags: int
    filled_missing_title: int
    removed_duplicates_by_id: int
    removed_exact_row_duplicates: int
    dedup_strategy: str

def _drop_unnamed_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    unnamed = tuple([c for c in df.columns if str(c).lower().startswith("unnamed:")])
    if unnamed:
        df = df.drop(columns=list(unnamed))
    return df, unnamed

def _normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    original_cols = list(df.columns)
    normalized_cols = [str(c).strip().lower() for c in original_cols]
    df.columns = normalized_cols
    rename_map: Dict[str, str] = {}
    if "long" not in df.columns:
        for alt in ("lon", "lng", "longitude"):
            if alt in df.columns:
                df = df.rename(columns={alt: "long"})
                rename_map[alt] = "long"
                break
    if "lat" not in df.columns and "latitude" in df.columns:
        df = df.rename(columns={"latitude": "lat"})
        rename_map["latitude"] = "lat"
    if "id" not in df.columns:
        for alt in ("photo_id", "id_photo"):
            if alt in df.columns:
                df = df.rename(columns={alt: "id"})
                rename_map[alt] = "id"
                break
    if "user" not in df.columns:
        for alt in ("owner", "photographer", "id_photographe"):
            if alt in df.columns:
                df = df.rename(columns={alt: "user"})
                rename_map[alt] = "user"
                break

    return df, rename_map

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _coerce_numeric_columns(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    coerced_counts: Dict[str, int] = {}

    for c in cols:
        if c not in df.columns:
            continue
        before_nonnull = int(df[c].notna().sum())
        df[c] = _to_numeric(df[c])
        after_nonnull = int(df[c].notna().sum())
        coerced_counts[c] = max(0, before_nonnull - after_nonnull)

    return df, coerced_counts

def _build_datetime(df: pd.DataFrame, prefix: str) -> pd.Series:
    required = [f"{prefix}_year", f"{prefix}_month", f"{prefix}_day", f"{prefix}_hour", f"{prefix}_minute"]
    for c in required:
        if c not in df.columns:
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

def _bbox_mask(df: pd.DataFrame, lat_col: str, lon_col: str, bbox: Dict[str, float]) -> pd.Series:
    return (
        (df[lat_col] >= bbox["lat_min"])
        & (df[lat_col] <= bbox["lat_max"])
        & (df[lon_col] >= bbox["lon_min"])
        & (df[lon_col] <= bbox["lon_max"])
    )

def _clean_text_basic(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype("string")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return s

def _normalize_tags(s: pd.Series) -> pd.Series:
    s = _clean_text_basic(s)
    s = s.str.replace(",", " ").str.replace(";", " ").str.replace("|", " ")

    def normalize_row_tags(txt: str) -> str:
        if not txt:
            return ""
        tokens = re.findall(r"[a-z0-9_\-@]+", txt)
        seen = set()
        out = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return " ".join(out)

    return s.apply(normalize_row_tags).astype("string")

def _remove_duplicate_coords(df: pd.DataFrame, lat_col: str, lon_col: str, precision: int = 5) -> Tuple[pd.DataFrame, int]:
    tmp = df.copy()
    tmp["_lat_r"] = tmp[lat_col].round(precision)
    tmp["_lon_r"] = tmp[lon_col].round(precision)

    before = len(tmp)
    tmp = tmp.drop_duplicates(subset=["_lat_r", "_lon_r"], keep="first")
    removed = before - len(tmp)

    tmp = tmp.drop(columns=["_lat_r", "_lon_r"])
    return tmp, int(removed)


def _deduplicate_by_id_keep_best_text(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if "id" not in df.columns:
        return df, 0

    tmp = df.copy()
    tags_len = tmp["tags"].fillna("").astype("string").str.len() if "tags" in tmp.columns else 0
    title_len = tmp["title"].fillna("").astype("string").str.len() if "title" in tmp.columns else 0
    tmp["_keep_score"] = tags_len + title_len

    tmp = tmp.sort_values(by=["id", "_keep_score"], ascending=[True, False])
    before = len(tmp)
    tmp = tmp.drop_duplicates(subset=["id"], keep="first")
    removed = before - len(tmp)

    tmp = tmp.drop(columns=["_keep_score"])
    return tmp, int(removed)

def clean_data(
    df: pd.DataFrame,
    *,
    bbox: Dict[str, float] = GRAND_LYON_BBOX,
    lat_col: str = "lat",
    lon_col: str = "long",
    drop_duplicate_coords: bool = False,   
    coords_precision: int = 5,
    strict_time_validation: bool = False,  
) -> Tuple[pd.DataFrame, CleaningReport]:
    n_before = int(len(df))
    work = df.copy()
    work, dropped_unnamed = _drop_unnamed_columns(work)
    work, rename_map = _normalize_column_names(work)

    numeric_cols = [
        "lat", "long",
        "date_taken_minute", "date_taken_hour", "date_taken_day", "date_taken_month", "date_taken_year",
        "date_upload_minute", "date_upload_hour", "date_upload_day", "date_upload_month", "date_upload_year",
    ]
    work, coerced_counts = _coerce_numeric_columns(work, numeric_cols)

    for c in ("id", "user", "tags", "title"):
        if c in work.columns:
            work[c] = work[c].astype("string")

    removed_missing_gps = 0
    removed_invalid_gps = 0
    removed_outside_bbox = 0
    removed_dup_coords = 0

    if lat_col in work.columns and lon_col in work.columns:
        missing_mask = work[lat_col].isna() | work[lon_col].isna()
        removed_missing_gps = int(missing_mask.sum())
        work = work.loc[~missing_mask].copy()
        invalid_mask = (work[lat_col] < -90) | (work[lat_col] > 90) | (work[lon_col] < -180) | (work[lon_col] > 180)
        removed_invalid_gps = int(invalid_mask.sum())
        work = work.loc[~invalid_mask].copy()

        bbox_mask = _bbox_mask(work, lat_col, lon_col, bbox)
        removed_outside_bbox = int((~bbox_mask).sum())
        work = work.loc[bbox_mask].copy()

        if drop_duplicate_coords:
            work, removed_dup_coords = _remove_duplicate_coords(work, lat_col, lon_col, precision=coords_precision)

    taken_dt = _build_datetime(work, "date_taken")
    upload_dt = _build_datetime(work, "date_upload")
    taken_rate = _success_rate(taken_dt)
    upload_rate = _success_rate(upload_dt)

    work["taken_dt"] = taken_dt
    work["upload_dt"] = upload_dt

    removed_invalid_time_rows = 0
    if strict_time_validation:
        invalid_time_mask = work["taken_dt"].isna() & work["upload_dt"].isna()
        removed_invalid_time_rows = int(invalid_time_mask.sum())
        work = work.loc[~invalid_time_mask].copy()

    dt_ref = work["taken_dt"].fillna(work["upload_dt"])
    work["year"] = dt_ref.dt.year.astype("Int64")
    work["month"] = dt_ref.dt.month.astype("Int64")
    work["day"] = dt_ref.dt.day.astype("Int64")
    work["hour"] = dt_ref.dt.hour.astype("Int64")

    filled_missing_tags = 0
    filled_missing_title = 0

    if "tags" in work.columns:
        filled_missing_tags = int(work["tags"].isna().sum())
        work["tags"] = _normalize_tags(work["tags"])

    if "title" in work.columns:
        filled_missing_title = int(work["title"].isna().sum())
        work["title"] = _clean_text_basic(work["title"])

    if "tags" in work.columns and "title" in work.columns:
        work["text"] = (work["title"].fillna("") + " " + work["tags"].fillna("")).str.strip()
    elif "title" in work.columns:
        work["text"] = work["title"].fillna("")
    elif "tags" in work.columns:
        work["text"] = work["tags"].fillna("")
    else:
        work["text"] = ""

    removed_by_id = 0
    removed_exact = 0
    dedup_strategy = "none"

    if "id" in work.columns:
        before = len(work)
        work, removed_by_id = _deduplicate_by_id_keep_best_text(work)
        removed_by_id = int(before - len(work))
        dedup_strategy = "by_photo_id_keep_best_text"

    before2 = len(work)
    work = work.drop_duplicates()
    removed_exact = int(before2 - len(work))

    work = work.reset_index(drop=True)
    n_after = int(len(work))

    report = CleaningReport(
        n_rows_before=n_before,
        n_rows_after=n_after,
        dropped_unnamed_columns=dropped_unnamed,
        renamed_columns=rename_map,
        coerced_numeric_cells=coerced_counts,
        removed_missing_gps_rows=removed_missing_gps,
        removed_invalid_gps_rows=removed_invalid_gps,
        removed_outside_bbox_rows=removed_outside_bbox,
        removed_duplicate_coords_rows=removed_dup_coords,
        taken_dt_success_rate=taken_rate,
        upload_dt_success_rate=upload_rate,
        removed_invalid_time_rows=removed_invalid_time_rows,
        filled_missing_tags=filled_missing_tags,
        filled_missing_title=filled_missing_title,
        removed_duplicates_by_id=removed_by_id,
        removed_exact_row_duplicates=removed_exact,
        dedup_strategy=dedup_strategy,
    )

    return work, report


def print_cleaning_report(rep: CleaningReport) -> None:
    print("\n" + "=" * 92)
    print("SESSION 1 — CLEANING REPORT")
    print("=" * 92)
    print(f"Rows before: {rep.n_rows_before:,}")
    print(f"Rows after:  {rep.n_rows_after:,}")
    print(f"Removed:     {rep.n_rows_before - rep.n_rows_after:,}")

    print("\n1) Schema/types normalization")
    if rep.dropped_unnamed_columns:
        print(f"- Dropped Unnamed columns: {', '.join(rep.dropped_unnamed_columns)}")
    else:
        print("- Dropped Unnamed columns: none")

    if rep.renamed_columns:
        print(f"- Renamed columns: {rep.renamed_columns}")
    else:
        print("- Renamed columns: none")

    coerced_issues = {k: v for k, v in rep.coerced_numeric_cells.items() if v > 0}
    if coerced_issues:
        print(f"- Non-numeric cells coerced to NaN (by col): {coerced_issues}")
    else:
        print("- Numeric coercion issues: none detected")

    print("\n2) Geographic cleaning (Grand Lyon bbox)")
    print(f"- Removed rows with missing GPS: {rep.removed_missing_gps_rows:,}")
    print(f"- Removed rows with invalid GPS: {rep.removed_invalid_gps_rows:,}")
    print(f"- Removed rows outside bbox:     {rep.removed_outside_bbox_rows:,}")
    print(f"- Removed duplicate coords:      {rep.removed_duplicate_coords_rows:,} (default OFF)")

    print("\n3) Temporal cleaning")
    if rep.taken_dt_success_rate is None:
        print("- taken_dt parse success: N/A")
    else:
        print(f"- taken_dt parse success:  {rep.taken_dt_success_rate * 100:.2f}%")

    if rep.upload_dt_success_rate is None:
        print("- upload_dt parse success: N/A")
    else:
        print(f"- upload_dt parse success: {rep.upload_dt_success_rate * 100:.2f}%")

    print(f"- Removed invalid time rows (strict mode): {rep.removed_invalid_time_rows:,}")

    print("\n4) Text cleaning")
    print(f"- Filled missing tags:  {rep.filled_missing_tags:,}")
    print(f"- Filled missing title: {rep.filled_missing_title:,}")

    print("\n5) Deduplication")
    print(f"- Strategy: {rep.dedup_strategy}")
    print(f"- Removed duplicates by id:      {rep.removed_duplicates_by_id:,}")
    print(f"- Removed exact row duplicates:  {rep.removed_exact_row_duplicates:,}")

    print("=" * 92 + "\n")

if __name__ == "__main__":
    try:
        from load_data import load_data, print_report

        df_raw, rep_raw = load_data("../data/flickr_data2.csv")
        print_report(rep_raw)

        df_clean, rep_clean = clean_data(df_raw)  
        print_cleaning_report(rep_clean)

        print("Clean Head(3):")
        print(df_clean.head(3))

        print("\nClean columns:")
        print(df_clean.columns.tolist())
    except Exception as e:
        print(f"[ERROR] {e}")
