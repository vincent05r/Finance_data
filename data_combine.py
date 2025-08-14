#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from dateutil import tz
from datetime import datetime, timedelta, timezone

# ----------------------------- Logging setup ----------------------------------
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("merge_financial_chunks")
    logger.setLevel(logging.INFO)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    # File handler (rotating)
    fh = RotatingFileHandler(log_dir / "merge.log", maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    # Attach
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

# ----------------------------- Configuration ----------------------------------

@dataclass
class Config:
    input_root: Path
    output_dir: Path
    tz_strategy: str = "to_utc"      # 'keep' | 'to_utc' | 'localize'
    assume_naive_tz: str = "UTC"     # used only if tz_strategy == 'localize' and timestamps are naive
    expected_freq_seconds: Optional[int] = None  # if provided, skip inference
    # gap detection policy:
    # flag gaps within a calendar day at delta > expected_freq * gap_factor
    gap_factor: float = 1.5
    max_within_day_gap_hours: float = 6.0  # beyond this we assume session break / trading halt & do not flag
    # CSV read options:
    csv_engine: str = "pyarrow"  # fast & robust; fallback to 'c' on failure
    # Output options:
    write_parquet: bool = False  # for large data; keep False unless needed


# ----------------------------- Helpers ----------------------------------------

STANDARD_COLS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names
    rename_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename_map)
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(*aliases):
        for a in aliases:
            if a in df.columns: return a
            if a in cols_lower: return cols_lower[a]
        return None

    col_map = {
        "Datetime": pick("Datetime", "datetime", "Timestamp", "timestamp"),
        "Open": pick("Open", "open"),
        "High": pick("High", "high"),
        "Low": pick("Low", "low"),
        "Close": pick("Close", "close", "Adj Close", "adj_close", "adj close"),
        "Volume": pick("Volume", "volume", "Vol", "vol"),
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; found={list(df.columns)}")

    df = df[[col_map["Datetime"], col_map["Open"], col_map["High"], col_map["Low"], col_map["Close"], col_map["Volume"]]]
    df.columns = STANDARD_COLS
    return df

def parse_datetime_col(
    s: pd.Series,
    tz_strategy: str,
    assume_naive_tz: str,
    logger: logging.Logger
) -> pd.DatetimeIndex:
    # robust parsing: pandas handles offsets like +08:00; errors='coerce' to drop bad rows later
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    # Check tz-awareness
    if tz_strategy == "keep":
        # Allow mixed; convert tz-aware to UTC for index normalization later
        if hasattr(dt, "dt"):
            # If any tz-aware, convert entire series to UTC (coercing naive as UTC)
            aware_mask = dt.dt.tz is not None
        # We'll unify below anyway
        pass

    if tz_strategy == "to_utc":
        # If tz-aware, convert; if naive, assume naive timestamps are already UTC
        if getattr(dt.dt, "tz", None) is None:
            # naive -> assume UTC to avoid accidental local time misinterpretation
            dt = dt.dt.tz_localize("UTC")
        else:
            dt = dt.dt.tz_convert("UTC")

    elif tz_strategy == "localize":
        # naive -> localize to assume_naive_tz then convert to UTC
        tzinfo = tz.gettz(assume_naive_tz)
        if tzinfo is None:
            raise ValueError(f"Unknown timezone: {assume_naive_tz}")
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tzinfo).dt.tz_convert("UTC")
        else:
            dt = dt.dt.tz_convert("UTC")

    # Final normalization: ensure tz-aware UTC index
    if getattr(dt.dt, "tz", None) is None:
        logger.warning("Datetime parsed as naive despite strategy; forcing UTC localization.")
        dt = dt.dt.tz_localize("UTC")
    else:
        # ensure it's UTC
        dt = dt.dt.tz_convert("UTC")

    return pd.DatetimeIndex(dt)

def infer_bar_seconds(ts_like, logger: logging.Logger) -> int:
    """
    Infer expected bar size (in seconds) from intra-day deltas.
    Accepts a pandas Series/DatetimeIndex/array-like and coerces to a tz-aware UTC DatetimeIndex.
    Heuristic: use the modal delta among small (<1h) gaps within same calendar day.
    Fallbacks: 60s (minute) if inference fails.
    """
    # Coerce to DatetimeIndex
    if isinstance(ts_like, pd.Series):
        idx = pd.DatetimeIndex(ts_like)
    elif isinstance(ts_like, pd.DatetimeIndex):
        idx = ts_like
    else:
        idx = pd.DatetimeIndex(ts_like)

    # Normalize to tz-aware UTC
    if idx.tz is None:
        # If naive, assume it’s already UTC (matches the rest of the pipeline)
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    # Guard: too few points
    if len(idx) < 3:
        return 60

    # Ensure sorted (don’t mutate caller)
    idx = idx.sort_values()

    # Compute intra-day deltas (avoid overnight gaps dominating)
    days = idx.date
    same_day = np.concatenate([[False], days[1:] == days[:-1]])
    deltas = np.diff(idx.view("i8")) / 1e9  # seconds

    small = (deltas > 0) & (deltas <= 3600) & same_day
    if np.any(small):
        vals = deltas[small].astype(int)
        # Mode
        unique, counts = np.unique(vals, return_counts=True)
        return int(unique[np.argmax(counts)])

    logger.warning("Could not infer bar size from intra-day deltas; defaulting to 60s.")
    return 60


def list_all_symbol_files(input_root: Path, logger: logging.Logger) -> Dict[str, List[Path]]:
    """Walk input_root; assume structure: input_root/run_YYYYmmddHHMMSS/<symbol>.csv"""
    mapping: Dict[str, List[Path]] = {}
    for folder, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith(".csv"):
                sym = Path(f).stem  # 'AAPL' from 'AAPL.csv'
                mapping.setdefault(sym, []).append(Path(folder) / f)
    logger.info(f"Discovered {sum(len(v) for v in mapping.values())} CSV files across {len(mapping)} symbols.")
    return mapping

def read_one_csv(path: Path, engine: str, logger: logging.Logger) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, engine=engine)
    except Exception as e1:
        logger.warning(f"pyarrow read failed for {path.name} ({e1}); retrying with engine='c'.")
        df = pd.read_csv(path, engine="c", on_bad_lines="skip", low_memory=False)
    df = standardize_columns(df)
    df["__origin_file"] = str(path)
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        mtime = datetime.now(tz=timezone.utc)
    df["__origin_mtime"] = mtime
    return df

def resolve_duplicates_and_conflicts(
    df: pd.DataFrame, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For identical timestamps, deduplicate exact duplicates; if OHLCV disagree, keep the row from the newest file by mtime."""
    df = df.sort_values(["Datetime", "__origin_mtime"]).reset_index(drop=True)

    # Identify groups by timestamp
    grp = df.groupby("Datetime", sort=False, group_keys=False)

    conflicts = []
    kept_rows = []
    for ts, g in grp:
        if len(g) == 1:
            kept_rows.append(g.iloc[0])
            continue
        # duplicates if rows are identical on OHLCV
        dedup = g.drop_duplicates(subset=["Open", "High", "Low", "Close", "Volume"])
        if len(dedup) == 1:
            kept_rows.append(dedup.iloc[0])
        else:
            # conflict: choose the newest file (max mtime); record conflict
            idx = g["__origin_mtime"].values.argmax()
            chosen = g.iloc[int(idx)]
            kept_rows.append(chosen)
            # log all variants
            for i, r in g.iterrows():
                if i == chosen.name:
                    continue
                conflicts.append({
                    "Datetime": ts,
                    "kept_from": chosen["__origin_file"],
                    "discarded_from": r["__origin_file"],
                    "kept_mtime": chosen["__origin_mtime"],
                    "discarded_mtime": r["__origin_mtime"],
                    "kept_values": (chosen["Open"], chosen["High"], chosen["Low"], chosen["Close"], chosen["Volume"]),
                    "discarded_values": (r["Open"], r["High"], r["Low"], r["Close"], r["Volume"]),
                })

    out_df = pd.DataFrame(kept_rows)
    conflicts_df = pd.DataFrame(conflicts)
    return out_df, conflicts_df

def detect_within_day_gaps(
    ts: pd.DatetimeIndex,
    expected_sec: int,
    gap_factor: float,
    max_within_day_gap_hours: float,
) -> pd.DataFrame:
    """Return a DataFrame of gaps within a calendar day.
       We flag delta >= expected_sec * gap_factor and < max_within_day_gap_hours.
    """
    if len(ts) < 2:
        return pd.DataFrame(columns=["start", "end", "gap_seconds", "missing_bars"])

    ts = ts.sort_values()
    days = ts.tz_convert("UTC").date
    same_day = np.concatenate([[False], days[1:] == days[:-1]])
    deltas = np.diff(ts.view("i8")) / 1e9  # seconds

    min_gap = expected_sec * gap_factor
    max_gap = max_within_day_gap_hours * 3600.0

    mask = same_day & (deltas >= min_gap) & (deltas < max_gap)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return pd.DataFrame(columns=["start", "end", "gap_seconds", "missing_bars"])

    starts = ts[idxs]
    ends = ts[idxs + 1]
    gaps = (ends.view("i8") - starts.view("i8")) / 1e9
    missing = np.maximum((gaps / expected_sec) - 1.0, 0.0).round().astype(int)

    return pd.DataFrame({
        "start": starts,
        "end": ends,
        "gap_seconds": gaps.astype(int),
        "missing_bars": missing,
    })

def ensure_dirs(base: Path) -> Dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    dirs = {
        "combined": base / "combined",
        "reports": base / "reports",
        "logs": base / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs

def format_timerange_for_name(start: pd.Timestamp, end: pd.Timestamp) -> str:
    # Use UTC ISO-like format without colons for filenames
    s = start.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
    e = end.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
    return f"{s}__to__{e}"

# ----------------------------- Main pipeline ----------------------------------

def process_symbol(
    symbol: str,
    files: List[Path],
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame, Dict]:
    """Return (combined_df, gaps_df, conflicts_df, summary_dict)."""

    # Read & concat
    parts: List[pd.DataFrame] = []
    read_errors = 0
    for p in sorted(files):
        try:
            df = read_one_csv(p, cfg.csv_engine, logger)
            parts.append(df)
        except Exception as e:
            logger.error(f"[{symbol}] Failed to read {p}: {e}")
            read_errors += 1

    if not parts:
        return None, pd.DataFrame(), pd.DataFrame(), {
            "symbol": symbol,
            "n_files": len(files),
            "n_files_failed": read_errors,
            "n_rows_in": 0,
            "n_rows_out": 0,
            "duplicates_dropped": 0,
            "conflicts": 0,
            "gaps": 0,
            "missing_bars_total": 0,
            "expected_freq_seconds": None,
            "start": None,
            "end": None,
            "notes": "No readable input."
        }

    raw = pd.concat(parts, axis=0, ignore_index=True)

    # Parse datetime robustly -> UTC tz-aware index
    dt = parse_datetime_col(raw["Datetime"], cfg.tz_strategy, cfg.assume_naive_tz, logger)
    raw["Datetime"] = dt
    raw = raw.dropna(subset=["Datetime"])
    raw = raw.sort_values("Datetime").reset_index(drop=True)

    n_in = len(raw)

    # Resolve duplicates & conflicts
    resolved, conflicts_df = resolve_duplicates_and_conflicts(raw, logger)
    # Drop helper cols after resolving
    resolved = resolved.sort_values("Datetime").drop_duplicates(subset=["Datetime"], keep="last")
    # ensure dtypes
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        # Use float for OHLC, Int64 for volume if integral
        if col == "Volume":
            # if all integral -> Int64
            try:
                if pd.api.types.is_float_dtype(resolved["Volume"]):
                    if np.all(np.isfinite(resolved["Volume"].values)):
                        ints = (resolved["Volume"] % 1 == 0).all()
                        if ints:
                            resolved["Volume"] = resolved["Volume"].astype("Int64")
                elif pd.api.types.is_object_dtype(resolved["Volume"]):
                    resolved["Volume"] = pd.to_numeric(resolved["Volume"], errors="coerce").astype("Int64")
            except Exception:
                resolved["Volume"] = pd.to_numeric(resolved["Volume"], errors="coerce")
        else:
            resolved[col] = pd.to_numeric(resolved[col], errors="coerce")

    # infer expected frequency
    if cfg.expected_freq_seconds is not None:
        exp_sec = int(cfg.expected_freq_seconds)
    else:
        exp_sec = infer_bar_seconds(resolved["Datetime"], logger)

    # detect within-day gaps
    gaps_df = detect_within_day_gaps(
        resolved["Datetime"].dt.tz_convert("UTC"),
        expected_sec=exp_sec,
        gap_factor=cfg.gap_factor,
        max_within_day_gap_hours=cfg.max_within_day_gap_hours,
    )

    # Prepare outputs
    start = resolved["Datetime"].iloc[0]
    end = resolved["Datetime"].iloc[-1]
    timerange_str = format_timerange_for_name(start, end)

    # Finalize combined DF for writing (drop helper columns)
    write_df = resolved.drop(columns=["__origin_file", "__origin_mtime"], errors="ignore")
    write_df = write_df[STANDARD_COLS]  # enforce column order

    summary = {
        "symbol": symbol,
        "n_files": len(files),
        "n_files_failed": read_errors,
        "n_rows_in": n_in,
        "n_rows_out": len(write_df),
        "duplicates_dropped": int(n_in - len(resolved)),
        "conflicts": int(conflicts_df["Datetime"].nunique()) if not conflicts_df.empty else 0,
        "gaps": int(len(gaps_df)),
        "missing_bars_total": int(gaps_df["missing_bars"].sum()) if not gaps_df.empty else 0,
        "expected_freq_seconds": exp_sec,
        "start": start.tz_convert("UTC"),
        "end": end.tz_convert("UTC"),
        "timerange": timerange_str,
        "output_file": f"{symbol}__combined__{timerange_str}.csv",
        "notes": "",
    }

    return write_df, gaps_df, conflicts_df, summary

def write_outputs_for_symbol(
    symbol: str,
    combined: pd.DataFrame,
    gaps: pd.DataFrame,
    conflicts: pd.DataFrame,
    dirs: Dict[str, Path],
    summary_row: Dict,
    cfg: Config,
):
    # combined data
    out_name = summary_row["output_file"]
    out_path = dirs["combined"] / out_name
    if cfg.write_parquet:
        out_path = out_path.with_suffix(".parquet")
        combined.to_parquet(out_path, index=False)
    else:
        combined.to_csv(out_path, index=False)

    # gaps & conflicts per symbol (optional per-symbol files; we aggregate later anyway)
    gaps_path = dirs["reports"] / f"{symbol}__gaps.csv"
    conf_path = dirs["reports"] / f"{symbol}__conflicts.csv"
    if not gaps.empty:
        df = gaps.copy()
        df.insert(0, "symbol", symbol)
        df.to_csv(gaps_path, index=False)
    if not conflicts.empty:
        df = conflicts.copy()
        df.insert(0, "symbol", symbol)
        df.to_csv(conf_path, index=False)

def write_reports(summary_rows: List[Dict], all_gaps: List[pd.DataFrame], all_conflicts: List[pd.DataFrame], dirs: Dict[str, Path]):
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("symbol")
    summary_df.to_csv(dirs["reports"] / "summary.csv", index=False)

    if all_gaps:
        gaps_df = pd.concat(all_gaps, ignore_index=True)
        gaps_df.to_csv(dirs["reports"] / "gaps.csv", index=False)
    else:
        (dirs["reports"] / "gaps.csv").write_text("symbol,start,end,gap_seconds,missing_bars\n")

    if all_conflicts:
        conf_df = pd.concat(all_conflicts, ignore_index=True)
        conf_df.to_csv(dirs["reports"] / "conflicts.csv", index=False)
    else:
        (dirs["reports"] / "conflicts.csv").write_text("symbol,Datetime,kept_from,discarded_from,kept_mtime,discarded_mtime,kept_values,discarded_values\n")

    # Markdown summary
    md = []
    md.append("# Merge Report\n")
    md.append(f"- Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} UTC\n")
    md.append(f"- Symbols processed: **{len(summary_rows)}**\n")
    if len(summary_rows):
        total_in = sum(r.get("n_rows_in", 0) for r in summary_rows)
        total_out = sum(r.get("n_rows_out", 0) for r in summary_rows)
        total_conf = sum(r.get("conflicts", 0) for r in summary_rows)
        total_gaps = sum(r.get("gaps", 0) for r in summary_rows)
        total_missing = sum(r.get("missing_bars_total", 0) for r in summary_rows)
        md.append(f"- Total rows in: **{total_in:,}**  | Total rows out: **{total_out:,}**\n")
        md.append(f"- Conflicted timestamps (unique): **{total_conf:,}**\n")
        md.append(f"- Within-day gaps (segments): **{total_gaps:,}**, missing bars total: **{total_missing:,}**\n")
    md.append("\n## Per-symbol Summary\n")
    if len(summary_rows):
        md.append("| Symbol | Files | Rows In | Rows Out | Conflicts | Gaps | Missing Bars | Freq (s) | Start (UTC) | End (UTC) |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for r in sorted(summary_rows, key=lambda x: x["symbol"]):
            md.append(
                f"| {r['symbol']} | {r['n_files']} | {r['n_rows_in']:,} | {r['n_rows_out']:,} | "
                f"{r['conflicts']:,} | {r['gaps']:,} | {r['missing_bars_total']:,} | "
                f"{r['expected_freq_seconds']} | {r['start']} | {r['end']} |\n"
            )
    (dirs["reports"] / "REPORT.md").write_text("".join(md))

def main():
    ap = argparse.ArgumentParser(description="Combine overlapped financial CSV chunks into continuous per-symbol series with reports.")
    ap.add_argument("--input-root", type=str, required=True, help="Root dir containing many run folders; each run folder has per-symbol CSVs.")
    ap.add_argument("--output-dir", type=str, required=True, help="Where to write combined data and reports.")
    ap.add_argument("--tz-strategy", type=str, default="to_utc", choices=["keep", "to_utc", "localize"],
                    help="How to treat timestamp timezones before merging.")
    ap.add_argument("--assume-naive-tz", type=str, default="UTC", help="Timezone for naive timestamps if tz-strategy=localize.")
    ap.add_argument("--expected-freq-seconds", type=int, default=None,
                    help="If set, do not infer bar size; use this value (e.g., 60 for 1-minute).")
    ap.add_argument("--gap-factor", type=float, default=1.5, help="Flag within-day gaps when delta >= expected_freq * gap_factor.")
    ap.add_argument("--max-within-day-gap-hours", type=float, default=6.0, help="Ignore gaps longer than this (assume session break).")
    ap.add_argument("--write-parquet", action="store_true", help="Also (or instead) write Parquet; faster for huge data.")
    args = ap.parse_args()

    cfg = Config(
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir),
        tz_strategy=args.tz_strategy,
        assume_naive_tz=args.assume_naive_tz,
        expected_freq_seconds=args.expected_freq_seconds,
        gap_factor=args.gap_factor,
        max_within_day_gap_hours=args.max_within_day_gap_hours,
        write_parquet=args.write_parquet,
    )

    dirs = ensure_dirs(cfg.output_dir)
    logger = setup_logger(dirs["logs"])
    logger.info("Starting merge pipeline.")
    logger.info(f"Config: {cfg}")

    # discover files per symbol
    sym_files = list_all_symbol_files(cfg.input_root, logger)
    if not sym_files:
        logger.error("No CSVs found under input-root.")
        sys.exit(2)

    all_summ: List[Dict] = []
    all_gaps: List[pd.DataFrame] = []
    all_conf: List[pd.DataFrame] = []

    for sym, files in sorted(sym_files.items()):
        logger.info(f"[{sym}] processing {len(files)} files")
        combined, gaps, conflicts, summary = process_symbol(sym, files, cfg, logger)

        all_summ.append(summary)

        if combined is None:
            logger.warning(f"[{sym}] skipped (no readable input).")
            continue

        # write outputs
        write_outputs_for_symbol(sym, combined, gaps, conflicts, dirs, summary, cfg)

        if not gaps.empty:
            g = gaps.copy()
            g.insert(0, "symbol", sym)
            all_gaps.append(g)
        if not conflicts.empty:
            c = conflicts.copy()
            c.insert(0, "symbol", sym)
            all_conf.append(c)

    # aggregate reports
    write_reports(all_summ, all_gaps, all_conf, dirs)
    logger.info("Done. Reports written to: %s", dirs["reports"])

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True  # safer writes
    main()
