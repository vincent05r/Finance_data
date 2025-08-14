#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re
import sys
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd

RE_COMBINED = re.compile(
    r"^(?P<sym>.+)__combined__(?P<start>\d{8}T\d{6}Z)__to__(?P<end>\d{8}T\d{6}Z)\.csv$"
)

REQUIRED_COLS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("validate_merge")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    fh = RotatingFileHandler(log_dir / "validate.log", maxBytes=3_000_000, backupCount=2)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

def load_reports(reports_dir: Path, logger: logging.Logger):
    summary_fp = reports_dir / "summary.csv"
    gaps_fp = reports_dir / "gaps.csv"
    conflicts_fp = reports_dir / "conflicts.csv"

    if not summary_fp.exists():
        raise FileNotFoundError(f"Missing {summary_fp}")
    summary = pd.read_csv(summary_fp)

    gaps = pd.DataFrame(columns=["symbol","start","end","gap_seconds","missing_bars"])
    if gaps_fp.exists() and gaps_fp.stat().st_size > 0:
        try:
            gaps = pd.read_csv(gaps_fp)
        except Exception as e:
            logger.warning(f"Failed to read gaps.csv ({e}); treating as empty.")

    conflicts = pd.DataFrame(columns=["symbol","Datetime"])
    if conflicts_fp.exists() and conflicts_fp.stat().st_size > 0:
        try:
            conflicts = pd.read_csv(conflicts_fp)
        except Exception as e:
            logger.warning(f"Failed to read conflicts.csv ({e}); treating as empty.")

    # Normalize datetime-like columns
    for c in ["start","end"]:
        if c in gaps.columns:
            gaps[c] = pd.to_datetime(gaps[c], utc=True, errors="coerce")
    if "Datetime" in conflicts.columns:
        conflicts["Datetime"] = pd.to_datetime(conflicts["Datetime"], utc=True, errors="coerce")

    return summary, gaps, conflicts

def coerce_dt_utc(s: pd.Series) -> pd.DatetimeIndex:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    idx = pd.DatetimeIndex(dt)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx

def infer_modal_delta_seconds(idx: pd.DatetimeIndex) -> int:
    if len(idx) < 3:
        return 60
    idx = idx.sort_values()
    deltas = np.diff(idx.view("i8")) / 1e9
    dates = idx.date
    same_day = (dates[1:] == dates[:-1])
    small = (deltas > 0) & (deltas <= 3600) & same_day
    if np.any(small):
        vals = deltas[small].astype(int)
        u, c = np.unique(vals, return_counts=True)
        return int(u[np.argmax(c)])
    return 60

def detect_within_day_gaps(idx: pd.DatetimeIndex, expected_sec: int, gap_factor=1.5, max_within_day_gap_hours=6.0) -> pd.DataFrame:
    if len(idx) < 2:
        return pd.DataFrame(columns=["start","end","gap_seconds","missing_bars"])
    idx = idx.sort_values()
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    deltas = np.diff(idx.view("i8")) / 1e9
    dates = idx.date
    same_day = (dates[1:] == dates[:-1])
    min_gap = expected_sec * gap_factor
    max_gap = max_within_day_gap_hours * 3600.0
    mask = same_day & (deltas >= min_gap) & (deltas < max_gap)
    if not np.any(mask):
        return pd.DataFrame(columns=["start","end","gap_seconds","missing_bars"])
    i = np.where(mask)[0]
    starts = idx[i]
    ends = idx[i+1]
    gaps = (ends.view("i8") - starts.view("i8")) / 1e9
    missing = np.maximum((gaps / expected_sec) - 1.0, 0.0).round().astype(int)
    return pd.DataFrame({"start":starts, "end":ends, "gap_seconds":gaps.astype(int), "missing_bars":missing})

def check_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    ok = all(c in df.columns for c in REQUIRED_COLS)
    if not ok:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        return False, f"Missing columns: {missing}"
    return True, ""

def check_ohlc_sanity(df: pd.DataFrame) -> List[str]:
    issues = []
    # Coerce numerics; don't crash on bad rows
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad_nan = df[["Open","High","Low","Close"]].isna().any(axis=1).sum()
    if bad_nan > 0:
        issues.append(f"NaN in OHLC rows: {bad_nan}")
    # Price relations
    mask_low = df["Low"] > df[["Open","Close"]].min(axis=1)
    mask_high = df["High"] < df[["Open","Close"]].max(axis=1)
    if mask_low.any():
        issues.append(f"Low greater than min(Open,Close): {int(mask_low.sum())} rows")
    if mask_high.any():
        issues.append(f"High less than max(Open,Close): {int(mask_high.sum())} rows")
    # Volume
    vol_neg = (df["Volume"] < 0).sum(skipna=True)
    if vol_neg > 0:
        issues.append(f"Negative volume rows: {int(vol_neg)}")
    # Integer-like volume check (allow NaN)
    v = df["Volume"].dropna().to_numpy()
    if v.size:
        if not np.allclose(v, np.round(v)):
            issues.append("Volume has non-integer values")
    return issues

def main():
    ap = argparse.ArgumentParser(description="Validate merged financial outputs.")
    ap.add_argument("--output-dir", type=str, required=True, help="Path that contains combined/ and reports/")
    ap.add_argument("--fail-on", type=str, default="any", choices=["none","schema","gaps_mismatch","any"], help="Failure sensitivity.")
    ap.add_argument("--sample-rows", type=int, default=0, help="Optional: print first N rows of first failing symbol.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    combined_dir = out_dir / "combined"
    reports_dir = out_dir / "reports"
    val_dir = out_dir / "validation"
    log_dir = out_dir / "logs"

    val_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir)

    summary, gaps_report, conflicts_report = load_reports(reports_dir, logger)
    summary["start"] = pd.to_datetime(summary["start"], utc=True, errors="coerce")
    summary["end"]   = pd.to_datetime(summary["end"],   utc=True, errors="coerce")

    # Build map: symbol -> expected_freq_seconds (from summary; fallback to infer)
    exp_freq_by_sym: Dict[str, int] = {}
    for _, r in summary.iterrows():
        sym = str(r["symbol"])
        v = r.get("expected_freq_seconds")
        if pd.notna(v):
            try:
                exp_freq_by_sym[sym] = int(v)
            except Exception:
                pass

    # Validate each combined file
    issues_rows: List[Dict] = []
    combined_files = sorted([p for p in combined_dir.glob("*.csv") if RE_COMBINED.match(p.name)])
    if not combined_files:
        logger.error(f"No combined CSVs found in {combined_dir}")
        sys.exit(2)

    for f in combined_files:
        m = RE_COMBINED.match(f.name)
        sym = m.group("sym")
        start_str = m.group("start")
        end_str = m.group("end")
        try:
            df = pd.read_csv(f)
        except Exception as e:
            issues_rows.append({"symbol": sym, "file": f.name, "level":"ERROR", "issue": f"Cannot read file: {e}"})
            continue

        ok, msg = check_schema(df)
        if not ok:
            issues_rows.append({"symbol": sym, "file": f.name, "level":"ERROR", "issue": msg})
            if args.fail_on in ("schema","any"):
                continue

        # Datetime normalization
        try:
            idx = coerce_dt_utc(df["Datetime"])
        except Exception as e:
            issues_rows.append({"symbol": sym, "file": f.name, "level":"ERROR", "issue": f"Datetime parse failure: {e}"})
            continue

        # Monotonicity & duplicates
        if not idx.is_monotonic_increasing:
            issues_rows.append({"symbol": sym, "file": f.name, "level":"ERROR", "issue": "Timestamps not strictly increasing"})
        dup = pd.Index(idx).duplicated().sum()
        if dup > 0:
            issues_rows.append({"symbol": sym, "file": f.name, "level":"ERROR", "issue": f"Duplicate timestamps: {dup}"})

        # Filename range vs actual bounds
        file_start = pd.to_datetime(start_str, format="%Y%m%dT%H%M%SZ", utc=True)
        file_end   = pd.to_datetime(end_str,   format="%Y%m%dT%H%M%SZ", utc=True)
        actual_start = idx.min()
        actual_end   = idx.max()
        if actual_start != file_start or actual_end != file_end:
            issues_rows.append({
                "symbol": sym, "file": f.name, "level":"WARN",
                "issue": f"File timerange mismatch: name({file_start}→{file_end}) vs data({actual_start}→{actual_end})"
            })

        # OHLC/Volume sanity
        ohlc_issues = check_ohlc_sanity(df)
        for it in ohlc_issues:
            issues_rows.append({"symbol": sym, "file": f.name, "level":"WARN", "issue": it})

        # Frequency: expected vs inferred
        expected = exp_freq_by_sym.get(sym, None)
        inferred = infer_modal_delta_seconds(idx)
        if expected is None:
            expected = inferred  # fallback for comparisons below
        else:
            if abs(int(expected) - int(inferred)) > max(1, int(0.1 * expected)):
                issues_rows.append({
                    "symbol": sym, "file": f.name, "level":"WARN",
                    "issue": f"Inferred freq {inferred}s differs from reported {expected}s"
                })

        # Recompute gaps and compare to gaps.csv rows for this symbol
        local_gaps = detect_within_day_gaps(idx, expected_sec=int(expected))
        local_gaps["symbol"] = sym

        if not gaps_report.empty:
            rep = gaps_report[gaps_report["symbol"] == sym].copy()
        else:
            rep = pd.DataFrame(columns=["symbol","start","end","gap_seconds","missing_bars"])

        # Normalize both sides for robust comparison (inner join on start/end)
        if not rep.empty or not local_gaps.empty:
            rep["start"] = pd.to_datetime(rep["start"], utc=True, errors="coerce")
            rep["end"]   = pd.to_datetime(rep["end"],   utc=True, errors="coerce")
            merged = pd.merge(
                local_gaps,
                rep[["start","end","gap_seconds","missing_bars"]],
                on=["start","end"], how="outer", suffixes=("_local","_report")
            )
            # mismatches: present in one side only or different gap_seconds by > expected
            only_local = merged["gap_seconds_report"].isna().sum()
            only_report = merged["gap_seconds_local"].isna().sum()
            diff = merged.dropna(subset=["gap_seconds_local","gap_seconds_report"])
            gap_diff = (diff["gap_seconds_local"] != diff["gap_seconds_report"]).sum()
            if only_local or only_report or gap_diff:
                issues_rows.append({
                    "symbol": sym, "file": f.name, "level": "WARN" if args.fail_on != "gaps_mismatch" else "ERROR",
                    "issue": f"Gaps mismatch: only_local={int(only_local)}, only_report={int(only_report)}, differing={int(gap_diff)}"
                })

        # Conflicts sanity: we cannot re-derive conflict count from the combined file,
        # but we can warn if summary says conflicts>0 while per-symbol conflicts file missing.
        # (Soft check; not an error.)
        # Nothing to do here beyond the warning.

        # Optional: sample rows on first error
        # (Printed at the end; we don't print mid-loop to keep logs clean.)

    # Write issues report
    issues_df = pd.DataFrame(issues_rows, columns=["symbol","file","level","issue"]).sort_values(["level","symbol","file"])
    issues_fp = val_dir / "issues.csv"
    issues_df.to_csv(issues_fp, index=False)

    # Write summary markdown
    md = []
    md.append("# Validation Summary\n\n")
    md.append(f"- Combined files checked: **{len(list(combined_dir.glob('*.csv')))}**\n")
    md.append(f"- Issues found: **{len(issues_df)}** (levels: ERROR/WARN)\n")
    levels = issues_df["level"].value_counts() if not issues_df.empty else {}
    if len(levels):
        for lvl, cnt in levels.items():
            md.append(f"  - {lvl}: **{int(cnt)}**\n")
    md.append("\n## Top 20 issues\n\n")
    if not issues_df.empty:
        head = issues_df.head(20)
        for _, r in head.iterrows():
            md.append(f"- [{r['level']}] {r['symbol']} / {r['file']}: {r['issue']}\n")
    else:
        md.append("_No issues detected._\n")
    (val_dir / "summary.md").write_text("".join(md))

    # Print quick console summary
    if not issues_df.empty and args.sample_rows > 0:
        first_err = issues_df[issues_df["level"] == "ERROR"].head(1)
        if not first_err.empty:
            fe = first_err.iloc[0]
            print("\n--- Sample rows from first error file ---")
            try:
                df = pd.read_csv(combined_dir / fe["file"])
                print(df.head(args.sample_rows).to_string(index=False))
            except Exception:
                pass

    logger.info("Validation issues written to: %s", issues_fp)
    logger.info("Summary written to: %s", val_dir / "summary.md")

    # Exit code policy
    rc = 0
    if args.fail_on == "any" and (issues_df["level"] == "ERROR").any():
        rc = 1
    elif args.fail_on == "schema" and (issues_df["issue"].str.contains("Missing columns", na=False)).any():
        rc = 1
    elif args.fail_on == "gaps_mismatch" and ((issues_df["level"] == "ERROR") & (issues_df["issue"].str.contains("Gaps mismatch", na=False))).any():
        rc = 1
    sys.exit(rc)

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
