#!/usr/bin/env python3
"""Analyze simulated food-and-beverage operations data for a portfolio demo.

All thresholds are illustrative. This project is not a production or food-safety
system and must not be used to control equipment or make operational decisions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED = {
    "date", "shift", "line", "units_produced", "planned_units",
    "downtime_minutes", "defects", "water_liters", "energy_kwh",
}
NUMERIC = [
    "units_produced", "planned_units", "downtime_minutes", "defects",
    "water_liters", "energy_kwh",
]
ATTAINMENT_FLOOR = 0.90
DEFECT_RATE_CEILING = 0.02
DOWNTIME_CEILING_MINUTES = 60.0


def load_data(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    missing_columns = sorted(REQUIRED - set(frame.columns))
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    issues: list[str] = []
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        issues.append(f"{int(frame['date'].isna().sum())} row(s) have invalid dates.")
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        count = int(frame[column].isna().sum())
        if count:
            issues.append(f"{count} row(s) have missing or invalid {column} values.")

    valid = frame.dropna(subset=["date", *NUMERIC]).copy()
    if (valid["planned_units"] <= 0).any():
        issues.append("Rows with planned_units <= 0 were excluded from KPI calculations.")
        valid = valid[valid["planned_units"] > 0]
    if valid.empty:
        raise ValueError("No valid rows remain after data checks.")
    return valid, issues


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["line", "shift"], as_index=False).agg(
        units_produced=("units_produced", "sum"),
        planned_units=("planned_units", "sum"),
        downtime_minutes=("downtime_minutes", "mean"),
        defects=("defects", "sum"),
        water_liters=("water_liters", "sum"),
        energy_kwh=("energy_kwh", "sum"),
    )
    grouped["attainment_pct"] = 100 * grouped["units_produced"] / grouped["planned_units"]
    grouped["defect_rate_pct"] = 100 * grouped["defects"] / grouped["units_produced"].clip(lower=1)
    grouped["water_l_per_1000_units"] = 1000 * grouped["water_liters"] / grouped["units_produced"].clip(lower=1)
    grouped["energy_kwh_per_1000_units"] = 1000 * grouped["energy_kwh"] / grouped["units_produced"].clip(lower=1)
    return grouped.round(2)


def find_exceptions(kpis: pd.DataFrame) -> pd.DataFrame:
    findings = []
    for row in kpis.to_dict(orient="records"):
        flags = []
        if row["attainment_pct"] < ATTAINMENT_FLOOR * 100:
            flags.append("Output below illustrative 90% plan threshold")
        if row["defect_rate_pct"] > DEFECT_RATE_CEILING * 100:
            flags.append("Defect rate above illustrative 2% threshold")
        if row["downtime_minutes"] > DOWNTIME_CEILING_MINUTES:
            flags.append("Average downtime above illustrative 60-minute threshold")
        if flags:
            findings.append({"line": row["line"], "shift": row["shift"], "flags": "; ".join(flags)})
    return pd.DataFrame(findings, columns=["line", "shift", "flags"])


def create_chart(frame: pd.DataFrame, path: Path) -> None:
    daily = frame.groupby(["date", "line"], as_index=False).agg(
        units_produced=("units_produced", "sum"),
        planned_units=("planned_units", "sum"),
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for line, subset in daily.groupby("line"):
        ax.plot(subset["date"], subset["units_produced"], marker="o", label=f"{line}: actual")
        ax.plot(subset["date"], subset["planned_units"], linestyle="--", alpha=.65, label=f"{line}: plan")
    ax.set_title("Simulated daily output vs. plan")
    ax.set_ylabel("Units")
    ax.set_xlabel("Date")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def local_summary(kpis: pd.DataFrame, exceptions: pd.DataFrame, issues: list[str]) -> str:
    return (
        f"Analyzed {len(kpis)} line/shift groups from simulated data. "
        f"Flagged {len(exceptions)} group(s) for review and found {len(issues)} data-quality issue(s). "
        "Confirm all thresholds and interpretations with the process owner."
    )


def ai_summary(kpis: pd.DataFrame, exceptions: pd.DataFrame, issues: list[str]) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return local_summary(kpis, exceptions, issues), "local fallback"

    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    payload = {
        "model": model,
        "input": (
            "Summarize this simulated food-and-beverage operations analysis for a technical "
            "deployment review. State key KPI patterns, flagged line/shift groups, and data "
            "quality issues. Do not infer root causes or give operational instructions. "
            "Clearly label the data and thresholds as illustrative.\n\n"
            + json.dumps({
                "kpis": kpis.to_dict(orient="records"),
                "exceptions": exceptions.to_dict(orient="records"),
                "data_quality_issues": issues,
            }, default=str)
        ),
        "max_output_tokens": 220,
    }
    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"AI summary request failed: {exc}") from exc

    text = body.get("output_text", "").strip()
    if not text:
        text = "\n".join(
            content.get("text", "")
            for item in body.get("output", [])
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        ).strip()
    if not text:
        raise RuntimeError("The AI API returned no summary text.")
    return text, f"OpenAI Responses API ({model})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze simulated operations data and produce a review pack.")
    parser.add_argument("--input", type=Path, default=Path("sample_operations.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    args = parser.parse_args()
    try:
        data, issues = load_data(args.input)
        kpis = summarize(data)
        exceptions = find_exceptions(kpis)
        args.outdir.mkdir(parents=True, exist_ok=True)
        kpis.to_csv(args.outdir / "line_shift_kpis.csv", index=False)
        exceptions.to_csv(args.outdir / "exceptions.csv", index=False)
        create_chart(data, args.outdir / "production_trend.png")
        summary, summary_source = ai_summary(kpis, exceptions, issues)
        quality_lines = [f"- {issue}" for issue in issues] if issues else ["- No missing or invalid required values were detected."]
        report = [
            "# Operations Analytics Review",
            "",
            "> SIMULATED DATA: illustrative thresholds and figures only; not for operational decisions.",
            "",
            f"**Summary source:** {summary_source}",
            "",
            "## AI or rule-based summary",
            summary,
            "",
            "## Data quality",
            *quality_lines,
            "",
            "## Illustrative thresholds",
            f"- Output attainment below {ATTAINMENT_FLOOR:.0%}",
            f"- Defect rate above {DEFECT_RATE_CEILING:.0%}",
            f"- Average downtime above {DOWNTIME_CEILING_MINUTES:.0f} minutes",
            "",
            "See `line_shift_kpis.csv`, `exceptions.csv`, and `production_trend.png` for details.",
        ]
        (args.outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(summary)
        print(f"Review pack written to {args.outdir.resolve()}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
