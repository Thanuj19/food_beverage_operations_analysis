# Food and Beverage Operations Analytics

A Linux-ready Python portfolio prototype that analyzes simulated line and shift data for a food and beverage operation. It uses Pandas to validate inputs, calculate production and quality KPIs, flag illustrative exceptions, and create a trend chart plus a Markdown review report. If `OPENAI_API_KEY` is available, the report also includes a concise OpenAI summary; otherwise it uses a local rule-based summary.

## Run on Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 food_ops_analytics.py --input sample_operations.csv --outdir output
```

To enable the optional GPT summary, provide `OPENAI_API_KEY` securely through your environment and run the same command. Set `OPENAI_MODEL` if you want to use another model available to your API account.

## Outputs

- `line_shift_kpis.csv`: output attainment, average downtime, defect rate, water use, and energy use by line and shift.
- `exceptions.csv`: line and shift groups that cross the prototype's illustrative thresholds.
- `production_trend.png`: daily actual output compared with plan.
- `report.md`: summary, data-quality notes, and threshold definitions.

The records are synthetic. Thresholds are demonstration settings, not validated plant specifications. This prototype is not for food-safety decisions, process control, or equipment operation. Do not send customer or confidential data to an external AI service.
