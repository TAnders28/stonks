"""Print this week's tradeable momentum picks for ~$115 account (read-only)."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
# default to latest mt_ run
runs = sorted((ROOT/"out").glob("mt_*"))
if not runs:
    raise SystemExit("no mt_ runs")
pred = json.loads((runs[-1]/"predict_week.json").read_text())
print(json.dumps(pred, indent=2))
