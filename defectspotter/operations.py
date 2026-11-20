"""Real-world view of an operating point: what one production shift looks like in parts, people and money.

Rates measured on the test split (false-reject rate, escape rate, share sent to CHECK, ...) are turned into counts for a
realistic shift. The default scenario is documented with sources; every value is editable in the app.

Default scenario (one inspection station):
* line speed 2,500 parts/hour, 8-hour shift -> 20,000 parts per shift; 1% defective (a typical first-pass yield).
* a human check of a flagged part takes 36 s (pick the part, look at it with the heat map, decide);
* inspector cost 30 USD/hour fully loaded: BLS median pay for quality-control inspectors is 22.82 USD/hour (2024),
  plus ~30% for benefits and overhead  ->  36 s x 30 USD/h = 0.30 USD per check (the value used for calibration);
* inspectors catch ~80% of the defects they are shown (industry-average visual-inspection hit rate; See 2015);
* a false reject costs the value of the scrapped/reworked part (2 USD: a mid-value part);
* an escape costs 400 USD: customer complaint handling, return, replacement and goodwill. The 1-10-100 rule of cost of
  quality (Labovitz, Chang & Rosansky 1992) says a defect costs ~100x more once it reaches the customer than at the
  source; its common extension puts a customer complaint at ~1,000x the inspection cost (0.30 USD -> ~300 USD).

Sources: https://www.bls.gov/ooh/production/quality-control-inspectors.htm ·
https://www.osti.gov/biblio/1236203 (See, "Visual inspection reliability for precision manufactured parts", Human
Factors 2015) · Labovitz, Chang & Rosansky, "Making Quality Work" (1992).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SOURCES = {
    "inspector wage": ("BLS Occupational Outlook Handbook — quality control inspectors, median pay 22.82 USD/hour (2024)",
                       "https://www.bls.gov/ooh/production/quality-control-inspectors.htm"),
    "human hit rate": ("See (2015), Visual inspection reliability for precision manufactured parts, Human Factors — "
                       "industry-average hit rate ~80%", "https://www.osti.gov/biblio/1236203"),
    "cost escalation": ("1-10-100 rule of cost of quality — Labovitz, Chang & Rosansky, Making Quality Work (1992)", ""),
}


@dataclass
class LineScenario:
    parts_per_hour: float = 2500.0
    shift_hours: float = 8.0
    defect_rate: float = 0.01
    seconds_per_check: float = 36.0
    inspector_cost_per_hour: float = 30.0
    human_detection_rate: float = 0.80
    cost_false_reject: float = 2.0
    cost_escape: float = 400.0
    shifts_per_year: int = 250

    @property
    def parts_per_shift(self) -> float:
        return self.parts_per_hour * self.shift_hours

    @property
    def cost_per_check(self) -> float:
        return self.inspector_cost_per_hour * self.seconds_per_check / 3600.0

    @property
    def checks_per_inspector_hour(self) -> float:
        return 3600.0 / self.seconds_per_check

    def to_dict(self) -> dict:
        return {**asdict(self), "parts_per_shift": self.parts_per_shift, "cost_per_check": self.cost_per_check}


def rates_from_scores(scores, labels, threshold: float, band: float) -> dict:
    """PASS / CHECK / REJECT shares per class at a threshold (labels: 0 = good, 1 = defective)."""
    s, y = np.asarray(scores, np.float64), np.asarray(labels).astype(int)
    good, bad = s[y == 0], s[y == 1]
    lo, hi = threshold - band, threshold + band
    share = lambda x, m: float(np.mean(m)) if len(x) else float("nan")  # noqa: E731
    return {"false_reject_rate": share(good, good >= hi), "good_check_rate": share(good, (good >= lo) & (good < hi)),
            "escape_rate": share(bad, bad < lo), "defect_check_rate": share(bad, (bad >= lo) & (bad < hi)),
            "defect_reject_rate": share(bad, bad >= hi)}


def shift_view(rates: dict, scenario: LineScenario | None = None) -> dict:
    """Counts for one shift: false rejects, parts sent to a human, inspector hours, defects caught / escaping, cost."""
    sc = scenario or LineScenario()
    n = sc.parts_per_shift
    good, defects = n * (1 - sc.defect_rate), n * sc.defect_rate
    r = {k: float(np.nan_to_num(rates.get(k, 0.0))) for k in
         ("false_reject_rate", "good_check_rate", "escape_rate", "defect_check_rate", "defect_reject_rate")}
    to_human = good * r["good_check_rate"] + defects * r["defect_check_rate"]
    caught_by_human = defects * r["defect_check_rate"] * sc.human_detection_rate
    escaping = defects * r["escape_rate"] + defects * r["defect_check_rate"] * (1 - sc.human_detection_rate)
    false_rejects = good * r["false_reject_rate"]
    hours = to_human * sc.seconds_per_check / 3600.0
    cost = false_rejects * sc.cost_false_reject + escaping * sc.cost_escape + to_human * sc.cost_per_check
    return {"parts": n, "defective": defects, "false_rejects": false_rejects, "to_human": to_human,
            "share_to_human": to_human / n if n else float("nan"), "inspector_hours": hours,
            "inspectors_needed": hours / sc.shift_hours, "defects_auto_rejected": defects * r["defect_reject_rate"],
            "defects_caught_by_human": caught_by_human, "defects_escaping": escaping,
            "cost_per_shift": cost, "cost_per_year": cost * sc.shifts_per_year}


def manual_inspection_view(scenario: LineScenario | None = None) -> dict:
    """The status quo: a human looks at every part."""
    sc = scenario or LineScenario()
    n, defects = sc.parts_per_shift, sc.parts_per_shift * sc.defect_rate
    hours = n * sc.seconds_per_check / 3600.0
    escaping = defects * (1 - sc.human_detection_rate)
    cost = escaping * sc.cost_escape + n * sc.cost_per_check
    return {"parts": n, "defective": defects, "false_rejects": 0.0, "to_human": n, "share_to_human": 1.0,
            "inspector_hours": hours, "inspectors_needed": hours / sc.shift_hours, "defects_auto_rejected": 0.0,
            "defects_caught_by_human": defects * sc.human_detection_rate, "defects_escaping": escaping,
            "cost_per_shift": cost, "cost_per_year": cost * sc.shifts_per_year}


def shift_table(rows: list[tuple[str, str, dict]], scenario: LineScenario | None = None) -> pd.DataFrame:
    """[(product, model, rates), ...] -> a readable one-shift table (plus the 100% manual reference)."""
    sc = scenario or LineScenario()

    def row(product: str, model: str, v: dict) -> dict:
        return {"product": product, "model": model,
                "parts to a human": f"{v['to_human']:,.0f} ({100 * v['share_to_human']:.1f}%)",
                "inspectors needed": round(v["inspectors_needed"], 1),
                "defects escaping": round(v["defects_escaping"], 1),
                "good parts scrapped": f"{v['false_rejects']:,.0f}",
                "defects caught": f"{v['defects_auto_rejected'] + v['defects_caught_by_human']:,.0f} of {v['defective']:,.0f}",
                "cost per shift (USD)": f"{v['cost_per_shift']:,.0f}", "cost per year (USD)": f"{v['cost_per_year']:,.0f}"}

    out = [row(product, model, shift_view(rates, sc)) for product, model, rates in rows]
    out.append(row("any", "100% manual inspection (status quo)", manual_inspection_view(sc)))
    return pd.DataFrame(out)
