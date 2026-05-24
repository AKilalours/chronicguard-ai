"""
ChronicGuard AI — Patient Risk Timeline
Tracks a simulated patient's risk score over time across multiple messages.
Detects deterioration trends — a patient whose risk is escalating over
multiple contacts may need proactive outreach before the next crisis.

Usage:
    from src.risk_timeline import RiskTimelineTracker
    tracker = RiskTimelineTracker()
    tracker.add_message("PT_001", message, risk_level, intent)
    trend = tracker.get_trend("PT_001")
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import numpy as np

RISK_SCORES = {"low": 1, "medium": 2, "high": 3, "urgent": 4}
RISK_LABELS = {1: "low", 2: "medium", 3: "high", 4: "urgent"}

SIMULATED_PATIENTS = {
    "PT_001": {
        "name": "Patient A — Deteriorating",
        "condition": "Hypertension + Diabetes",
        "messages": [
            ("I missed my blood pressure medication today", "medication_question", "low",   0),
            ("My BP has been a bit high this week",         "symptom_escalation",  "medium", 2),
            ("Headache and dizziness for two days",          "symptom_escalation",  "high",   5),
            ("BP reading 175/105 and I feel terrible",       "symptom_escalation",  "high",   7),
            ("Chest tightness and can't catch my breath",    "symptom_escalation",  "urgent", 9),
        ],
    },
    "PT_002": {
        "name": "Patient B — Improving",
        "condition": "Diabetes",
        "messages": [
            ("Blood sugar over 300 for three days",         "symptom_escalation",  "urgent", 0),
            ("Started new insulin, sugar still high",       "medication_question", "high",   3),
            ("Sugar down to 220, feeling better",           "lab_results",         "medium", 6),
            ("A1C came back 7.2, doctor is happy",          "lab_results",         "low",    10),
            ("Can I schedule my quarterly check-in?",       "appointment_admin",   "low",    14),
        ],
    },
    "PT_003": {
        "name": "Patient C — Crisis Risk",
        "condition": "CHF + Depression",
        "messages": [
            ("Can I reschedule my appointment",              "appointment_admin",   "low",    0),
            ("Feeling very tired and out of breath",         "symptom_escalation",  "medium", 4),
            ("Gained 5 pounds this week, ankles swollen",    "symptom_escalation",  "high",   6),
            ("I feel hopeless, nothing is getting better",   "crisis",              "high",   8),
            ("I don't want to be here anymore",              "crisis",              "urgent", 10),
        ],
    },
}


@dataclass
class TimelinePoint:
    timestamp: float
    days_offset: int
    message: str
    intent: str
    risk_level: str
    risk_score: int
    requires_review: bool


@dataclass
class PatientTimeline:
    patient_id: str
    name: str
    condition: str
    points: list[TimelinePoint] = field(default_factory=list)

    @property
    def trend(self) -> str:
        if len(self.points) < 2:
            return "stable"
        scores = [p.risk_score for p in self.points[-3:]]
        if scores[-1] > scores[0]:
            return "deteriorating"
        elif scores[-1] < scores[0]:
            return "improving"
        return "stable"

    @property
    def trend_slope(self) -> float:
        if len(self.points) < 2:
            return 0.0
        scores = [p.risk_score for p in self.points]
        days = [p.days_offset for p in self.points]
        if len(set(days)) < 2:
            return 0.0
        return float(np.polyfit(days, scores, 1)[0])

    @property
    def current_risk(self) -> str:
        return self.points[-1].risk_level if self.points else "unknown"

    @property
    def peak_risk(self) -> str:
        if not self.points:
            return "unknown"
        return RISK_LABELS[max(p.risk_score for p in self.points)]

    @property
    def proactive_outreach_recommended(self) -> bool:
        return self.trend == "deteriorating" and self.trend_slope > 0.2

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "condition": self.condition,
            "current_risk": self.current_risk,
            "peak_risk": self.peak_risk,
            "trend": self.trend,
            "trend_slope": round(self.trend_slope, 3),
            "proactive_outreach_recommended": self.proactive_outreach_recommended,
            "n_contacts": len(self.points),
            "points": [
                {
                    "days_offset": p.days_offset,
                    "message": p.message,
                    "intent": p.intent,
                    "risk_level": p.risk_level,
                    "risk_score": p.risk_score,
                    "requires_review": p.requires_review,
                }
                for p in self.points
            ],
        }


class RiskTimelineTracker:
    """Tracks patient risk scores over time and detects deterioration trends."""

    def __init__(self):
        self._timelines: dict[str, PatientTimeline] = {}

    def add_message(
        self,
        patient_id: str,
        message: str,
        risk_level: str,
        intent: str,
        days_offset: int | None = None,
        patient_name: str = "Unknown",
        condition: str = "Chronic condition",
    ):
        if patient_id not in self._timelines:
            self._timelines[patient_id] = PatientTimeline(
                patient_id=patient_id,
                name=patient_name,
                condition=condition,
            )
        timeline = self._timelines[patient_id]
        offset = days_offset if days_offset is not None else len(timeline.points) * 2
        point = TimelinePoint(
            timestamp=time.time(),
            days_offset=offset,
            message=message,
            intent=intent,
            risk_level=risk_level,
            risk_score=RISK_SCORES.get(risk_level, 1),
            requires_review=risk_level in ("high", "urgent") or intent == "crisis",
        )
        timeline.points.append(point)

    def get_trend(self, patient_id: str) -> PatientTimeline | None:
        return self._timelines.get(patient_id)

    def get_all_timelines(self) -> list[PatientTimeline]:
        return list(self._timelines.values())

    def get_proactive_outreach_list(self) -> list[PatientTimeline]:
        return [t for t in self._timelines.values() if t.proactive_outreach_recommended]

    def load_simulated_patients(self):
        """Load the three simulated patient timelines."""
        for pid, data in SIMULATED_PATIENTS.items():
            for msg, intent, risk, day in data["messages"]:
                self.add_message(
                    patient_id=pid,
                    message=msg,
                    risk_level=risk,
                    intent=intent,
                    days_offset=day,
                    patient_name=data["name"],
                    condition=data["condition"],
                )

    def to_dict(self) -> dict:
        return {
            "n_patients": len(self._timelines),
            "proactive_outreach_needed": len(self.get_proactive_outreach_list()),
            "timelines": [t.to_dict() for t in self._timelines.values()],
        }


if __name__ == "__main__":
    tracker = RiskTimelineTracker()
    tracker.load_simulated_patients()

    print("Patient Risk Timeline Analysis")
    print("=" * 55)

    for timeline in tracker.get_all_timelines():
        print(f"\n{timeline.name} ({timeline.condition})")
        print(f"  Trend:    {timeline.trend.upper()}")
        print(f"  Current:  {timeline.current_risk}")
        print(f"  Peak:     {timeline.peak_risk}")
        print(f"  Slope:    {timeline.trend_slope:+.3f} risk units/day")
        print(f"  Outreach: {'RECOMMENDED' if timeline.proactive_outreach_recommended else 'Not needed'}")
        for pt in timeline.points:
            marker = "!" if pt.requires_review else " "
            print(f"  Day {pt.days_offset:>2} {marker} [{pt.risk_level:>6}] {pt.message[:50]}")

    outreach = tracker.get_proactive_outreach_list()
    print(f"\nProactive outreach recommended: {len(outreach)} patient(s)")
    for t in outreach:
        print(f"  {t.patient_id} — {t.name} (slope: {t.trend_slope:+.3f})")

    Path("results").mkdir(exist_ok=True)
    with open("results/risk_timelines.json", "w") as f:
        json.dump(tracker.to_dict(), f, indent=2)
    print(f"\nSaved → results/risk_timelines.json")
