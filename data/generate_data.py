"""
ChronicGuard AI — Synthetic Data Generator
Generates labeled patient messages across 6 intent categories and 4 risk levels.
All data is fully synthetic; no PHI involved.
"""

import csv
import random
import json
from pathlib import Path

random.seed(42)

# ── Label definitions ──────────────────────────────────────────────────────────
INTENTS = [
    "medication_question",
    "symptom_escalation",
    "appointment_admin",
    "lab_results",
    "care_gap",
    "crisis",
]

RISK_LEVELS = ["low", "medium", "high", "urgent"]

# ── Message templates per intent ───────────────────────────────────────────────
TEMPLATES = {
    "medication_question": {
        "low": [
            "Can I take my metformin with food or does it need to be on an empty stomach?",
            "Is it okay to take my blood pressure pill at night instead of the morning?",
            "I forgot if I took my lisinopril today. Should I take it now?",
            "My pharmacist gave me a generic version of my statin. Is that the same thing?",
            "Can I take ibuprofen while I'm on my blood thinner?",
            "How long does it take for metformin to start working?",
            "Should I stop my cholesterol medication before my blood draw tomorrow?",
            "Is it safe to drink alcohol occasionally while on my diabetes medication?",
        ],
        "medium": [
            "I've been forgetting to take my blood pressure medication for the past week. What should I do?",
            "I ran out of my insulin refill and the pharmacy says it'll be 3 days. What do I do?",
            "I accidentally took double my metformin dose this morning. Is that dangerous?",
            "My doctor changed my blood pressure medication but I'm not sure I'm taking the right dose.",
            "I stopped taking my cholesterol medication because it was making my legs hurt. Should I restart?",
            "I've been splitting my lisinopril in half because the pills are expensive. Is that okay?",
        ],
        "high": [
            "I forgot to take my blood pressure medicine for two days. Should I double the dose?",
            "I've been skipping my insulin because I can't afford the copay. My sugars have been really high.",
            "I took someone else's blood pressure medication by accident. It's a different drug than mine.",
            "I've been feeling dizzy and I think it might be from my new medication. Should I stop taking it?",
        ],
        "urgent": [
            "I took too many of my heart pills by accident. I'm not sure how many.",
            "I think I'm having a reaction to my new medication. My throat feels tight.",
        ],
    },
    "symptom_escalation": {
        "low": [
            "I've had a mild headache for two days. Could it be related to my blood pressure?",
            "My ankles have been slightly swollen this week. Is this normal with my medication?",
            "I've been feeling more tired than usual lately. Is that related to my diabetes?",
            "I have a slight cough that started when I began my new medication.",
            "I've been feeling a little lightheaded when I stand up quickly.",
        ],
        "medium": [
            "My blood sugar readings have been running higher than usual for the past few days.",
            "I've had a headache every day this week and my blood pressure monitor shows high readings.",
            "My feet have been swelling more than usual and my shoes feel tight.",
            "I've been short of breath when climbing stairs but it goes away when I rest.",
            "I've gained about 5 pounds in the last week and I'm not sure why.",
            "My blood pressure has been reading 155/95 consistently at home.",
        ],
        "high": [
            "I have chest tightness and shortness of breath today. It started this morning.",
            "My blood sugar has been over 300 for two days and I feel very thirsty and tired.",
            "I've had a severe headache for 24 hours and my blood pressure reading was 180/110.",
            "I'm having vision changes and headache — my blood pressure is very high today.",
            "My heart has been racing and I've been dizzy since yesterday afternoon.",
        ],
        "urgent": [
            "I'm having chest pain that spreads to my left arm. Started about an hour ago.",
            "I can't catch my breath and I feel pressure on my chest. It's been getting worse.",
            "I'm having trouble speaking and my face feels numb on one side.",
            "I feel like I might pass out. My heart is pounding and I'm sweating a lot.",
            "My blood sugar is 42 and I've had juice but it's not coming up.",
        ],
    },
    "appointment_admin": {
        "low": [
            "Can I reschedule my diabetes follow-up appointment next Tuesday?",
            "What do I need to bring to my next appointment?",
            "Do I need to fast before my appointment tomorrow?",
            "Can I get a referral to a dietitian through the care program?",
            "How do I get a copy of my medical records?",
            "Is there a nurse line I can call after hours?",
            "Can I do my next check-in by phone instead of coming in?",
        ],
        "medium": [
            "I haven't been able to get an appointment with my cardiologist for 3 months. Who can help?",
            "My doctor retired and I haven't been assigned a new one yet. I need a refill.",
            "I missed my last two appointments due to transportation issues. Can someone help me reschedule?",
        ],
        "high": [],
        "urgent": [],
    },
    "lab_results": {
        "low": [
            "I got a notification my lab results are ready but I can't figure out how to view them.",
            "My A1C result came back as 6.9. Is that good or bad?",
            "What does it mean that my LDL is 112?",
            "My eGFR was 68 — my doctor mentioned it but didn't explain what it means.",
        ],
        "medium": [
            "No one called me after my lab results came in last week. My A1C was 8.2.",
            "My potassium level came back low and I haven't heard from anyone about what to do.",
            "My creatinine has been trending up over the last three tests. Should I be worried?",
            "My thyroid levels are off according to my results but I haven't heard from my doctor.",
        ],
        "high": [
            "My lab results show my kidney function is worse. My eGFR dropped from 55 to 38.",
            "I got a flagged result for high potassium and no one has called me. I take a potassium-sparing diuretic.",
        ],
        "urgent": [
            "My glucose was 512 on my lab work and I feel terrible. No one has called me.",
        ],
    },
    "care_gap": {
        "low": [
            "I haven't had an eye exam in two years. Is that something the program can help with?",
            "I know I'm due for a foot exam but I haven't scheduled one yet.",
            "My doctor mentioned I should get a flu shot. Can the care team remind me?",
            "I haven't done my annual blood work yet this year.",
        ],
        "medium": [
            "My pharmacy says my refill was denied and I'm almost out of my blood pressure medication.",
            "I've missed the last three months of medication refills for my diabetes. I ran out.",
            "I haven't seen a specialist in over a year even though my doctor referred me.",
            "I was supposed to start physical therapy but nobody followed up with authorization.",
        ],
        "high": [
            "I've been out of my blood thinner for five days. The prior authorization was denied.",
            "I have end-stage kidney disease and I missed my last two dialysis sessions.",
        ],
        "urgent": [],
    },
    "crisis": {
        "low": [],
        "medium": [
            "I'm really struggling mentally. Managing all these medications is overwhelming and I feel hopeless.",
            "I've been so stressed about my diagnosis that I haven't been sleeping or eating well.",
        ],
        "high": [
            "I feel like giving up on my medications. There's no point. Nothing is getting better.",
            "I haven't been taking care of myself at all. I don't really care what happens to me anymore.",
        ],
        "urgent": [
            "I don't want to be here anymore. I've been thinking about hurting myself.",
            "I've been having thoughts of suicide. My health problems are too much to handle.",
        ],
    },
}

CARE_GAP_MAP = {
    "medication_question": ["medication_adherence"],
    "symptom_escalation": ["symptom_monitoring", "urgent_followup"],
    "appointment_admin": ["appointment_scheduling"],
    "lab_results": ["lab_followup", "care_coordination"],
    "care_gap": ["medication_gap", "preventive_care", "specialist_referral"],
    "crisis": ["mental_health_referral", "safety_assessment"],
}

RECOMMENDED_ACTIONS = {
    ("medication_question", "low"): "Send patient education message",
    ("medication_question", "medium"): "Care manager outreach within 24h",
    ("medication_question", "high"): "Same-day care manager call",
    ("medication_question", "urgent"): "Immediate clinical escalation",
    ("symptom_escalation", "low"): "Schedule routine follow-up",
    ("symptom_escalation", "medium"): "Care manager outreach within 4h",
    ("symptom_escalation", "high"): "Immediate care manager call + provider alert",
    ("symptom_escalation", "urgent"): "Call 911 / direct to ED",
    ("appointment_admin", "low"): "Scheduling team handles",
    ("appointment_admin", "medium"): "Care manager coordinates appointment",
    ("appointment_admin", "high"): "Escalate to care manager",
    ("appointment_admin", "urgent"): "Immediate clinical escalation",
    ("lab_results", "low"): "Send results summary message",
    ("lab_results", "medium"): "Care manager review within 24h",
    ("lab_results", "high"): "Same-day provider notification",
    ("lab_results", "urgent"): "Immediate clinical escalation",
    ("care_gap", "low"): "Add to care gap closure queue",
    ("care_gap", "medium"): "Care manager outreach within 24h",
    ("care_gap", "high"): "Same-day care manager call",
    ("care_gap", "urgent"): "Immediate clinical escalation",
    ("crisis", "low"): "Warm handoff to care manager",
    ("crisis", "medium"): "Immediate care manager outreach",
    ("crisis", "high"): "Same-day mental health referral",
    ("crisis", "urgent"): "Crisis line + immediate provider alert",
}


def requires_human_review(risk_level: str, intent: str) -> bool:
    if risk_level in ("high", "urgent"):
        return True
    if intent == "crisis":
        return True
    return False


def generate_dataset(n_per_cell: int = 12) -> list[dict]:
    rows = []
    msg_id = 1
    for intent, risk_dict in TEMPLATES.items():
        for risk, messages in risk_dict.items():
            if not messages:
                continue
            sample_size = min(n_per_cell, len(messages))
            sampled = random.choices(messages, k=n_per_cell)
            for msg in sampled:
                rows.append(
                    {
                        "id": f"MSG_{msg_id:04d}",
                        "message": msg,
                        "intent": intent,
                        "risk_level": risk,
                        "care_gaps": json.dumps(CARE_GAP_MAP.get(intent, [])),
                        "recommended_action": RECOMMENDED_ACTIONS.get(
                            (intent, risk), "Care manager review"
                        ),
                        "requires_human_review": requires_human_review(risk, intent),
                        "rationale": f"Intent: {intent.replace('_', ' ')}. Risk: {risk}.",
                    }
                )
                msg_id += 1
    random.shuffle(rows)
    return rows


def save_dataset(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {path}")


if __name__ == "__main__":
    rows = generate_dataset(n_per_cell=12)
    save_dataset(rows, Path(__file__).parent / "synthetic_messages.csv")

    # Print distribution
    from collections import Counter
    intent_counts = Counter(r["intent"] for r in rows)
    risk_counts = Counter(r["risk_level"] for r in rows)
    print("\nIntent distribution:")
    for k, v in sorted(intent_counts.items()):
        print(f"  {k:<25} {v}")
    print("\nRisk distribution:")
    for k, v in sorted(risk_counts.items()):
        print(f"  {k:<10} {v}")
