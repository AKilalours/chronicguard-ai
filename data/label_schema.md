# Label Schema — ChronicGuard AI

## Overview
All messages are labeled across three dimensions: **intent**, **risk level**, and **care gaps**.
Labels were assigned by the project author using clinical reasoning guidelines informed by
CMS Chronic Care Management (CCM) program documentation and standard triage protocols.

---

## Intent Categories

| Label | Description | Examples |
|---|---|---|
| `medication_question` | Patient asks about dosing, timing, side effects, refills, or interactions | "Can I take ibuprofen with my blood thinner?" |
| `symptom_escalation` | Patient reports new or worsening physical symptoms | "I've had chest tightness since this morning" |
| `appointment_admin` | Scheduling, referrals, records, or care coordination requests | "Can I reschedule my follow-up?" |
| `lab_results` | Patient asking about or reacting to recent lab/diagnostic results | "My A1C came back 8.2 and no one called me" |
| `care_gap` | Identified lapse in expected care activity (medication, visit, test) | "I've been out of my blood thinner for 5 days" |
| `crisis` | Patient expressing emotional distress, hopelessness, or suicidal ideation | "I don't want to be here anymore" |

**Labeling rule:** Assign the *most specific* intent. A message about a missed medication
refill that is causing a care gap is labeled `care_gap`, not `medication_question`.

---

## Risk Level

| Label | Definition | Required Action SLA |
|---|---|---|
| `low` | Routine inquiry; no immediate safety concern | Response within 48h |
| `medium` | Elevated concern; may deteriorate without follow-up | Care manager outreach within 24h |
| `high` | Significant safety concern; clear clinical relevance | Same-day outreach or provider alert |
| `urgent` | Immediate threat to life or patient safety | Real-time escalation; may require 911 / ED |

**Safety note:** In a production system, false negatives on `urgent` and `high` are
categorically worse than false positives. The evaluation framework reflects this asymmetry
by treating urgent-class recall as a hard constraint (target ≥ 0.92), not a trade-off metric.

---

## Care Gap Types

| Label | Definition |
|---|---|
| `medication_adherence` | Patient not taking medication as prescribed |
| `medication_gap` | Patient out of medication; refill lapsed or denied |
| `symptom_monitoring` | Symptoms reported but not yet clinically evaluated |
| `urgent_followup` | Clinical follow-up overdue given reported symptom severity |
| `appointment_scheduling` | Needed appointment not yet booked |
| `lab_followup` | Lab result received but not acted on by care team |
| `care_coordination` | Specialist referral or inter-team handoff not completed |
| `preventive_care` | Routine preventive service overdue (eye exam, foot exam, flu shot) |
| `specialist_referral` | Referral issued but appointment not yet confirmed |
| `mental_health_referral` | Patient showing signs of distress requiring behavioral health support |
| `safety_assessment` | Patient expressing self-harm ideation; immediate assessment needed |

---

## Human Review Flag

`requires_human_review = True` is set automatically when:
- `risk_level` is `high` or `urgent`
- `intent` is `crisis` (regardless of risk level)

This reflects the principle that high-stakes healthcare communication should never be
fully automated without a licensed care manager in the loop.

---

## Dataset Integrity Notes

- All messages are **synthetically generated**. No real patient data or PHI was used.
- Messages were designed to reflect realistic chronic care management scenarios based
  on published CCM program literature.
- Label assignment reflects the author's clinical reasoning, not a validated clinical instrument.
  In a production setting, labels would be reviewed and adjudicated by licensed clinicians.
