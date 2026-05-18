# Crisis Comms Playbook — Tier 2: Outage / Data Issue

**Trigger:** Extended downtime (>15 min customer-visible), data loss, payment failure,
significant degradation that customers are experiencing.

**Target window:** First customer-facing status update within 30 min of detection.
Subsequent updates every 30–60 min until resolved.

---

## Immediate actions (first 30 min)

1. **Open incident in B3.** `/incident open` or B3 auto-opens from Z8 oncall alert.
2. **Freeze marketing sends.** `crisis/freeze_marketing` — no new A2 launches or B1 drips during outage.
3. **Draft first status update.** Use `incident/draft_update` (B3 verb). Keep it factual, no speculation.
4. **Post to status page.** After founder review via founder_action card.
5. **Notify affected users if data issue.** Consider B1 transactional email sequence.

## Status update cadence

| Time since detection | Update frequency |
|---|---|
| 0–2 hours | Every 30 min |
| 2–8 hours | Every 60 min |
| 8+ hours | Every 2 hours |
| Resolved | Immediate resolution notice + postmortem ETA |

## Holding statement shape (Tier 2)

```
We are currently experiencing [what is affected].
[Who is impacted] may notice [specific symptom].
Our team is actively investigating. Next update: [time].
We apologize for the disruption.
```

## Refund / credit policy decision card

Surfaced as a founder_action when incident duration > threshold:

- **>2h downtime:** Consider prorated credit for affected period.
- **Data loss:** Credit + personal outreach to affected users.
- **Payment failure:** Full refund + credit; no founder decision required if amount <$50.

## Recovery comms

1. **Resolution notice.** Posted immediately when resolved; includes root cause summary (non-technical).
2. **Postmortem.** Agent auto-drafts within 1h of resolution; founder edits + publishes within 7 days.
3. **Follow-up email.** If >1h downtime, send recovery email via B1 transactional sequence.

## Escalation triggers

- Confirmed data breach (PII/credentials exposed) → Tier 3.
- Regulatory inquiry received → Tier 3.
- Material litigation risk → Tier 4.

---

*Founder reviews each public update via founder_action card (max 4h SLA).*
*If founder unavailable >4h, B6 Tier 3 playbook review auto-triggers.*
