# Crisis Comms Playbook — Tier 3: Security Incident / Breach

**Trigger:** Credentials leaked, unauthorized data access, PII exposed, security
vulnerability exploited, or credible report of a breach.

**Legal mandate:** GDPR requires notification to supervisory authority within 72 hours
of becoming aware of a personal data breach. Other jurisdictions vary (see table below).
The 72h disclosure timer activates automatically when this tier is opened.

---

## CRITICAL: First 1 hour

1. **Engage counsel immediately.** Founder_action "counsel engaged?" surfaces with two-button ack.
   - [Yes, counsel notified] → proceed to step 2.
   - [No — escalate now] → system suggests contacts; timer continues.
2. **Freeze ALL marketing.** `crisis/freeze_marketing` auto-runs at Tier 3 open.
3. **Contain the breach.** Rotate exposed credentials via Z8 oncall (rotate_failed_key verb).
4. **Identify scope.** What data? Which users? What time window?
5. **Document everything.** Every action timestamped (for regulator disclosure).

## 72h disclosure timer

KutAI runs `crisis/disclosure_timer` every 6h. Escalating reminders:

| Hours elapsed | Urgency | Action |
|---|---|---|
| 0–24h | Routine | "X hours remaining on GDPR 72h window" |
| 24–48h | Elevated | Urgent founder_action |
| 48–60h | Critical | "18h remaining — regulator notice must be filed" |
| 60–72h | BLOCKING | "FILING NOW or confirm extension" — counsel decision |

## Jurisdiction matrix

| Jurisdiction | Regulator notification | Customer notification | Deadline |
|---|---|---|---|
| GDPR (EU) | Supervisory authority | If "high risk to rights" | 72h for regulator |
| UK GDPR | ICO | If likely adverse effect | 72h for regulator |
| CCPA (California) | None required | Affected consumers | "Expedient" |
| LGPD (Brazil) | ANPD | Affected data subjects | "Reasonable" |
| PDPA (Turkey) | KVKK | Affected data subjects | 72h |
| Other | Check counsel | Check counsel | Check counsel |

## Holding statement shape (Tier 3)

```
We recently became aware of a security incident affecting [product name].
We are investigating the scope and impact. The security of your data is our priority.
[If confirmed breach:] We are notifying affected users and relevant authorities as required by law.
We will provide a full update by [date/time].
```

**Note:** Do NOT confirm breach scope publicly until counsel approves. Use "incident" not "breach"
until legally confirmed.

## Regulator notice template (pre-draft)

Agent will draft using jurisdiction-aware template. Counsel must review before filing.
Sections: data controller identity, breach description, categories + approx number of data
subjects affected, likely consequences, measures taken/proposed.

## Customer notification (legally-reviewed template)

After counsel approval. Sections: what happened, what data was involved, what we did,
what you should do, how to contact us.

## Press response

"We take security seriously and are investigating a potential incident. We will share more
information when we have a complete picture. Our security team and legal counsel are engaged."
Do NOT speculate on cause, scope, or attribution.

## Recovery path

1. Postmortem (internal) — root cause, timeline, remediation.
2. Customer notification email — after counsel approval.
3. Regulator filing — counsel-led; agent provides draft.
4. Public disclosure (if required) — 30 days after resolution.

---

*Founder owns all external communications. Agent drafts + tracks + reminds.*
*Counsel must review ALL customer notifications and regulator filings before send.*
