# Court Team Backlog — Prediction Report

Ticket-level analysis of the court team backlog predictions
(`court_team_backlog_predictions_full.csv`). Each ticket (`zendesk_id`) is
assigned its dominant `detected_as` (model detection) and `automated_type`
(automation decision) across its attachments.

## 1. Headline numbers

| Metric | Count | Share |
|---|---:|---:|
| Total tickets | 8,435 | 100% |
| Detected (any type) | 6,664 | 79.0% |
| Automated (any type) | 1,630 | 19.3% |

- Detection covers ~4 out of 5 tickets, but only ~1 in 5 is automated end-to-end.
- The gap between detection (79.0%) and automation (19.3%) is the main automation headroom.

## 2. Detected type distribution (tickets)

| Detected as | Tickets | Share |
|---|---:|---:|
| dritt | 3,781 | 44.8% |
| none | 1,771 | 21.0% |
| standalone_invoice | 1,346 | 16.0% |
| pfub | 909 | 10.8% |
| va | 512 | 6.1% |
| ladung | 116 | 1.4% |

## 3. Automated type distribution (tickets)

| Automated as | Tickets | Share |
|---|---:|---:|
| none | 6,805 | 80.7% |
| invoice | 1,350 | 16.0% |
| drittauskunft | 168 | 2.0% |
| ladung_va | 110 | 1.3% |
| pfub_erlass | 2 | 0.0% |

## 4. Automation rate per detected type

Automated counted as a strict subset of detected (a ticket counts as automated
for a type only when its automation decision matches the detected type).

| Detected as | Detected | Automated | Automation rate |
|---|---:|---:|---:|
| dritt | 3,781 | 167 | 4.4% |
| standalone_invoice | 1,346 | 1,326 | 98.5% |
| pfub | 909 | 2 | 0.2% |
| va | 512 | 0 | 0.0% |
| ladung | 116 | 110 | 94.8% |

- High automation: standalone_invoice (98.5%) and ladung (94.8%).
- Largest untapped volume: **dritt** — 3,781 detected but only 4.4% automated.
- pfub and va are detected in volume but almost never automated (0.2% / 0.0%).

## 5. Invoice inside per detected type (attachment level)

An "invoice inside" is a common blocker for automation.

| Detected as | Detected docs | With invoice inside | Share |
|---|---:|---:|---:|
| dritt | 3,781 | 2,712 | 71.7% |
| pfub | 909 | 835 | 91.9% |
| va | 512 | 425 | 83.0% |

- A large share of dritt/pfub/va documents contain an invoice inside, which
  blocks straight-through automation and largely explains the low automation
  rates for dritt, pfub and va in section 4.

## Notes

- `total_tickets` = unique `zendesk_id` values.
- Ticket-level detection/automation uses the dominant (mode) value across a
  ticket's attachments; per-document figures in section 5 are computed on
  `analysis_df` (attachment level).
