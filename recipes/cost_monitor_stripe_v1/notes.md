# cost_monitor (stripe)

`cost_pull` aggregates `GET /v1/balance_transactions` for the last
24 hours and feeds the daily total into `cost_anomaly.is_anomaly()`.
