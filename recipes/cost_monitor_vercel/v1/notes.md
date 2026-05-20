# cost_monitor (vercel)

`cost_pull` reads `GET /v9/teams/{teamId}/usage` and feeds the
day-window total into `cost_anomaly.is_anomaly()`.
