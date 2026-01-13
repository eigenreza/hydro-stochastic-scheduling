# Data Dictionary — weekly_panel.csv

| Column | Units | Description |
|--------|-------|-------------|
| week_start (index) | ISO date (Monday) | First day of ISO week (Monday–Sunday) |
| inflow_GWh_week | GWh/week | Energy-equivalent inflow; converted from mean daily discharge [m³/s] using fixed-head approximation (η=0.88, H_net=264.5 m, k=0.3836 GWh per m³/s) |
| price_avg_NOK_MWh | NOK/MWh | Mean day-ahead price in NO2 bidding zone over the week |
| price_std_NOK_MWh | NOK/MWh | Std. dev. of daily average prices within the week |
| iso_year | integer | ISO 8601 year |
| iso_week | integer | ISO 8601 week number (1–53) |
