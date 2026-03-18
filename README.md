# Dashboard Clarifications

## Active Signal

"Active Signal" means a trade opportunity has been detected by the strategy. This does **not** always mean a trade order has been placed. Whether an order is actually executed depends on:

- The `auto_trade_enabled` setting in `config.py`
- Risk and safety filters (e.g., VIX, confidence threshold, max orders per day)

To confirm if a trade is live, check the "Open Positions" or "Active Positions" section on the dashboard.

## Top Scored Candidates

"Top Scored Candidates" are the best option contracts (CE or PE) for the current signal direction:

- If the signal is **SELL**, the system recommends selling Put Options (PE) or Call Options (CE) based on the detected bearish or bullish signal.
- The "Type" column shows whether the candidate is a Call (CE) or Put (PE) option.

The candidates are filtered and scored based on the current market signal, risk, and scoring logic.

## Order Placement Logic

- If `auto_trade_enabled` is set to `True` in `config.py`, and all risk/safety checks pass, the system will automatically place an order when a valid signal is detected.
- If not, the system will only display the signal and wait for manual action.

**Summary:**

- "Active Signal" = trade opportunity detected (not always executed)
- "Top Scored Candidates" = best options to trade for the current signal direction
- "Open Positions" = actual trades that have been placed

# intraday-algo
