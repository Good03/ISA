# User Manual — MNM Terminal

**Application:** NASDAQ-100 LSTM / Ridge Forecasting Workspace
**URL after install:** <http://localhost:8080>

> **Disclaimer.** Educational prototype. Forecasts are not financial advice.


---

## 1. Basic usage

1. **Pick a ticker(AAPL, AMZN, META...)** in the left **Watchlist** — chart, metrics and
   next-day forecast load instantly.
2. **Choose a model** with the `LSTM` (red) / `Ridge` (blue) buttons.
3. **Read the chart** — solid white = actual close, dashed = model
   prediction. Hover for exact values.
4. **Filter by date** above the chart to zoom in. Metrics in the right
   rail recompute on the visible window.

---

## 2. Retraining (optional)

Open **Hyperparameters & data splits** on the left.

* **LSTM** — set `Look-back`, `Units`, `Epochs`, click **Run training**.
  An overlay shows the staged progress; CPU training takes 2-10 min.
* **Ridge** — drag the `Alpha` slider (log scale, 10⁻⁴–10⁴), click
  **Apply & refit**. Returns in under a second.

After a run, the new metrics, the chart and the cross-symbol quality
table on the **Risk & Quality** tab all update in place.

---

## 3. Risk & Quality tab

A second tab summarises:

* **EU AI Act classification** (Limited Risk, Article 52).
* **Risk register** with severity / likelihood / mitigation.
* **Model quality** table across every trained ticker.
* **Known limitations** and an **improvement roadmap**.

Open it before drawing any conclusions about model quality.


