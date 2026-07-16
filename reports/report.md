# Forecasting German Electricity Demand — Report

*Target length 6–8 pages including figures; references ≤ 0.5 page.*

## 1. Introduction
Motivation, the forecasting problem, and what "good" looks like operationally.
State the two-year (104-week) evaluation horizon and the seasonal-naive
reference model.

## 2. Data and preprocessing
Source (OPSD 60-min, 2020-10-06 release), the German actual-load column, MW→GW,
UTC→Europe/Berlin, the 2015-01-01→Oct-2020 window, and weekly/daily aggregation.
Note gap handling.

## 3. Exploratory analysis
Weekly and daily plots; STL decomposition (trend + annual seasonality +
residual); ACF/PACF. **Stationarity:** report ADF and KPSS on the level series
and after differencing, and interpret the two together. *(Figures: series,
decomposition, acf_pacf.)*

## 4. Forecasting methods
Brief description of each model and why it is included: benchmarks, SARIMAX
(with the AIC grid search and the temperature exog for the conditional
forecast), the feature-based gradient-boosting model, and the LSTM. One or two
sentences of theory per model with a citation.

## 5. Evaluation design
Train/test split (final 104 weeks, no shuffling), metrics (MAE, RMSE, MASE,
Bias), and the rationale for MASE as the headline. Mention rolling-origin as a
stronger alternative.

## 6. Results
The model-comparison table and the overlaid forecast figure. SARIMAX prediction
interval and its empirical coverage.

| model | MAE | RMSE | MASE | Bias |
|-------|-----|------|------|------|
| seasonal_naive | | | | |
| … | | | | |

## 7. Error analysis
Where models fail: error-over-time plot, residual ACF/Q-Q for SARIMAX,
behaviour around Christmas/New-Year and unusually hot/cold weeks. Discuss *why*,
tying back to model structure.

## 8. Discussion
Which models beat seasonal-naive and by how much; whether covariates helped;
whether complexity was justified; interpretability vs accuracy vs maintenance.

## 9. Limitations
Conditional-forecast caveat for temperature; short weekly sample for the LSTM;
single-location temperature proxy; COVID-era demand shift in 2020.

## 10. Conclusion
Recommended model for operational use and why.

---

## Assignment questions (Part 7)

1. **All models vs seasonal-naive** — which, if any, give a meaningful
   improvement (MASE and RMSE, with uncertainty)?
2. **Data leakage in temperature/lag features** — how `shift`-based
   construction and train-only fitting prevent it.
3. **SARIMAX differencing & seasonal period** — justify `d`, `D`, and `s = 52`
   from the ACF/PACF and stationarity tests.
4. **Do temperature/holiday covariates help?** Are they known at the forecast
   origin (temperature: no → conditional; holidays: yes)?
5. **Interpretability vs complexity** — SARIMAX vs gradient boosting vs LSTM.
6. **Operational recommendation** — justify via accuracy, uncertainty,
   interpretability, and maintenance.

## References
*(≤ 0.5 page. e.g. Hyndman & Athanasopoulos, *Forecasting: Principles and
Practice*; Hochreiter & Schmidhuber 1997 for LSTM; the OPSD data package;
relevant load-forecasting papers.)*
