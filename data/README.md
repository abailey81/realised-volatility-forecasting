# Data

The raw high-frequency price data and the macro series are **not included** in
this repository for licensing reasons. The pipeline expects the following
layout, which it then turns into daily realised measures and feature matrices:

```
data/
  raw/                 minute-bar OHLCV per stock: AAPL.txt, AMZN.txt, JPM.txt
                       format: MM/DD/YYYY,HH:MM,Open,High,Low,Close,Volume
  intermediate/        daily RV / RV± / RQ parquet  (produced by stage 1)
  features/            M_HAR and M_ALL feature matrices (produced by stage 3)
  macro/               FRED / Yahoo caches          (produced by stage 2)
```

Macro inputs are public: VIX (`VIXCLS`), EPU (`USEPUINDXD`), the 3-month T-bill
(`DTB3`) and the ADS index from FRED, and the Hang Seng index from Yahoo
Finance. Implied volatility is proxied by VIX (the paper uses licensed
OptionMetrics data). Stage 2 downloads and caches these; place the minute-bar
files in `data/raw/` yourself.
