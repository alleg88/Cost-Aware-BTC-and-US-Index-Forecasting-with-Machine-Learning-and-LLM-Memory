# Research questions and evidence scope

1. **RQ1:** How do nine pre-specified ML model families compare in short-horizon
   directional prediction and net-of-cost trading performance?
2. **RQ2:** Do pre-specified all-nine ensemble methods improve net-of-cost performance
   over the best eligible single model?
3. **RQ3:** Do sentiment features improve performance over a market-only control?
4. **RQ4:** Does host-controlled LLM routing improve BTC net-of-cost performance over
   fixed controls?
5. **RQ5:** Do channel or volatility signals improve BTC net-of-cost performance over
   matched controls?

| Question | Implemented comparison | Boundary |
|---|---|---|
| RQ1 | Nine individual families under one chronological protocol | BTC confirmatory; indices supplementary and non-confirmatory |
| RQ2 | All-nine soft vote, directional vote and causal logistic meta-model versus an eligible single model | BTC Qualified Union is a separate sparse comparison |
| RQ3 | Market-only versus DeBERTa and structured-LLM feature arms | Q2 feed-availability check is supplementary |
| RQ4 | Real memory versus no-memory, shuffled-memory and deterministic routing controls | Available-stage answer complete; no router candidate entered Q2 |
| RQ5 | Channel-conditioned versus matched channel-blind volatility/opportunity windows | Development-stage non-promotion decision; branch closed before calibration |
