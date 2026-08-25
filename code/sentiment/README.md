# Sentiment and event data

Collectors prepare English source-whitelisted GDELT news, FRED releases, Fear & Greed and
registered direct events derived from Truth Social/Federal Reserve evidence. Scorers write
versioned, resumable caches:

- `score.py`: deterministic finance DeBERTa sentiment.
- `score_llm.py`: Ollama LLM sentiment, relevance, impact and asset fields in batches of 10.
- `index_scoring.py`: frozen index DeBERTa and DeepSeek scoring contracts.

Each unique headline is scored once per model/prompt identity. Downstream features use
as-of timestamps. Fresh BigQuery acquisition and cloud scoring require their respective
external credentials; secrets are not read from tracked files.
