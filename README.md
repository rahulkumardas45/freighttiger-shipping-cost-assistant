# FreightTiger Smart Shipping Cost Assistant

## What it does
This notebook analyses weekly freight cost per tonne-km by route. It compares each observed route-week with its prior eight-calendar-week average and the current-week average of other routes sharing the same supplied route_type. It then uses ChromaDB semantic retrieval over the provided context notes. Python verifies route scope, effective date window and whether the note actually supports a cost increase before assigning a justified verdict. Gemini paraphrases only a verified supporting note; it cannot change calculations, verdicts or matched_note_id.

## Data and formula
- Week starts on Monday (`week_of`) and ends on Sunday.
- Weekly cost per tonne-km = sum of freight cost / sum of (quantity tonnes x distance km).
- Own baseline = mean of available values in the previous eight calendar weeks, excluding current week. Missing weeks remain missing; fewer prior weeks are not padded.
- Peer baseline = mean in the same week across other routes with the same route_type, excluding the current route.

## Flag rule
An anomaly is flagged if either available baseline is exceeded by at least 10%. The case study does not specify a threshold; 10% is this implementation's documented choice. `No` means no anomaly crossed the threshold; `No (justified)` means an anomaly has a route/date/cause-verified note; `Yes` means an unexplained anomaly requiring human review.

## Note validity assumptions
The provided notes CSV does not include structured effective dates. The manual interpretations are documented in `note_policy.csv` and in the NOTE_POLICY cell. Specific route/weather/festival notes are confined to their stated dates. Neutral notes and notes that explicitly say costs were not affected do not justify a rise. N004 is excluded because the text says the affected routes are not in this dataset. N003 mentions a 5-7% diesel effect and is not accepted as enough to justify an anomaly over the chosen 10% threshold by itself. These assumptions should be discussed during the walkthrough and revised if HR clarifies intended windows.

## Reproducibility and evaluation
`reproducibility_report.json` contains hashes for three repeated runs of the deterministic decision layer. It compares numeric calculations, flags and matched note IDs; natural-language wording is downstream and does not decide these values. `retrieval_audit.csv` gives vector-search rank, similarity, route/date verification and rejection reason for retrieved notes. Core logic assertions check Monday grouping, tonne-km formula, prior-only history and peer exclusion.

## LLM and cost
Embedding model: `sentence-transformers/all-MiniLM-L6-v2` locally in Colab. Vector database: local ChromaDB. Explanation model: `gemini-3.5-flash-lite` at temperature 0. Gemini calls are made only for route-weeks with verified supporting notes, not for every shipment. `token_cost_log.json` records returned API token counts and estimated published paid-tier-equivalent cost; actual billing may differ, especially under free-tier access. Pricing source: https://ai.google.dev/gemini-api/docs/pricing. Re-check it before the interview.

## Files
- `freighttiger_output.csv`: required output with exact eight-column schema.
- `note_policy.csv`: transparent reviewed note date/scope assumptions.
- `retrieval_audit.csv`: evidence retrieval and validation audit.
- `reproducibility_report.json`: three-run deterministic check.
- `token_cost_log.json`: LLM calls, tokens, and cost estimate.
- `FreightTiger_Complete_RAG_Colab.ipynb`: executable notebook.

## Run
Open the notebook in Google Colab, upload the three provided CSV files, set `GEMINI_API_KEY` in Colab Secrets, and run cells from top to bottom. Do not put API keys in source control or in your submission.
