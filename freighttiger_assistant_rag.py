# FreightTiger Smart Shipping Cost Assistant — source exported from the explained Colab notebook.
# Run in Google Colab after uploading the three CSV inputs.
# Set GEMINI_API_KEY in Colab Secrets; no key is stored in this file.

from pathlib import Path
from getpass import getpass
import csv
import json
import hashlib
import re
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from IPython.display import display

DATA_DIR = Path("/content")
SHIPMENTS_FILE = DATA_DIR / "shipment_records.csv"
NOTES_FILE = DATA_DIR / "context_notes.csv"
TEMPLATE_FILE = DATA_DIR / "sample_output_format_v2.csv"
OUTPUT_FILE = DATA_DIR / "freighttiger_output.csv"

HISTORY_WEEKS = 8
ANOMALY_THRESHOLD = 0.10  # Chosen heuristic; not specified by FreightTiger
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-3.5-flash-lite"

# Standard paid-tier equivalent rates used in the final recorded run.
# Re-check the official page on the submission day; rates can change.
INPUT_USD_PER_MILLION_TOKENS = 0.30
OUTPUT_USD_PER_MILLION_TOKENS = 2.50
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"

EXPECTED_OUTPUT_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history",
    "vs_similar_routes", "flagged", "matched_note_id", "reason"
]

print("Configuration loaded.")
print(f"History window: {HISTORY_WEEKS} previous calendar weeks")
print(f"Chosen anomaly threshold: {ANOMALY_THRESHOLD:.0%}")

# %%

def load_input_files():
    """Read the supplied inputs and validate required columns and template header."""
    for path in [SHIPMENTS_FILE, NOTES_FILE, TEMPLATE_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}. Upload it in Step 2.")

    shipment_df = pd.read_csv(SHIPMENTS_FILE)
    notes_df = pd.read_csv(NOTES_FILE)
    with open(TEMPLATE_FILE, "r", encoding="utf-8-sig", newline="") as f:
        template_header = next(csv.reader(f))

    required_shipment_columns = {
        "origin", "destination", "route_type", "quantity_tonnes",
        "distance_km", "freight_cost_inr", "shipment_date"
    }
    required_note_columns = {"note_id", "date", "applies_to", "note"}

    missing_ship = required_shipment_columns - set(shipment_df.columns)
    missing_notes = required_note_columns - set(notes_df.columns)
    if missing_ship:
        raise ValueError(f"Missing shipment columns: {sorted(missing_ship)}")
    if missing_notes:
        raise ValueError(f"Missing note columns: {sorted(missing_notes)}")
    if template_header != EXPECTED_OUTPUT_COLUMNS:
        raise ValueError(
            "Template header/order does not match assignment contract.\n"
            f"Found: {template_header}\nExpected: {EXPECTED_OUTPUT_COLUMNS}"
        )
    return shipment_df, notes_df

shipments_raw, notes_raw = load_input_files()
print(f"Shipment rows: {len(shipments_raw):,}")
print(f"Context notes: {len(notes_raw):,}")
print("Shipment columns:", shipments_raw.columns.tolist())
print("Note columns:", notes_raw.columns.tolist())
display(shipments_raw.head(3))
display(notes_raw)

# %%

NOTE_POLICY = {
    # note_id: (effective_start, effective_end, scope_kind, can_support_anomaly, policy_explanation)
    "N001": ("2025-02-24", "2025-03-08", "route", True,
             "Flooding caused detours and higher trip costs on Chennai-Bangalore during the stated dates."),
    "N002": ("2025-01-20", "2025-01-26", "route", True,
             "The regional festival caused a temporary surcharge on Ahmedabad-Mumbai during that festival week."),
    "N003": ("2025-05-05", "2025-05-11", "all", False,
             "The note reports a roughly 5-7% diesel effect for the stated week; it does not substantiate an anomaly that crosses our 10% threshold by itself."),
    "N004": ("2024-03-11", "2024-03-17", "none", False,
             "The note explicitly says the routes affected by the toll change are not in this dataset."),
    "N005": ("2024-07-29", "2024-08-04", "route", False,
             "The note says maintenance caused minor delays but costs were not significantly affected."),
    "N006": ("2025-09-22", "2025-09-28", "all", False,
             "The note reports stable demand/no major disruptions; it does not provide a cause for higher costs. Date window is conservatively limited to the report week."),
    "N007": ("2024-05-20", "2024-05-26", "route", False,
             "The note describes improved road conditions on Delhi-Jaipur, not a cost-increasing disruption."),
    "N008": ("2025-06-09", "2025-06-30", "route", False,
             "The note reports no significant disruptions on Kolkata-Bhubaneswar from its report date through the stated quarter end."),
    "N009": ("2025-03-17", "2025-03-23", "route", False,
             "The note says Chennai-Bangalore had returned to normal after flood-related repairs."),
    "N010": ("2025-10-27", "2025-11-02", "all", False,
             "The note says vehicle-tracking compliance costs were absorbed by transporters without a rate change; window limited to report week."),
}

policy_rows = []
for note_id, values in NOTE_POLICY.items():
    start, end, scope_kind, can_support, explanation = values
    policy_rows.append({
        "note_id": note_id,
        "effective_start": pd.Timestamp(start),
        "effective_end": pd.Timestamp(end),
        "scope_kind": scope_kind,
        "can_support_anomaly": bool(can_support),
        "policy_explanation": explanation,
    })
policy_df = pd.DataFrame(policy_rows)

notes = notes_raw.copy()
notes["date"] = pd.to_datetime(notes["date"], errors="raise")
notes["applies_to"] = notes["applies_to"].astype(str).str.strip()
notes["note"] = notes["note"].astype(str).str.strip()
notes = notes.merge(policy_df, on="note_id", how="left", validate="one_to_one")
if notes["effective_start"].isna().any():
    missing_ids = notes.loc[notes["effective_start"].isna(), "note_id"].tolist()
    raise ValueError(f"Add a reviewed date/scope policy for notes: {missing_ids}")
if set(notes["note_id"]) != set(NOTE_POLICY):
    warnings.warn("The notes CSV differs from the policy; review NOTE_POLICY before continuing.")

display(notes[["note_id", "date", "applies_to", "effective_start", "effective_end", "can_support_anomaly", "policy_explanation"]])
notes[["note_id", "date", "applies_to", "effective_start", "effective_end", "scope_kind", "can_support_anomaly", "policy_explanation", "note"]].to_csv(DATA_DIR / "note_policy.csv", index=False)

# %%

def prepare_shipments(raw_df):
    """Validate shipment values and create route, Monday week, and tonne-km."""
    data = raw_df.copy()
    for col in ["origin", "destination", "route_type"]:
        data[col] = data[col].astype("string").str.strip()
        if data[col].isna().any() or data[col].eq("").any():
            raise ValueError(f"Blank values in required column {col}")

    data["shipment_date"] = pd.to_datetime(data["shipment_date"], errors="coerce")
    if data["shipment_date"].isna().any():
        raise ValueError("Invalid shipment_date found.")

    for col in ["quantity_tonnes", "distance_km", "freight_cost_inr"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isna().any():
            raise ValueError(f"Invalid numeric value in {col}")

    if (data["quantity_tonnes"] <= 0).any():
        raise ValueError("quantity_tonnes must be greater than zero")
    if (data["distance_km"] <= 0).any():
        raise ValueError("distance_km must be greater than zero")
    if (data["freight_cost_inr"] < 0).any():
        raise ValueError("freight_cost_inr cannot be negative")

    data["route"] = data["origin"] + "-" + data["destination"]
    data["week_of"] = (
        data["shipment_date"]
        - pd.to_timedelta(data["shipment_date"].dt.weekday, unit="D")
    ).dt.normalize()
    data["tonne_km"] = data["quantity_tonnes"] * data["distance_km"]
    return data

prepared_shipments = prepare_shipments(shipments_raw)
print("Shipment date range:", prepared_shipments["shipment_date"].min().date(), "to", prepared_shipments["shipment_date"].max().date())
print("Unique routes:", prepared_shipments[["route", "route_type"]].drop_duplicates().shape[0])
display(prepared_shipments[["route", "route_type", "shipment_date", "week_of", "tonne_km"]].head())

# %%

def calculate_weekly_costs(prepared_df):
    """Return a complete route-by-calendar-week grid with weekly cost per tonne-km."""
    weekly_totals = (
        prepared_df.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            total_freight_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_date", "count"),
        )
    )
    weekly_totals["cost_per_tonne_km"] = (
        weekly_totals["total_freight_cost"] / weekly_totals["total_tonne_km"]
    )

    routes = prepared_df[["route", "route_type"]].drop_duplicates().copy()
    weeks = pd.date_range(prepared_df["week_of"].min(), prepared_df["week_of"].max(), freq="W-MON")
    calendar = pd.DataFrame({"week_of": weeks})
    routes["_join"] = 1
    calendar["_join"] = 1
    grid = routes.merge(calendar, on="_join").drop(columns="_join")
    weekly = grid.merge(
        weekly_totals,
        on=["route", "route_type", "week_of"],
        how="left",
        validate="one_to_one",
    )
    return weekly.sort_values(["route", "week_of"]).reset_index(drop=True)

weekly_template = calculate_weekly_costs(prepared_shipments)
print("All route/calendar-week rows:", len(weekly_template))
print("Route-weeks with shipments:", int(weekly_template["cost_per_tonne_km"].notna().sum()))
display(weekly_template[weekly_template["cost_per_tonne_km"].notna()].head(8)[["route", "route_type", "week_of", "shipment_count", "cost_per_tonne_km"]])

# %%

def add_baselines_and_flags(weekly_df):
    """Add the prior-only eight-week baseline, peer baseline, and anomaly fields."""
    data = weekly_df.copy().sort_values(["route", "week_of"]).reset_index(drop=True)

    data["vs_own_history"] = data.groupby("route")["cost_per_tonne_km"].transform(
        lambda values: values.shift(1).rolling(window=HISTORY_WEEKS, min_periods=1).mean()
    )

    grouped = data.groupby(["week_of", "route_type"])["cost_per_tonne_km"]
    total_cost = grouped.transform("sum")
    available_count = grouped.transform("count")
    current_present = data["cost_per_tonne_km"].notna().astype(int)
    peer_count = available_count - current_present
    peer_sum = total_cost - data["cost_per_tonne_km"].fillna(0)
    data["vs_similar_routes"] = np.where(
        data["cost_per_tonne_km"].notna() & (peer_count > 0),
        peer_sum / peer_count,
        np.nan,
    )

    def pct_change(current, baseline):
        if pd.isna(current) or pd.isna(baseline) or baseline == 0:
            return np.nan
        return float(current / baseline - 1.0)

    data["own_change"] = [pct_change(c, b) for c, b in zip(data["cost_per_tonne_km"], data["vs_own_history"])]
    data["peer_change"] = [pct_change(c, b) for c, b in zip(data["cost_per_tonne_km"], data["vs_similar_routes"])]
    data["own_triggered"] = data["own_change"].ge(ANOMALY_THRESHOLD).fillna(False)
    data["peer_triggered"] = data["peer_change"].ge(ANOMALY_THRESHOLD).fillna(False)
    data["is_anomaly"] = data["own_triggered"] | data["peer_triggered"]
    return data

weekly = add_baselines_and_flags(weekly_template)
rows_with_shipments = weekly[weekly["cost_per_tonne_km"].notna()].copy()
print("Anomalies flagged:", int(rows_with_shipments["is_anomaly"].sum()), "of", len(rows_with_shipments), "observed route-weeks")
display(rows_with_shipments[["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "own_change", "peer_change", "is_anomaly"]].head(12))

# %%

def run_core_tests(prepared_df, weekly_df):
    assert (prepared_df["week_of"].dt.weekday == 0).all(), "week_of must be Monday"
    assert (prepared_df["tonne_km"] > 0).all(), "Every shipment must have positive tonne-km"

    # Manually recompute a weekly cost from the raw rows.
    sample = weekly_df[weekly_df["cost_per_tonne_km"].notna()].iloc[0]
    subset = prepared_df[
        (prepared_df["route"] == sample["route"])
        & (prepared_df["week_of"] == sample["week_of"])
    ]
    expected = subset["freight_cost_inr"].sum() / subset["tonne_km"].sum()
    assert np.isclose(sample["cost_per_tonne_km"], expected), "Weekly cost formula failed"

    # The first calendar week has no earlier history.
    first_week = weekly_df["week_of"].min()
    first_rows = weekly_df[weekly_df["week_of"] == first_week]
    assert first_rows["vs_own_history"].isna().all(), "History baseline leaked current/future values"

    # Peer average for one observed route/week must exclude the route itself.
    observed = weekly_df[weekly_df["cost_per_tonne_km"].notna() & weekly_df["vs_similar_routes"].notna()].iloc[0]
    peers = weekly_df[
        (weekly_df["week_of"] == observed["week_of"])
        & (weekly_df["route_type"] == observed["route_type"])
        & (weekly_df["route"] != observed["route"])
        & weekly_df["cost_per_tonne_km"].notna()
    ]["cost_per_tonne_km"]
    assert np.isclose(observed["vs_similar_routes"], peers.mean()), "Peer average failed to exclude current route"

    return "PASS: Monday grouping, tonne-km, weekly rate formula, prior-only history, peer exclusion."

print(run_core_tests(prepared_shipments, weekly))

# %%

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

notes = notes.reset_index(drop=True).copy()
notes["document_text"] = (
    "Note ID: " + notes["note_id"].astype(str)
    + ". Applies to: " + notes["applies_to"].astype(str)
    + ". Reported date: " + notes["date"].dt.strftime("%Y-%m-%d")
    + ". Note: " + notes["note"].astype(str)
)

note_vectors = embedding_model.encode(
    notes["document_text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True,
)

# Recreate this project's collection on a rerun to avoid stale/duplicate notes.
chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "freighttiger_chroma_db"))
collection_name = "freighttiger_context_notes"
existing_names = [c.name if hasattr(c, "name") else str(c) for c in chroma_client.list_collections()]
if collection_name in existing_names:
    chroma_client.delete_collection(collection_name)
notes_collection = chroma_client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"},
)

metadata_rows = []
for _, note in notes.iterrows():
    metadata_rows.append({
        "note_id": str(note["note_id"]),
        "applies_to": str(note["applies_to"]),
        "reported_date": note["date"].strftime("%Y-%m-%d"),
        "effective_start": pd.Timestamp(note["effective_start"]).strftime("%Y-%m-%d"),
        "effective_end": pd.Timestamp(note["effective_end"]).strftime("%Y-%m-%d"),
        "scope_kind": str(note["scope_kind"]),
        "can_support_anomaly": bool(note["can_support_anomaly"]),
    })

notes_collection.add(
    ids=notes["note_id"].astype(str).tolist(),
    documents=notes["document_text"].tolist(),
    embeddings=np.asarray(note_vectors, dtype=np.float32).tolist(),
    metadatas=metadata_rows,
)
print("Chroma collection created:", notes_collection.name)
print("Notes indexed:", notes_collection.count())

# %%

def build_retrieval_query(row):
    own = "unavailable" if pd.isna(row["own_change"]) else f"{row['own_change']:+.1%}"
    peer = "unavailable" if pd.isna(row["peer_change"]) else f"{row['peer_change']:+.1%}"
    return (
        f"Find context notes that may explain an unusual freight cost increase for route {row['route']} "
        f"with route type {row['route_type']} during week starting {pd.Timestamp(row['week_of']).date()}. "
        f"Change versus its own prior history is {own}; change versus other routes of the same type is {peer}. "
        "Look for a route- and date-specific festival surcharge, fuel price change, toll change, or weather disruption."
    )


def retrieve_and_verify_notes(analysis_df):
    """Semantic retrieval followed by deterministic route/date/cause verification."""
    anomaly_rows = analysis_df[analysis_df["is_anomaly"]].copy().reset_index(drop=True)
    verified_by_key = {}
    nearest_by_key = {}
    audit_rows = []

    if anomaly_rows.empty:
        return verified_by_key, nearest_by_key, pd.DataFrame(columns=[
            "route", "week_of", "rank", "note_id", "similarity", "route_match",
            "date_match", "can_support_anomaly", "accepted_as_evidence", "rejection_reason"
        ])

    query_texts = [build_retrieval_query(row) for _, row in anomaly_rows.iterrows()]
    query_vectors = embedding_model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)
    search_results = notes_collection.query(
        query_embeddings=np.asarray(query_vectors, dtype=np.float32).tolist(),
        n_results=max(1, notes_collection.count()),
        include=["metadatas", "documents", "distances"],
    )

    for query_index, (_, row) in enumerate(anomaly_rows.iterrows()):
        route = str(row["route"])
        week_start = pd.Timestamp(row["week_of"]).normalize()
        week_end = week_start + pd.Timedelta(days=6)
        key = (route, week_start.strftime("%Y-%m-%d"))
        ranked_notes = []

        for rank, (metadata, document, distance) in enumerate(zip(
            search_results["metadatas"][query_index],
            search_results["documents"][query_index],
            search_results["distances"][query_index],
        ), start=1):
            note_id = str(metadata["note_id"])
            applies_to = str(metadata["applies_to"])
            route_match = applies_to in {route, "All Routes"}
            effective_start = pd.Timestamp(metadata["effective_start"])
            effective_end = pd.Timestamp(metadata["effective_end"])
            # Conservative evidence policy: the note must cover the entire Monday-Sunday week.
            # This is an implementation choice; the brief requires route/time applicability but does not prescribe this exact test.
            date_match = effective_start <= week_start and effective_end >= week_end
            can_support = bool(metadata["can_support_anomaly"])
            accepted = bool(route_match and date_match and can_support)

            if not route_match:
                rejection = f"scope is {applies_to}, not {route}"
            elif not date_match:
                rejection = (f"effective window {effective_start.date()} to {effective_end.date()} "
                             f"does not cover the full week {week_start.date()} to {week_end.date()}")
            elif not can_support:
                rejection = "note does not substantiate a cost increase above the chosen threshold"
            else:
                rejection = ""

            audit_rows.append({
                "route": route,
                "week_of": key[1],
                "rank": rank,
                "note_id": note_id,
                "similarity": round(1.0 - float(distance), 6),
                "route_match": route_match,
                "date_match": date_match,
                "can_support_anomaly": can_support,
                "accepted_as_evidence": accepted,
                "rejection_reason": rejection,
            })
            ranked_notes.append({
                "note_id": note_id,
                "applies_to": applies_to,
                "reported_date": str(metadata["reported_date"]),
                "effective_start": effective_start,
                "effective_end": effective_end,
                "note_text": str(document),
                "distance": float(distance),
                "route_match": route_match,
                "date_match": date_match,
                "can_support_anomaly": can_support,
                "accepted": accepted,
                "rejection_reason": rejection,
            })

        # Search results arrive in ascending cosine distance, i.e. best semantic match first.
        nearest_by_key[key] = ranked_notes[0] if ranked_notes else None
        support_candidates = [candidate for candidate in ranked_notes if candidate["accepted"]]
        verified_by_key[key] = support_candidates[0] if support_candidates else None

    audit_df = pd.DataFrame(audit_rows)
    return verified_by_key, nearest_by_key, audit_df

verified_evidence, nearest_retrieved_notes, retrieval_audit = retrieve_and_verify_notes(weekly)
print("Anomalous route-weeks sent to semantic retrieval:", int(weekly["is_anomaly"].sum()))
print("Route-weeks with a route/date/cause-verified support note:", sum(v is not None for v in verified_evidence.values()))
retrieval_audit.to_csv(DATA_DIR / "retrieval_audit.csv", index=False)
display(retrieval_audit[retrieval_audit["accepted_as_evidence"]].head(10))

# %%

STABLE_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
    "own_change", "peer_change", "is_anomaly"
]

def run_deterministic_pass():
    prepared = prepare_shipments(shipments_raw)
    fresh_weekly = add_baselines_and_flags(calculate_weekly_costs(prepared))
    fresh_verified, fresh_nearest, fresh_audit = retrieve_and_verify_notes(fresh_weekly)
    # Add the verified evidence ID to make the citation itself part of repeatability.
    observed = fresh_weekly[fresh_weekly["cost_per_tonne_km"].notna()].copy()
    observed["matched_note_id"] = [
        (fresh_verified.get((str(row["route"]), pd.Timestamp(row["week_of"]).strftime("%Y-%m-%d"))) or {}).get("note_id", "")
        if bool(row["is_anomaly"]) else ""
        for _, row in observed.iterrows()
    ]
    return observed.reset_index(drop=True)

repeat_runs = [run_deterministic_pass() for _ in range(3)]
repeat_hashes = {}
for index, result_df in enumerate(repeat_runs, start=1):
    stable_bytes = result_df[STABLE_COLUMNS + ["matched_note_id"]].to_csv(index=False, float_format="%.12f").encode("utf-8")
    repeat_hashes[f"run_{index}"] = hashlib.sha256(stable_bytes).hexdigest()

repeatability_passed = len(set(repeat_hashes.values())) == 1
print(json.dumps(repeat_hashes, indent=2))
print("Core numbers, flags, and citations identical across 3 runs:", repeatability_passed)
if not repeatability_passed:
    raise AssertionError("Repeatability check failed. Inspect the analysis before submission.")
with open(DATA_DIR / "reproducibility_report.json", "w", encoding="utf-8") as f:
    json.dump({"identical": repeatability_passed, "compared_fields": STABLE_COLUMNS + ["matched_note_id"], "sha256": repeat_hashes}, f, indent=2)

# %%

GEMINI_API_KEY = None
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    GEMINI_API_KEY = getpass("Enter Gemini API key (input is hidden): ").strip()
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key is required for the LLM explanation step.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print("Gemini client configured. The key is not printed or written to project files.")

# %%

TOKEN_USAGE = {
    "calls_attempted": 0,
    "calls_succeeded": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "usage_metadata_missing_calls": 0,
    "api_errors": 0,
    "fallback_explanations": 0,
}

def deterministic_evidence_sentence(evidence):
    """Fallback sentence grounded only in the manually reviewed policy."""
    note_id = evidence["note_id"]
    policy = notes.loc[notes["note_id"].eq(note_id), "policy_explanation"].iloc[0]
    return policy


def write_grounded_explanation(row, evidence):
    """Generate one concise note-grounded sentence; do not let the LLM decide the verdict."""
    note_id = evidence["note_id"]
    note_row = notes.loc[notes["note_id"].eq(note_id)].iloc[0]
    prompt = f"""
You are writing one short explanation for a shipping-cost analyst.
Use ONLY the verified context note below. Do not invent causes, dates, locations, or facts.
Do not change the verdict. Do not mention any other note ID. Do not repeat numerical comparisons.
Write 1-2 plain-English sentences explaining what the provided note says and why it is relevant to this route-week. If the note is cautious, preserve that caution. Do not claim the note proves more than it states.

Verified route: {row['route']}
Verified week starting: {pd.Timestamp(row['week_of']).date()}
Verified note ID: {note_id}
Note reported date: {note_row['date'].date()}
Note text: {note_row['note']}
Reviewed evidence policy: {note_row['policy_explanation']}
Return only the explanation text, no heading and no JSON.
""".strip()

    TOKEN_USAGE["calls_attempted"] += 1
    try:
        response = gemini_client.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=256,
            ),
        )
        TOKEN_USAGE["calls_succeeded"] += 1
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            input_count = getattr(usage, "prompt_token_count", None)
            output_count = getattr(usage, "candidates_token_count", None)
            if input_count is not None and output_count is not None:
                TOKEN_USAGE["input_tokens"] += int(input_count)
                TOKEN_USAGE["output_tokens"] += int(output_count)
            else:
                TOKEN_USAGE["usage_metadata_missing_calls"] += 1
        else:
            TOKEN_USAGE["usage_metadata_missing_calls"] += 1

        generated_text = (getattr(response, "text", None) or "").strip().strip('"')
        unexpected_ids = set(re.findall(r"\bN\d{3}\b", generated_text)) - {note_id}
        if not generated_text or unexpected_ids or len(generated_text) > 700:
            TOKEN_USAGE["fallback_explanations"] += 1
            return deterministic_evidence_sentence(evidence)
        return generated_text
    except Exception as exc:
        TOKEN_USAGE["api_errors"] += 1
        TOKEN_USAGE["fallback_explanations"] += 1
        print(f"Gemini call failed for {note_id}; using deterministic fallback: {type(exc).__name__}")
        return deterministic_evidence_sentence(evidence)

print("Grounded explanation function ready.")

# %%

def format_change(value, label):
    if pd.isna(value):
        return f"Unavailable: no valid {label} baseline"
    return f"{value:+.1%} vs {label}"


def describe_rejected_nearest(note_candidate, route, week_of):
    """Explain briefly why the nearest retrieved note was not accepted as evidence."""
    if not note_candidate:
        return "No context note was retrieved."
    note_id = note_candidate["note_id"]
    if not note_candidate["route_match"]:
        return f"The closest retrieved note {note_id} applies to {note_candidate['applies_to']}, not {route}, so it was rejected."
    if not note_candidate["date_match"]:
        week_start = pd.Timestamp(week_of).normalize()
        week_end = week_start + pd.Timedelta(days=6)
        return (
            f"The closest retrieved note {note_id} has a reviewed effective window of "
            f"{note_candidate['effective_start'].date()} to {note_candidate['effective_end'].date()}, "
            f"which does not cover the full week ({week_start.date()} to {week_end.date()}); "
            f"it was rejected under the conservative evidence policy."
        )
    return f"The closest retrieved note {note_id} does not establish a cost-increasing cause: {note_candidate['rejection_reason']}."


def create_output_row(row, verified_map, nearest_map):
    """Create one contract-compatible output row; decisions are deterministic."""
    route = str(row["route"])
    week_text = pd.Timestamp(row["week_of"]).strftime("%Y-%m-%d")
    key = (route, week_text)
    is_anomaly = bool(row["is_anomaly"])

    own_text = format_change(row["own_change"], "this route's past average")
    peer_text = format_change(row["peer_change"], "similar-length routes this week")
    comparison_text = f"Comparison: {own_text}; {peer_text}."

    note_id = ""
    if not is_anomaly:
        verdict = "No"
        reason = f"No increase exceeded the configured {ANOMALY_THRESHOLD:.0%} threshold. {comparison_text}"
    else:
        evidence = verified_map.get(key)
        if evidence:
            verdict = "No (justified)"
            note_id = evidence["note_id"]
            generated = write_grounded_explanation(row, evidence)
            note_date = pd.Timestamp(evidence["reported_date"]).strftime("%Y-%m-%d")
            reason = f"Justified by note {note_id} dated {note_date}: {generated} {comparison_text}"
        else:
            verdict = "Yes"
            nearest = nearest_map.get(key)
            rejection = describe_rejected_nearest(nearest, route, row["week_of"])
            reason = (
                f"No supporting note passed route, date, and cause verification. "
                f"The increase is unexplained and flagged for human review. {rejection} {comparison_text}"
            )

    return {
        "route": route,
        "week_of": week_text,
        "cost_per_tonne_km": round(float(row["cost_per_tonne_km"]), 2),
        "vs_own_history": own_text,
        "vs_similar_routes": peer_text,
        "flagged": verdict,
        "matched_note_id": note_id,
        "reason": reason,
    }

observed_rows = weekly[weekly["cost_per_tonne_km"].notna()].copy()
output_rows = [create_output_row(row, verified_evidence, nearest_retrieved_notes) for _, row in observed_rows.iterrows()]
output = pd.DataFrame(output_rows, columns=EXPECTED_OUTPUT_COLUMNS)

# Output contract checks.
assert output.columns.tolist() == EXPECTED_OUTPUT_COLUMNS
assert output["route"].notna().all() and output["week_of"].notna().all()
assert set(output["flagged"].unique()).issubset({"Yes", "No", "No (justified)"})
justified_mask = output["flagged"].eq("No (justified)")
assert output.loc[justified_mask, "matched_note_id"].ne("").all()
assert output.loc[~justified_mask, "matched_note_id"].eq("").all()

output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
roundtrip = pd.read_csv(OUTPUT_FILE, keep_default_na=False)
assert roundtrip.columns.tolist() == EXPECTED_OUTPUT_COLUMNS
print("Output saved:", OUTPUT_FILE)
print("Rows:", len(output))
print("Verdict counts:\n", output["flagged"].value_counts())
display(output.head(12))

# %%

input_tokens = int(TOKEN_USAGE["input_tokens"])
output_tokens = int(TOKEN_USAGE["output_tokens"])
paid_equivalent_cost = (
    input_tokens * INPUT_USD_PER_MILLION_TOKENS / 1_000_000
    + output_tokens * OUTPUT_USD_PER_MILLION_TOKENS / 1_000_000
)

cost_log = {
    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "shipment_rows": int(len(shipments_raw)),
    "context_note_count": int(len(notes)),
    "output_rows": int(len(output)),
    "observed_route_weeks_flagged_as_anomalies": int(rows_with_shipments["is_anomaly"].sum()),
    "route_weeks_with_verified_support_note": int(output["flagged"].eq("No (justified)").sum()),
    "embedding_model_local": EMBEDDING_MODEL_NAME,
    "vector_database": "ChromaDB local persistent collection",
    "llm_model": LLM_MODEL_NAME,
    "llm_calls_attempted": int(TOKEN_USAGE["calls_attempted"]),
    "llm_calls_succeeded": int(TOKEN_USAGE["calls_succeeded"]),
    "input_tokens_reported_by_api": input_tokens,
    "output_tokens_reported_by_api": output_tokens,
    "usage_metadata_missing_calls": int(TOKEN_USAGE["usage_metadata_missing_calls"]),
    "llm_api_errors": int(TOKEN_USAGE["api_errors"]),
    "fallback_explanations": int(TOKEN_USAGE["fallback_explanations"]),
    "paid_tier_input_usd_per_million_tokens": INPUT_USD_PER_MILLION_TOKENS,
    "paid_tier_output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION_TOKENS,
    "estimated_paid_tier_equivalent_cost_usd": round(paid_equivalent_cost, 8),
    "pricing_source": PRICING_SOURCE,
    "pricing_checked_on": "2026-10-15",
    "cost_note": "Published paid-tier equivalent; actual API charges may be zero if eligible free-tier access applies. Verify current pricing before submission. Local embeddings are excluded from LLM token counts.",
}
with open(DATA_DIR / "token_cost_log.json", "w", encoding="utf-8") as f:
    json.dump(cost_log, f, indent=2)
display(pd.DataFrame([cost_log]).T.rename(columns={0: "value"}))