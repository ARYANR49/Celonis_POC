#!/usr/bin/env python3
import os
import json
import uuid
import time
import hashlib
import logging
import io
import pandas as pd
from jsonschema import validate, ValidationError
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# ------------------ CONFIG ------------------

load_dotenv()
# support both env names for convenience
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI API key not found in environment (.env) — set GEMINI_API_KEY or GOOGLE_API_KEY")

MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash")
INPUT_CSV = os.getenv("INPUT_CSV", "leads_Sheet.csv")
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "enriched_leads.jsonl")
DLQ_JSONL = os.getenv("DLQ_JSONL", "dlq.jsonl")
CACHE_FILE = os.getenv("CACHE_FILE", "cache.json")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", "0.3"))

# ------------------ LOGGING ------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------ SCHEMA ------------------

SCHEMA = {
    "type": "object",
    "required": ["urgency", "persona_type", "summary"],
    "properties": {
        "urgency": {"type": "string", "enum": ["High", "Medium", "Low"]},
        "persona_type": {"type": "string", "enum": ["Decision Maker", "Practitioner", "Other"]},
        "summary": {"type": "string", "minLength": 1, "maxLength": 300},
    },
    "additionalProperties": False,
}

# ------------------ PROMPT ------------------

PROMPT_TEMPLATE = """
You are a JSON-only assistant.
Given the following job title and comment from a business lead,
output ONLY valid JSON with keys: "urgency", "persona_type", "summary".

Rules:
- urgency: one of "High", "Medium", "Low".
  * High = demo/pricing request, urgent contact.
  * Medium = interested, follow-up requested.
  * Low = vague/research inquiries.
- persona_type: one of "Decision Maker", "Practitioner", "Other".
- summary: concise one-sentence summary of the intent.

Return ONLY the JSON object and nothing else.

job_title: {JOB_TITLE}
comment: {COMMENT}
"""

# ------------------ MODEL ------------------

# Initialize LangChain Google Gemini wrapper
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=API_KEY,
    temperature=0.0,
    max_output_tokens=512
)

# ------------------ HELPERS ------------------

def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_cache(path: str):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ------------------ ROBUST CSV READER ------------------

def robust_read_csv(path: str) -> pd.DataFrame:
    # Try normal read first
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", skipinitialspace=True)
    except Exception as e:
        logger.warning("Initial read_csv failed: %s. Falling back to tolerant python engine.", e)
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", engine="python")

    df = df.fillna("")

    # If only one column and the column header contains commas/quotes, attempt cleaning
    if len(df.columns) == 1:
        colname = str(df.columns[0])
        if (',' in colname) or ('"' in colname) or (colname.count('"') > 0):
            logger.info("Detected single-column CSV with embedded commas/quotes. Applying cleaning pass.")
            with open(path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()

            cleaned_lines = []
            for line in raw_lines:
                ln = line.rstrip("\n\r")
                ln2 = ln.replace('""', '"')
                if ln2.startswith('"') and ln2.endswith('"'):
                    ln2 = ln2[1:-1]
                cleaned_lines.append(ln2 + "\n")

            cleaned_text = "".join(cleaned_lines)
            try:
                df2 = pd.read_csv(io.StringIO(cleaned_text), dtype=str)
                df2 = df2.fillna("")
                clean_cols = [c.strip().strip('"').strip("'").strip() for c in df2.columns.astype(str)]
                df2.columns = clean_cols
                logger.info("Cleaned CSV parsed successfully. Columns: %s", df2.columns.tolist())
                return df2
            except Exception as e:
                logger.error("Failed to parse cleaned CSV: %s", e)
                # fallback to original df

    # Normalize column names for standard case
    clean_cols = [c.strip().strip('"').strip("'").strip() for c in df.columns.astype(str)]
    df.columns = clean_cols
    return df

# ------------------ NORMALIZATION & VALIDATION ------------------

def normalize_and_validate(raw_output: dict) -> dict:
    urgency = str(raw_output.get("urgency", "")).strip()
    persona = str(raw_output.get("persona_type", "")).strip()
    summary = str(raw_output.get("summary", "")).strip()

    u_map = {"urgent": "High", "high": "High", "medium": "Medium", "low": "Low", "informational": "Low"}
    p_map = {
        "decision-maker": "Decision Maker", "decision maker": "Decision Maker",
        "practitioner": "Practitioner", "engineer": "Practitioner",
        "student": "Other", "researcher": "Other", "other": "Other"
    }

    u = u_map.get(urgency.lower(), urgency if urgency in ["High","Medium","Low"] else "Medium")
    p = p_map.get(persona.lower(), persona if persona in ["Decision Maker","Practitioner","Other"] else "Other")

    normalized = {
        "urgency": u,
        "persona_type": p,
        "summary": summary[:300]
    }

    validate(instance=normalized, schema=SCHEMA)
    return normalized

# ------------------ ROUTING ------------------

def assign_team(urgency: str, persona_type: str) -> str:
    if urgency == "High" and persona_type == "Decision Maker":
        return "Strategic Sales"
    if urgency == "High" and persona_type == "Practitioner":
        return "Enterprise Sales"
    if urgency == "Medium":
        return "Sales Development"
    if urgency == "Low":
        return "Nurture Campaign"
    return "Manual Triage"

# ------------------ ENRICHMENT ------------------

def enrich_lead(job_title: str, comment: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(JOB_TITLE=(job_title or "").strip(), COMMENT=(comment or "").strip()[:1000])
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(RATE_LIMIT_SECONDS)
            messages = [
                SystemMessage(content="Return only valid JSON, no extra text."),
                HumanMessage(content=prompt)
            ]
            resp = llm.invoke(messages)
            text = resp.content.strip()

            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object found in LLM response.")
            parsed = json.loads(text[start:end+1])
            validated = normalize_and_validate(parsed)
            return validated
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.warning("Parse/validation issue attempt %d: %s", attempt, e)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(1.5 ** attempt)
        except Exception as e:
            logger.warning("LLM call attempt %d failed: %s", attempt, e)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(1.5 ** attempt)

# ------------------ MAIN ------------------

def main():
    if not os.path.exists(INPUT_CSV):
        logger.error("Input CSV not found: %s", INPUT_CSV)
        return

    df = robust_read_csv(INPUT_CSV)
    logger.info("CSV columns detected: %s", df.columns.tolist())
    logger.info("Sample rows:\n%s", df.head(5).to_string(index=False))

    expected = {"email", "job_title", "comment"}
    missing = expected - set(df.columns)
    if missing:
        logger.error("Missing expected columns %s in CSV. Aborting.", missing)
        return

    cache = load_cache(CACHE_FILE)
    out_f = open(OUTPUT_JSONL, "a", encoding="utf-8")
    dlq_f = open(DLQ_JSONL, "a", encoding="utf-8")

    for i, row in df.iterrows():
        email = (row.get("email") or "").strip()
        job_title = (row.get("job_title") or "").strip()
        comment = (row.get("comment") or "").strip()

        if not (email or job_title or comment):
            logger.warning("Row %d empty or all fields blank: %s", i, row.to_dict())
            dlq_f.write(json.dumps({
                "id": str(uuid.uuid4()),
                "email": email,
                "job_title": job_title,
                "comment": comment,
                "error": "Empty row fields after parsing",
                "row_index": i,
                "raw": row.to_dict()
            }, ensure_ascii=False) + "\n")
            dlq_f.flush()
            continue

        key = fingerprint(job_title + "\n" + comment)
        if key in cache:
            enrichment = cache[key]
            logger.info("Cache hit row %d (%s)", i, job_title or email)
        else:
            try:
                enrichment = enrich_lead(job_title, comment)
                cache[key] = enrichment
                save_cache(CACHE_FILE, cache)
                logger.info("Enriched row %d successfully", i)
            except Exception as e:
                logger.error("Enrichment failed for row %d: %s", i, e)
                dlq_f.write(json.dumps({
                    "id": str(uuid.uuid4()),
                    "email": email,
                    "job_title": job_title,
                    "comment": comment,
                    "error": str(e),
                    "row_index": i
                }, ensure_ascii=False) + "\n")
                dlq_f.flush()
                continue

        assigned = assign_team(enrichment["urgency"], enrichment["persona_type"])
        record = {
            "id": str(uuid.uuid4()),
            "email": email,
            "job_title": job_title,
            "comment": comment,
            "enrichment": enrichment,
            "assigned_team": assigned,
            "meta": {"processed_at": time.time(), "cache_key": key}
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        logger.info("Row %d routed to %s", i, assigned)

    out_f.close()
    dlq_f.close()
    save_cache(CACHE_FILE, cache)
    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
