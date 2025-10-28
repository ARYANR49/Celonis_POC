#  Lead Enrichment & Routing Pipeline (Google Gemini + LangChain)


> AI-powered pipeline that enriches inbound business leads from CSV exports.  
> Automatically detects persona type, urgency, intent summary, and routes leads to the correct team.

---

##  Overview

Modern marketing forms generate thousands of leads.  
This pipeline automatically:

- Reads a CSV export (`email`, `job_title`, `comment`)
- Enriches each row using Google Gemini
- Produces structured JSON:
  - urgency (`High`, `Medium`, `Low`)
  - persona_type (`Decision Maker`, `Practitioner`, `Other`)
  - one-sentence summary
- Validates output via schema
- Routes to different teams
- Caches repeated leads
- Captures failures in a Dead Letter Queue (DLQ)

---

##  Features

AI lead enrichment  
Persona & urgency detection  
CSV auto-repair for malformed files  
Retry logic + rate limiting  
JSON Schema validation  
JSONL output for analytics ingestion  
Duplicate caching via SHA-256  
Dead Letter Queue for human review  

---

##  Routing Rules

| Urgency | Persona Type   | Assigned Team    |
|---------|----------------|------------------|
| High    | Decision Maker | Strategic Sales  |
| High    | Practitioner   | Enterprise Sales |
| Medium  | Any            | Sales Development|
| Low     | Any            | Nurture Campaign |
| Unknown | Any            | Manual Triage    |

---

##  Tech Stack

| Component    | Purpose          |
|--------------|------------------|
| Python 3.10+ | Runtime          |
| Google Gemini| LLM inference    |
| LangChain    | Orchestration    |
| pandas       | CSV parsing      |
| jsonschema   | Validation       |
| dotenv       | Config management|
| logging      | Observability    |

---

##  Project Structure

├── Main.py # main script
├── leads_Sheet.csv # example input
├── enriched_leads.jsonl # successful enrichments
├── dlq.jsonl # rows that failed
├── cache.json # dedupe cache
├── .env # configuration
├── requirements.txt # dependencies
└── README.md # documentation

##  Installation

git clone https://github.com/ARYANR49/Celonis_POC.git
cd lead-enrichment-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

##  Run the Pipeline

python main.py

## Sample Input CSV

email,job_title,comment
carla.dunn@globaltech.com,Chief Information Officer,We need to improve our order management efficiency. My team is available for a technical deep dive next week.
david.lee@innovateinc.com,Senior Process Analyst,My manager asked me to find a solution for mapping our accounts payable process. Need a demo ASAP.
lisa.chen@healthsys.org,Head of Patient Services,Our patient intake process has bottlenecks. Looking urgently for a solution.

## Sample Output JSONL

{
  "id": "4d8e3e95-0544-4e88-b5fb-31047564e6bb",
  "email": "carla.dunn@globaltech.com",
  "job_title": "Chief Information Officer",
  "comment": "We need to improve our order management efficiency. My team is available for a technical deep dive next week.",
  "enrichment": {
    "urgency": "Medium",
    "persona_type": "Decision Maker",
    "summary": "Follow-up requested from interested Decision Maker."
  },
  "assigned_team": "Sales Development",
  "meta": {
    "processed_at": 1761289814.063334,
    "cache_key": "..."
  }
}

## Future Improvements

1) Add a Simple Web UI:

 •	Provide a friendly interface for uploading lead CSV files

 •	Allow users to download enriched outputs and DLQ items

 •	Provide filters (urgency, persona, assigned team)

2) Deploy as a Cloud Service

3) Analytics Dashboard:

 •	Visualize urgency distribution, persona trends, DLQ rate, cost/lead

 •	Identify patterns (e.g., most urgent industries)

4) Add authentication for UI/API access
