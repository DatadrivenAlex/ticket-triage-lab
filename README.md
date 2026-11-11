# Ticket Triage Lab

A Python backend and machine-learning demo for routing synthetic support tickets. It classifies text into account, billing or technical categories using TF-IDF and multinomial Naive Bayes, then lets a reviewer correct the suggestion through an API. Predictions and reviewer decisions are stored in SQLite.

## How it works

1. The service trains a small local model from synthetic examples when first needed; no model download or external API is required.
2. `POST /tickets` records the suggestion and its model score. Scores below 0.55 are flagged for extra attention.
3. A reviewer submits the final category using `POST /tickets/{id}/review`. The original prediction remains visible alongside the decision.
4. `GET /tickets?status=pending` lists the review queue; `GET /tickets/{id}` retrieves a ticket.

All tickets await a human decision, including suggestions above the threshold. The score is an uncalibrated model probability, not a guarantee of correctness.

## Run locally

Python 3.11+ is required. In a virtual environment:

```bash
pip install -e . pytest httpx
uvicorn triage.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API. Example:

```bash
curl -X POST http://127.0.0.1:8000/tickets -H 'Content-Type: application/json' -d '{"text":"My invoice was charged twice"}'
curl -X POST http://127.0.0.1:8000/tickets/1/review -H 'Content-Type: application/json' -d '{"category":"billing","note":"Reviewed"}'
pytest -q
```

Set `TRIAGE_DB` to choose the SQLite file; the default is `triage.db` in the working directory. The database is created on first use.

## Limitations

This is a small educational demo trained only on English synthetic examples, not a production classifier. There is no measured real-world accuracy or confidence calibration. It has no authentication, tenant isolation, rate limiting, encryption, migrations or production deployment setup; run locally and do not enter real customer data. Review is a one-time transition, not a full audit/event system. The initial training samples are not an independent evaluation set.
