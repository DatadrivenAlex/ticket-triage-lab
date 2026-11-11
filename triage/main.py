"""FastAPI support triage demo. This service has no authentication."""
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from triage.classifier import predict

app = FastAPI(title="Ticket Triage Lab", version="0.1.0")
Category = Literal["account", "billing", "technical"]

class TicketInput(BaseModel):
    text: str = Field(min_length=3, max_length=4000)

class ReviewInput(BaseModel):
    category: Category
    note: str = Field(default="", max_length=1000)

def connection():
    db = sqlite3.connect(os.environ.get("TRIAGE_DB", "triage.db"))
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with closing(connection()) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY, text TEXT NOT NULL, predicted_category TEXT NOT NULL,
            confidence REAL NOT NULL, requires_review INTEGER NOT NULL, status TEXT NOT NULL,
            final_category TEXT, review_note TEXT, created_at TEXT NOT NULL, reviewed_at TEXT
        )""")

def item(row):
    result = dict(row)
    result["requires_review"] = bool(result["requires_review"])
    return result

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/tickets", status_code=201)
def create_ticket(ticket: TicketInput):
    if not ticket.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be blank")
    init_db()
    result = predict(ticket.text)
    with closing(connection()) as db, db:
        cur = db.execute("""INSERT INTO tickets (text, predicted_category, confidence,
          requires_review, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)""",
          (ticket.text.strip(), result["category"], result["confidence"],
           result["requires_review"], datetime.now(timezone.utc).isoformat()))
        row = db.execute("SELECT * FROM tickets WHERE id = ?", (cur.lastrowid,)).fetchone()
    return item(row)

@app.get("/tickets")
def list_tickets(status: Literal["pending", "reviewed"] | None = None, limit: int = 50):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be 1..100")
    init_db()
    with closing(connection()) as db:
        if status:
            rows = db.execute("SELECT * FROM tickets WHERE status = ? ORDER BY id DESC LIMIT ?", (status, limit)).fetchall()
        else:
            rows = db.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [item(row) for row in rows]

@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int):
    init_db()
    with closing(connection()) as db:
        row = db.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return item(row)

@app.post("/tickets/{ticket_id}/review")
def review_ticket(ticket_id: int, review: ReviewInput):
    init_db()
    with closing(connection()) as db, db:
        cur = db.execute("""UPDATE tickets SET status = 'reviewed', final_category = ?,
            review_note = ?, reviewed_at = ? WHERE id = ? AND status = 'pending'""",
            (review.category, review.note, datetime.now(timezone.utc).isoformat(), ticket_id))
        row = db.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        if cur.rowcount == 0:
            raise HTTPException(status_code=409, detail="Ticket already reviewed")
        return item(row)
