"""
main.py
-------
FastAPI application wiring together DatabaseManager, AIClassifier and
LeadScorer behind REST endpoints. Run with:

    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for Swagger UI.
"""

import logging
import os
import sqlite3
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html"
)

from app.chat_assistant import ChatAssistant
from app.database import DatabaseManager
from app.ai_classifier import AIClassifier
from app.lead_scorer import LeadScorer
from app.models import (
    CustomerCreate, CustomerOut, CustomerUpdate, ChatRequest, ChatResponse,
    ChatMessageRequest, ChatMessageResponse,
    LeadUpdate, AnalyticsResponse,
)

app = FastAPI(
    title="AI Customer Support & Lead Qualification Agent",
    description="Prototype: classifies customer queries, scores leads, stores everything in SQL.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Singletons - instantiated once at startup (dependency-injection-lite)
db = DatabaseManager()
classifier = AIClassifier()
scorer = LeadScorer()
assistant = ChatAssistant(classifier)


@app.get("/", include_in_schema=False)
def root():
    logger.info("Serving frontend/index.html")
    return FileResponse(FRONTEND_PATH, media_type="text/html")


@app.get("/health", tags=["Health"])
def health_check():
    logger.info("Health check requested")
    return {"status": "ok", "service": "AI Customer Support & Lead Qualification Agent"}


# ----------------------------------------------------------------------
# Customers
# ----------------------------------------------------------------------
@app.post("/customers", response_model=CustomerOut, tags=["Customers"])
def create_customer(payload: CustomerCreate):
    """Register a new customer (idempotent on email)."""
    logger.info("Customer registration requested")
    customer = db.create_customer(payload.name, payload.email, payload.phone)
    return customer


@app.get("/customers", tags=["Customers"])
def list_customers():
    return db.list_customers()


@app.get("/customers/{customer_id}", response_model=CustomerOut, tags=["Customers"])
def get_customer(customer_id: int):
    customer = db.get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    logger.info("Customer retrieved (customer_id=%s)", customer_id)
    return customer


@app.put("/customers/{customer_id}", response_model=CustomerOut, tags=["Customers"])
def update_customer(customer_id: int, payload: CustomerUpdate):
    existing = db.get_customer(customer_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Customer not found")
    fields = payload.model_dump(exclude_unset=True)
    if any(fields.get(field, existing[field]) is None for field in ("name", "email")):
        raise HTTPException(
            status_code=422, detail="Customer name and email cannot be null"
        )
    try:
        updated = db.update_customer(customer_id, fields)
    except sqlite3.IntegrityError as exc:
        if "UNIQUE constraint failed: customers.email" in str(exc):
            raise HTTPException(
                status_code=409, detail="A customer with this email already exists"
            ) from exc
        raise
    return updated


@app.delete("/customers/{customer_id}", tags=["Customers"])
def delete_customer(customer_id: int):
    logger.info("Customer deletion requested (customer_id=%s)", customer_id)
    if not db.delete_customer(customer_id):
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"status": "deleted", "customer_id": customer_id}


# ----------------------------------------------------------------------
# Chat / query processing + lead qualification
# ----------------------------------------------------------------------
@app.post("/chat/message", response_model=ChatMessageResponse, tags=["Chat"])
def reply_to_chat_message(payload: ChatMessageRequest):
    """Classify a message, reply from known facts, and persist the interaction."""
    message = payload.message.strip()
    logger.info(
        "Chat message received (customer_id=%s)",
        payload.customer_id if payload.customer_id is not None else "guest",
    )
    history = [turn.model_dump() for turn in payload.history]
    result = assistant.respond(message, payload.context, history)
    lead = None

    if not result["blocked"]:
        if payload.customer_id is None:
            db.add_guest_conversation(
                message,
                result["intent"],
                result["product_interest"],
                result["reply"],
            )
        else:
            customer = db.get_customer(payload.customer_id)
            if not customer:
                raise HTTPException(status_code=404, detail="Customer not found")
            lead = _save_customer_conversation(
                payload.customer_id, customer, message, result
            )

    return ChatMessageResponse(
        reply=result["reply"],
        intent=result["intent"],
        confidence=result["confidence"],
        product_interest=result["product_interest"],
        lead=lead,
        customer_id=payload.customer_id,
        answer_source=result["answer_source"],
    )


def _save_customer_conversation(
    customer_id: int,
    customer: Dict[str, Any],
    message: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info("Persisting classified customer conversation (customer_id=%s)", customer_id)
    previous_messages = db.get_conversations_for_customer(customer_id)
    db.add_conversation(
        customer_id,
        message,
        result["intent"],
        result["product_interest"],
    )

    scoring = scorer.score(
        intent=result["intent"],
        confidence=result["confidence"],
        message=message,
        product_interest=result["product_interest"],
        has_phone=bool(customer.get("phone")),
        previous_message_count=len(previous_messages),
    )
    lead = db.upsert_lead(
        customer_id=customer_id,
        category=scoring["category"],
        score=scoring["score"],
        next_action=scoring["next_action"],
        explanation=scoring["explanation"],
    )
    lead["explanation"] = scoring["explanation"]
    db.log_prediction(lead.get("lead_id"), result["intent"], result["confidence"])
    return lead


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def process_chat(payload: ChatRequest):
    """Classify and answer a customer query, then qualify and store the lead."""
    customer = db.get_customer(payload.customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found. Register them first via POST /customers.")

    result = assistant.respond(payload.message)
    lead = None
    if not result["blocked"]:
        lead = _save_customer_conversation(
            payload.customer_id, customer, payload.message, result
        )
    return ChatResponse(
        reply=result["reply"],
        intent=result["intent"],
        confidence=result["confidence"],
        product_interest=result["product_interest"],
        lead=lead,
    )


# ----------------------------------------------------------------------
# Leads (CRUD)
# ----------------------------------------------------------------------
@app.get("/leads", tags=["Leads"])
def get_leads():
    return db.list_leads()


@app.get("/leads/{lead_id}", tags=["Leads"])
def get_lead(lead_id: int):
    lead = db.get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead


@app.put("/leads/{lead_id}", tags=["Leads"])
def update_lead(lead_id: int, payload: LeadUpdate):
    existing = db.get_lead(lead_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Lead not found")
    logger.info("Lead update requested (lead_id=%s)", lead_id)
    updated = db.update_lead(lead_id, payload.model_dump(exclude_unset=True))
    return updated


@app.delete("/leads/{lead_id}", tags=["Leads"])
def delete_lead(lead_id: int):
    logger.info("Lead deletion requested (lead_id=%s)", lead_id)
    deleted = db.delete_lead(lead_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Lead not found")
    return {"status": "deleted", "lead_id": lead_id}


# ----------------------------------------------------------------------
# Conversation history
# ----------------------------------------------------------------------
@app.get("/customers/{customer_id}/conversations", tags=["Chat"])
def get_conversation_history(customer_id: int):
    customer = db.get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return db.get_conversations_for_customer(customer_id)


# ----------------------------------------------------------------------
# Analytics
# ----------------------------------------------------------------------
@app.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics():
    logger.info("Analytics requested")
    return db.get_analytics()
