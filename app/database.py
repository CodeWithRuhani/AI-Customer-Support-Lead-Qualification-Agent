"""
database.py
-----------
DatabaseManager encapsulates ALL SQL access for the application.
Nobody else in the codebase should write raw SQL - this is the single
point of contact with SQLite (demonstrates encapsulation).
"""

import json
import sqlite3
import os
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get(
    "DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "support_agent.db")
)
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "schema.sql")

class DatabaseManager:
    """Encapsulates the SQLite connection and every CRUD operation."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._initialise()

    # ------------------------------------------------------------------
    # connection helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialise(self):
        """Create missing tables/indexes without dropping existing data."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with self._get_conn() as conn:
            if not os.path.exists(SCHEMA_PATH):
                raise FileNotFoundError(f"Database schema not found: {SCHEMA_PATH}")
            with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
                conn.executescript(schema_file.read())
            lead_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(leads)")
            }
            if "score_explanation" not in lead_columns:
                conn.execute(
                    "ALTER TABLE leads ADD COLUMN score_explanation TEXT NOT NULL DEFAULT '[]'"
                )
                logger.info("Added score explanation column to existing leads table")
        logger.info("Database schema ready at %s", self.db_path)

    # ------------------------------------------------------------------
    # customers
    # ------------------------------------------------------------------
    def create_customer(self, name: str, email: str, phone: Optional[str]) -> Dict[str, Any]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO customers (name, email, phone) VALUES (?, ?, ?)
                   ON CONFLICT(email) DO NOTHING""",
                (name, email, phone),
            )
            if cur.rowcount == 0:
                cur.execute("SELECT * FROM customers WHERE email = ?", (email,))
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("Customer insert conflicted but no matching email exists")
                logger.info("Existing customer reused (customer_id=%s)", row["customer_id"])
                return dict(row)
            cur.execute("SELECT * FROM customers WHERE customer_id = ?", (cur.lastrowid,))
            logger.info("Customer created (customer_id=%s)", cur.lastrowid)
            return dict(cur.fetchone())

    def get_customer(self, customer_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_customer_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM customers WHERE email = ?", (email,)).fetchone()
            return dict(row) if row else None

    def list_customers(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM customers ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def update_customer(self, customer_id: int, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {"name", "email", "phone"}
        fields = {key: value for key, value in fields.items() if key in allowed}
        if fields:
            set_clause = ", ".join(f"{key} = ?" for key in fields)
            with self._get_conn() as conn:
                conn.execute(
                    f"UPDATE customers SET {set_clause} WHERE customer_id = ?",
                    [*fields.values(), customer_id],
                )
        return self.get_customer(customer_id)

    def delete_customer(self, customer_id: int) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute(
                "DELETE FROM customers WHERE customer_id = ?", (customer_id,)
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # conversations
    # ------------------------------------------------------------------
    def add_conversation(self, customer_id: int, message: str, intent: str,
                          product_interest: Optional[str]) -> Dict[str, Any]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO conversations (customer_id, message, intent, product_interest)
                   VALUES (?, ?, ?, ?)""",
                (customer_id, message, intent, product_interest),
            )
            cur.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?", (cur.lastrowid,)
            )
            logger.info(
                "Customer conversation stored (customer_id=%s, intent=%s)",
                customer_id,
                intent,
            )
            return dict(cur.fetchone())

    def add_guest_conversation(
        self, message: str, intent: str, product_interest: str, reply: str
    ) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO guest_conversations
                   (message, intent, product_interest, reply) VALUES (?, ?, ?, ?)""",
                (message, intent, product_interest, reply),
            )
        logger.info("Guest conversation stored (intent=%s)", intent)

    def get_conversations_for_customer(self, customer_id: int) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE customer_id = ? ORDER BY timestamp",
                (customer_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # leads
    # ------------------------------------------------------------------
    def upsert_lead(
        self,
        customer_id: int,
        category: str,
        score: int,
        next_action: str,
        explanation: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        explanation_json = json.dumps(explanation)
        with self._get_conn() as conn:
            cur = conn.cursor()
            existing = cur.execute(
                "SELECT lead_id FROM leads WHERE customer_id = ?", (customer_id,)
            ).fetchone()
            if existing:
                cur.execute(
                    """UPDATE leads SET category = ?, score = ?, next_action = ?,
                       score_explanation = ?, updated_at = datetime('now')
                       WHERE customer_id = ?""",
                    (category, score, next_action, explanation_json, customer_id),
                )
                lead_id = existing["lead_id"]
            else:
                cur.execute(
                    """INSERT INTO leads
                       (customer_id, category, score, next_action, score_explanation)
                       VALUES (?, ?, ?, ?, ?)""",
                    (customer_id, category, score, next_action, explanation_json),
                )
                lead_id = cur.lastrowid
            cur.execute("SELECT * FROM leads WHERE lead_id = ?", (lead_id,))
            logger.info(
                "Lead saved (customer_id=%s, lead_id=%s, score=%s, category=%s)",
                customer_id,
                lead_id,
                score,
                category,
            )
            return self._lead_to_dict(cur.fetchone())

    @staticmethod
    def _lead_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        lead = dict(row)
        raw_explanation = lead.pop("score_explanation", "[]")
        try:
            lead["explanation"] = json.loads(raw_explanation)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Lead has invalid stored score explanation (lead_id=%s)",
                lead.get("lead_id"),
            )
            lead["explanation"] = []
        return lead

    def get_lead(self, lead_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM leads WHERE lead_id = ?", (lead_id,)).fetchone()
            return self._lead_to_dict(row) if row else None

    def list_leads(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT l.*, c.name, c.email, c.phone
                   FROM leads l JOIN customers c ON c.customer_id = l.customer_id
                   ORDER BY l.score DESC"""
            ).fetchall()
            return [self._lead_to_dict(row) for row in rows]

    def update_lead(self, lead_id: int, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not fields:
            return self.get_lead(lead_id)
        allowed = {"category", "score", "status", "next_action"}
        fields = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not fields:
            return self.get_lead(lead_id)
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [lead_id]
        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE leads SET {set_clause}, updated_at = datetime('now') WHERE lead_id = ?",
                values,
            )
        return self.get_lead(lead_id)

    def delete_lead(self, lead_id: int) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute("DELETE FROM leads WHERE lead_id = ?", (lead_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # predictions (audit trail)
    # ------------------------------------------------------------------
    def log_prediction(self, lead_id: Optional[int], predicted_intent: str, confidence: float):
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO predictions (lead_id, predicted_intent, confidence)
                   VALUES (?, ?, ?)""",
                (lead_id, predicted_intent, confidence),
            )

    # ------------------------------------------------------------------
    # analytics
    # ------------------------------------------------------------------
    def get_analytics(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            total_customers = conn.execute("SELECT COUNT(*) c FROM customers").fetchone()["c"]
            total_leads = conn.execute("SELECT COUNT(*) c FROM leads").fetchone()["c"]
            by_category = {
                row["category"]: row["c"]
                for row in conn.execute(
                    "SELECT category, COUNT(*) c FROM leads GROUP BY category"
                ).fetchall()
            }
            customer_intents = {
                row["intent"]: row["c"]
                for row in conn.execute(
                    "SELECT intent, COUNT(*) c FROM conversations GROUP BY intent"
                ).fetchall()
            }
            guest_intents = {
                row["intent"]: row["c"]
                for row in conn.execute(
                    "SELECT intent, COUNT(*) c FROM guest_conversations GROUP BY intent"
                ).fetchall()
            }
            by_intent = {
                intent: customer_intents.get(intent, 0) + guest_intents.get(intent, 0)
                for intent in ("Sales", "Support", "Complaint", "General")
                if customer_intents.get(intent, 0) + guest_intents.get(intent, 0) > 0
            }
            avg_score_row = conn.execute("SELECT AVG(score) a FROM leads").fetchone()
            avg_score = round(avg_score_row["a"], 2) if avg_score_row["a"] is not None else 0
            lead_distribution = {
                category: by_category.get(category, 0)
                for category in ("Hot", "Warm", "Cold")
            }
            customer_conversations = conn.execute(
                "SELECT COUNT(*) c FROM conversations"
            ).fetchone()["c"]
            guest_conversations = conn.execute(
                "SELECT COUNT(*) c FROM guest_conversations"
            ).fetchone()["c"]
            total_conversations = customer_conversations + guest_conversations

        return {
            "total_customers": total_customers,
            "total_leads": total_leads,
            "total_conversations": total_conversations,
            "hot_leads": lead_distribution["Hot"],
            "warm_leads": lead_distribution["Warm"],
            "cold_leads": lead_distribution["Cold"],
            "lead_distribution": lead_distribution,
            "sales_enquiries": by_intent.get("Sales", 0),
            "average_lead_score": avg_score,
            "query_distribution_by_intent": by_intent,
        }
