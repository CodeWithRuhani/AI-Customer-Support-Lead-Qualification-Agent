-- ============================================================
-- AI Customer Support & Lead Qualification Agent
-- Relational schema (SQLite dialect). All DDL is additive and safe to
-- execute during application startup without deleting existing data.
-- ============================================================

PRAGMA foreign_keys = ON;

-- Customers: master record of every person who has contacted us
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    phone         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Conversations: every inbound query + the AI's classification of it
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL,
    message         TEXT NOT NULL,
    intent          TEXT NOT NULL CHECK (intent IN ('Sales','Support','Complaint','General')),
    product_interest TEXT,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Leads: one row per customer representing their current sales-qualification state
CREATE TABLE IF NOT EXISTS leads (
    lead_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   INTEGER NOT NULL,
    category      TEXT NOT NULL CHECK (category IN ('Hot','Warm','Cold')),
    score         INTEGER NOT NULL CHECK (score BETWEEN 0 AND 100),
    status        TEXT NOT NULL DEFAULT 'New' CHECK (status IN ('New','Contacted','Qualified','Converted','Lost')),
    next_action   TEXT,
    score_explanation TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Predictions: audit trail of every AI intent prediction made (confidence etc.)
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id           INTEGER,
    predicted_intent  TEXT NOT NULL,
    confidence        REAL NOT NULL,
    predicted_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (lead_id) REFERENCES leads(lead_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_customer ON conversations(customer_id);
CREATE INDEX IF NOT EXISTS idx_leads_customer ON leads(customer_id);
CREATE INDEX IF NOT EXISTS idx_leads_category ON leads(category);

-- Messages from visitors who choose not to provide personal information.
CREATE TABLE IF NOT EXISTS guest_conversations (
    guest_conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    message               TEXT NOT NULL,
    intent                TEXT NOT NULL CHECK (intent IN ('Sales','Support','Complaint','General')),
    product_interest      TEXT,
    reply                 TEXT NOT NULL,
    timestamp             TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_guest_conversations_timestamp
    ON guest_conversations(timestamp);

-- ----------------------------------------------------------------
-- Example analytical queries (joins) used by GET /analytics
-- ----------------------------------------------------------------

-- Category-wise lead distribution
-- SELECT category, COUNT(*) AS total FROM leads GROUP BY category;

-- Customers with their latest lead score (join)
-- SELECT c.name, c.email, l.score, l.category
-- FROM customers c
-- JOIN leads l ON c.customer_id = l.customer_id
-- ORDER BY l.score DESC;

-- Full conversation + lead history for one customer (multi-table join)
-- SELECT c.name, conv.message, conv.intent, l.score, l.category
-- FROM customers c
-- JOIN conversations conv ON conv.customer_id = c.customer_id
-- LEFT JOIN leads l ON l.customer_id = c.customer_id
-- WHERE c.customer_id = ?;
