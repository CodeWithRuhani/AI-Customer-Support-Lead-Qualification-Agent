"""
models.py
---------
Pydantic request/response schemas used for FastAPI validation.
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Dict, Any, Literal


class CustomerCreate(BaseModel):
    name: str = Field(..., min_length=1, example="Priya Sharma")
    email: EmailStr = Field(..., example="priya@example.com")
    phone: Optional[str] = Field(None, example="+91-9876543210")


class CustomerOut(BaseModel):
    customer_id: int
    name: str
    email: str
    phone: Optional[str]
    created_at: str


class CustomerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None


class ChatRequest(BaseModel):
    customer_id: int = Field(..., example=1)
    message: str = Field(
        ..., min_length=1, max_length=2000,
        example="I'd like a quote for your enterprise plan",
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be blank")
        return value


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    customer_id: Optional[int] = Field(None, ge=1)
    context: List[str] = Field(default_factory=list, max_length=8)
    history: List[ChatHistoryMessage] = Field(default_factory=list, max_length=8)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be blank")
        return value

    @field_validator("context")
    @classmethod
    def validate_context(cls, value: List[str]) -> List[str]:
        if any(len(item) > 500 for item in value):
            raise ValueError("Each context message must be 500 characters or fewer")
        return value


class ChatMessageResponse(BaseModel):
    reply: str
    intent: str
    confidence: float
    product_interest: str
    lead: Optional[Dict[str, Any]] = None
    customer_id: Optional[int] = None
    answer_source: Literal["ollama", "local"]


class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float
    product_interest: str
    lead: Optional[Dict[str, Any]] = None


class LeadUpdate(BaseModel):
    category: Optional[str] = Field(None, pattern="^(Hot|Warm|Cold)$")
    score: Optional[int] = Field(None, ge=0, le=100)
    status: Optional[str] = Field(None, pattern="^(New|Contacted|Qualified|Converted|Lost)$")
    next_action: Optional[str] = None


class AnalyticsResponse(BaseModel):
    total_customers: int
    total_leads: int
    total_conversations: int
    hot_leads: int
    warm_leads: int
    cold_leads: int
    lead_distribution: Dict[str, int]
    sales_enquiries: int
    average_lead_score: float
    query_distribution_by_intent: Dict[str, int]
