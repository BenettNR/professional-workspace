from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"


class TransactionCategory(str, Enum):
    GROCERIES = "groceries"
    DINING = "dining"
    TRANSPORT = "transport"
    UTILITIES = "utilities"
    HEALTH = "health"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    TRAVEL = "travel"
    FINANCE = "finance"
    GOVERNMENT = "government"
    SALARY = "salary"
    TRANSFER = "transfer"
    OTHER = "other"


class Transaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    date: date
    description_raw: str
    description_clean: Optional[str] = None
    amount: Decimal
    type: TransactionType
    balance: Optional[Decimal] = None
    category: Optional[TransactionCategory] = None
    merchant: Optional[str] = None
    reference: Optional[str] = None


class StatementMetadata(BaseModel):
    account_number_masked: Optional[str] = None
    bsb_masked: Optional[str] = None
    account_holder_masked: Optional[str] = None
    bank_name: Optional[str] = None
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    opening_balance: Optional[Decimal] = None
    closing_balance: Optional[Decimal] = None


class CategorySummary(BaseModel):
    category: TransactionCategory
    total: Decimal
    count: int
    percentage: float


class StatementSummary(BaseModel):
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    total_credits: Decimal = Decimal("0")
    total_debits: Decimal = Decimal("0")
    net_cash_flow: Decimal = Decimal("0")
    transaction_count: int = 0
    largest_debit: Optional[Transaction] = None
    largest_credit: Optional[Transaction] = None
    category_breakdown: list[CategorySummary] = Field(default_factory=list)
    recurring_payments: list[Transaction] = Field(default_factory=list)
    narrative: Optional[str] = None


class ProcessingResult(BaseModel):
    statement_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: StatementMetadata
    transactions: list[Transaction]
    summary: Optional[StatementSummary] = None
    pii_entities_masked: int = 0
    pii_audit_log: list[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class RawDocument:
    """Intermediate representation after parsing, before transaction extraction."""
    source_file: str
    format: str
    text: str = ""
    tables: list = field(default_factory=list)  # list[pd.DataFrame]
