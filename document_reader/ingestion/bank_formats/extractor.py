"""
Generic transaction extractor that handles common Australian bank statement formats.
Works on RawDocument → (list[Transaction], StatementMetadata).
"""

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional

import pandas as pd
from dateutil import parser as dateutil_parser

from extraction.models import RawDocument, StatementMetadata, Transaction, TransactionType

# Column name aliases used by various AU banks
_DATE_COLS = {"date", "transaction date", "value date", "posted date", "txn date", "trans date"}
_DESC_COLS = {"description", "details", "particulars", "narrative", "transaction", "memo", "reference"}
_DEBIT_COLS = {"debit", "debits", "withdrawal", "withdrawals", "dr"}
_CREDIT_COLS = {"credit", "credits", "deposit", "deposits", "cr"}
_AMOUNT_COLS = {"amount", "net amount", "transaction amount"}
_BALANCE_COLS = {"balance", "running balance", "closing balance"}

# Regex for finding transaction lines in plain text (PDF/OCR fallback)
_TXN_LINE_RE = re.compile(
    r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}\s+\w{3}\s+\d{4})"  # date
    r"\s+(.+?)\s+"  # description
    r"([\-]?\$?[\d,]+\.\d{2})"  # amount
    r"(?:\s+([\-]?\$?[\d,]+\.\d{2}))?",  # optional balance
    re.IGNORECASE,
)

_ACCOUNT_RE = re.compile(r"\b(\d{6,9})\b")
_BSB_RE = re.compile(r"\b(\d{3}-\d{3})\b")
_BALANCE_HEADER_RE = re.compile(
    r"(?:opening|closing|brought forward|carried forward)\s+balance[:\s]+([\-]?\$?[\d,]+\.\d{2})",
    re.IGNORECASE,
)
_PERIOD_RE = re.compile(
    r"(?:statement period|period)[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})"
    r"\s*(?:to|-)\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
    re.IGNORECASE,
)


def _normalise_col(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def _find_col(df: pd.DataFrame, aliases: set[str]) -> Optional[str]:
    for col in df.columns:
        if _normalise_col(col) in aliases:
            return col
    return None


def _parse_amount(value: str) -> Optional[Decimal]:
    if not value or str(value).strip() in {"", "nan", "None", "-"}:
        return None
    cleaned = re.sub(r"[,$\s]", "", str(value))
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _parse_date(value: str) -> Optional[date]:
    if not value or str(value).strip() in {"", "nan", "None"}:
        return None
    try:
        return dateutil_parser.parse(str(value), dayfirst=True).date()
    except Exception:
        return None


def _extract_from_dataframe(df: pd.DataFrame) -> list[Transaction]:
    transactions: list[Transaction] = []

    date_col = _find_col(df, _DATE_COLS)
    desc_col = _find_col(df, _DESC_COLS)
    debit_col = _find_col(df, _DEBIT_COLS)
    credit_col = _find_col(df, _CREDIT_COLS)
    amount_col = _find_col(df, _AMOUNT_COLS)
    balance_col = _find_col(df, _BALANCE_COLS)

    if not date_col or not desc_col:
        return transactions

    for _, row in df.iterrows():
        txn_date = _parse_date(str(row[date_col]))
        if not txn_date:
            continue

        description = str(row[desc_col]).strip()
        if not description or description == "nan":
            continue

        balance = _parse_amount(str(row[balance_col])) if balance_col else None

        # Two-column debit/credit layout
        if debit_col and credit_col:
            debit = _parse_amount(str(row[debit_col]))
            credit = _parse_amount(str(row[credit_col]))
            if debit and debit != Decimal("0"):
                amount, txn_type = abs(debit), TransactionType.DEBIT
            elif credit and credit != Decimal("0"):
                amount, txn_type = abs(credit), TransactionType.CREDIT
            else:
                continue
        elif amount_col:
            raw = _parse_amount(str(row[amount_col]))
            if raw is None:
                continue
            if raw < 0:
                amount, txn_type = abs(raw), TransactionType.DEBIT
            else:
                amount, txn_type = raw, TransactionType.CREDIT
        else:
            continue

        transactions.append(
            Transaction(
                date=txn_date,
                description_raw=description,
                amount=amount,
                type=txn_type,
                balance=balance,
            )
        )

    return transactions


def _extract_from_text(text: str) -> list[Transaction]:
    transactions: list[Transaction] = []
    for match in _TXN_LINE_RE.finditer(text):
        txn_date = _parse_date(match.group(1))
        if not txn_date:
            continue
        description = match.group(2).strip()
        amount_raw = _parse_amount(match.group(3))
        if amount_raw is None:
            continue
        balance = _parse_amount(match.group(4)) if match.group(4) else None

        if amount_raw < 0:
            amount, txn_type = abs(amount_raw), TransactionType.DEBIT
        else:
            amount, txn_type = amount_raw, TransactionType.CREDIT

        transactions.append(
            Transaction(
                date=txn_date,
                description_raw=description,
                amount=amount,
                type=txn_type,
                balance=balance,
            )
        )
    return transactions


def _extract_metadata(text: str) -> StatementMetadata:
    metadata = StatementMetadata()

    bsb_match = _BSB_RE.search(text)
    if bsb_match:
        metadata.bsb_masked = bsb_match.group(1)

    period_match = _PERIOD_RE.search(text)
    if period_match:
        metadata.period_start = _parse_date(period_match.group(1))
        metadata.period_end = _parse_date(period_match.group(2))

    for m in _BALANCE_HEADER_RE.finditer(text):
        context = text[max(0, m.start() - 20) : m.start()].lower()
        amount = _parse_amount(m.group(1))
        if "open" in context or "brought" in context:
            metadata.opening_balance = amount
        elif "clos" in context or "carried" in context:
            metadata.closing_balance = amount

    return metadata


def extract_statement(
    doc: RawDocument,
) -> tuple[list[Transaction], StatementMetadata]:
    transactions: list[Transaction] = []

    for df in doc.tables:
        extracted = _extract_from_dataframe(df)
        if extracted:
            transactions.extend(extracted)

    if not transactions and doc.text:
        transactions = _extract_from_text(doc.text)

    metadata = _extract_metadata(doc.text)

    if transactions:
        dates = [t.date for t in transactions]
        if not metadata.period_start:
            metadata.period_start = min(dates)
        if not metadata.period_end:
            metadata.period_end = max(dates)

    return transactions, metadata
