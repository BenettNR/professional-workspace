"""
Analytics aggregation tests — pure logic, no I/O.
"""

from datetime import date
from decimal import Decimal

import pytest

from analytics.aggregator import aggregate, _detect_recurring
from extraction.models import Transaction, TransactionCategory, TransactionType


def _txn(amount: str, txn_type: TransactionType, desc: str = "TEST", category=None) -> Transaction:
    return Transaction(
        date=date(2024, 3, 1),
        description_raw=desc,
        amount=Decimal(amount),
        type=txn_type,
        category=category,
    )


DEBIT = TransactionType.DEBIT
CREDIT = TransactionType.CREDIT


def test_total_credits_and_debits():
    txns = [
        _txn("100.00", CREDIT),
        _txn("200.00", CREDIT),
        _txn("50.00", DEBIT),
    ]
    stats = aggregate(txns)
    assert stats["total_credits"] == Decimal("300.00")
    assert stats["total_debits"] == Decimal("50.00")
    assert stats["net_cash_flow"] == Decimal("250.00")


def test_largest_debit_and_credit():
    txns = [
        _txn("10.00", DEBIT),
        _txn("999.00", DEBIT),
        _txn("5000.00", CREDIT),
    ]
    stats = aggregate(txns)
    assert stats["largest_debit"].amount == Decimal("999.00")
    assert stats["largest_credit"].amount == Decimal("5000.00")


def test_category_breakdown_ordering():
    txns = [
        _txn("10.00", DEBIT, category=TransactionCategory.DINING),
        _txn("500.00", DEBIT, category=TransactionCategory.GROCERIES),
        _txn("5.00", DEBIT, category=TransactionCategory.DINING),
    ]
    stats = aggregate(txns)
    breakdown = stats["category_breakdown"]
    assert breakdown[0].category == TransactionCategory.GROCERIES
    assert breakdown[1].category == TransactionCategory.DINING
    assert breakdown[1].count == 2


def test_recurring_detection():
    txns = [
        Transaction(date=date(2024, 1, 1), description_raw="NETFLIX", amount=Decimal("19.99"), type=DEBIT),
        Transaction(date=date(2024, 2, 1), description_raw="NETFLIX", amount=Decimal("19.99"), type=DEBIT),
        Transaction(date=date(2024, 3, 1), description_raw="NETFLIX", amount=Decimal("19.99"), type=DEBIT),
        Transaction(date=date(2024, 1, 15), description_raw="ONE OFF PURCHASE", amount=Decimal("55.00"), type=DEBIT),
    ]
    recurring = _detect_recurring(txns, min_occurrences=2)
    assert len(recurring) == 1
    assert recurring[0].description_raw == "NETFLIX"


def test_empty_transactions():
    stats = aggregate([])
    assert stats["total_credits"] == Decimal("0")
    assert stats["total_debits"] == Decimal("0")
    assert stats["transaction_count"] == 0
