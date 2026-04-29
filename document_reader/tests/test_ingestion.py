"""
Ingestion + extraction tests using in-memory CSV data.
No file system or external dependencies required beyond pandas.
"""

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ingestion.bank_formats.extractor import extract_statement, _parse_amount, _parse_date
from extraction.models import RawDocument, TransactionType


def _make_csv_doc(csv_content: str) -> RawDocument:
    df = pd.read_csv(io.StringIO(csv_content), dtype=str)
    return RawDocument(source_file="test.csv", format="csv", tables=[df])


def test_parse_amount_positive():
    assert _parse_amount("1,234.56") is not None
    from decimal import Decimal
    assert _parse_amount("1,234.56") == Decimal("1234.56")


def test_parse_amount_negative():
    from decimal import Decimal
    assert _parse_amount("-50.00") == Decimal("-50.00")


def test_parse_amount_empty():
    assert _parse_amount("") is None
    assert _parse_amount("nan") is None


def test_parse_date_au_format():
    d = _parse_date("15/03/2024")
    assert d is not None
    assert d.day == 15
    assert d.month == 3


def test_parse_date_text_format():
    d = _parse_date("15 Mar 2024")
    assert d is not None
    assert d.month == 3


def test_extract_from_csv_debit_credit_columns():
    csv = """Date,Description,Debit,Credit,Balance
15/03/2024,WOOLWORTHS SYDNEY,45.60,,1234.40
16/03/2024,SALARY PAYMENT,,3500.00,4734.40
17/03/2024,NETFLIX,19.99,,4714.41
"""
    doc = _make_csv_doc(csv)
    transactions, metadata = extract_statement(doc)
    assert len(transactions) == 3
    assert transactions[0].type == TransactionType.DEBIT
    assert transactions[1].type == TransactionType.CREDIT
    from decimal import Decimal
    assert transactions[1].amount == Decimal("3500.00")


def test_extract_from_csv_single_amount_column():
    csv = """Transaction Date,Details,Amount,Balance
01/04/2024,COLES SUPERMARKET,-82.30,5000.00
02/04/2024,Direct Credit Payroll,5000.00,10000.00
"""
    doc = _make_csv_doc(csv)
    transactions, metadata = extract_statement(doc)
    assert len(transactions) == 2
    assert transactions[0].type == TransactionType.DEBIT
    assert transactions[1].type == TransactionType.CREDIT


def test_extract_period_from_transactions():
    csv = """Date,Description,Amount,Balance
01/01/2024,Opening,-10.00,1000.00
31/01/2024,Closing,-10.00,990.00
"""
    doc = _make_csv_doc(csv)
    transactions, metadata = extract_statement(doc)
    from datetime import date
    assert metadata.period_start == date(2024, 1, 1)
    assert metadata.period_end == date(2024, 1, 31)
