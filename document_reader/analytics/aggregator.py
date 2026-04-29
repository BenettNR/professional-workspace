"""
Computes aggregate statistics from a list of masked, enriched transactions.
"""

from collections import Counter, defaultdict
from decimal import Decimal

from extraction.models import CategorySummary, Transaction, TransactionCategory, TransactionType


def _detect_recurring(transactions: list[Transaction], min_occurrences: int = 2) -> list[Transaction]:
    """
    Identify recurring payments by grouping on merchant + approximate amount.
    Returns one representative transaction per recurring group.
    """
    key_counts: Counter = Counter()
    key_to_txn: dict[str, Transaction] = {}

    for txn in transactions:
        if txn.type != TransactionType.DEBIT:
            continue
        merchant = (txn.description_clean or txn.description_raw).lower()[:30]
        # Round to nearest $1 to tolerate minor amount variation
        amount_bucket = round(float(txn.amount))
        key = f"{merchant}|{amount_bucket}"
        key_counts[key] += 1
        key_to_txn[key] = txn

    return [
        key_to_txn[k]
        for k, count in key_counts.items()
        if count >= min_occurrences
    ]


def aggregate(transactions: list[Transaction]) -> dict:
    total_credits = Decimal("0")
    total_debits = Decimal("0")
    largest_debit: Transaction | None = None
    largest_credit: Transaction | None = None
    category_totals: defaultdict[TransactionCategory, Decimal] = defaultdict(Decimal)
    category_counts: defaultdict[TransactionCategory, int] = defaultdict(int)

    for txn in transactions:
        if txn.type == TransactionType.CREDIT:
            total_credits += txn.amount
            if largest_credit is None or txn.amount > largest_credit.amount:
                largest_credit = txn
        else:
            total_debits += txn.amount
            if largest_debit is None or txn.amount > largest_debit.amount:
                largest_debit = txn

        if txn.category:
            category_totals[txn.category] += txn.amount
            category_counts[txn.category] += 1

    total_spend = sum(category_totals.values()) or Decimal("1")
    category_breakdown = [
        CategorySummary(
            category=cat,
            total=total,
            count=category_counts[cat],
            percentage=round(float(total / total_spend) * 100, 1),
        )
        for cat, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "total_credits": total_credits,
        "total_debits": total_debits,
        "net_cash_flow": total_credits - total_debits,
        "transaction_count": len(transactions),
        "largest_debit": largest_debit,
        "largest_credit": largest_credit,
        "category_breakdown": category_breakdown,
        "recurring_payments": _detect_recurring(transactions),
    }
