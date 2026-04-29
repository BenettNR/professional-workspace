import json

from extraction.models import StatementMetadata, StatementSummary, Transaction
from analytics.aggregator import aggregate
from enrichment.claude_client import ClaudeClient


def build_summary(
    transactions: list[Transaction],
    metadata: StatementMetadata,
    generate_narrative: bool = True,
) -> StatementSummary:
    stats = aggregate(transactions)

    summary = StatementSummary(
        period_start=metadata.period_start,
        period_end=metadata.period_end,
        **{k: v for k, v in stats.items()},
    )

    if generate_narrative:
        payload = {
            "period": f"{metadata.period_start} to {metadata.period_end}",
            "total_credits": str(summary.total_credits),
            "total_debits": str(summary.total_debits),
            "net_cash_flow": str(summary.net_cash_flow),
            "transaction_count": summary.transaction_count,
            "top_categories": [
                {"category": c.category.value, "total": str(c.total), "pct": c.percentage}
                for c in summary.category_breakdown[:5]
            ],
            "recurring_count": len(summary.recurring_payments),
        }
        client = ClaudeClient()
        summary.narrative = client.generate_narrative(json.dumps(payload))

    return summary
