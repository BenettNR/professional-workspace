from extraction.models import ProcessingResult


def to_markdown(result: ProcessingResult) -> str:
    lines: list[str] = []
    m = result.metadata
    s = result.summary

    lines += [
        "# Bank Statement Analysis Report",
        "",
        "## Statement Details",
        f"| Field | Value |",
        f"|---|---|",
        f"| Bank | {m.bank_name or 'Unknown'} |",
        f"| Account | {m.account_number_masked or '[MASKED]'} |",
        f"| BSB | {m.bsb_masked or '[MASKED]'} |",
        f"| Account Holder | {m.account_holder_masked or '[MASKED]'} |",
        f"| Period | {m.period_start} → {m.period_end} |",
        f"| Opening Balance | ${m.opening_balance or '—'} |",
        f"| Closing Balance | ${m.closing_balance or '—'} |",
        "",
    ]

    if s:
        lines += [
            "## Summary",
            f"| Metric | Amount |",
            f"|---|---|",
            f"| Total Credits | ${s.total_credits:,.2f} |",
            f"| Total Debits | ${s.total_debits:,.2f} |",
            f"| Net Cash Flow | ${s.net_cash_flow:,.2f} |",
            f"| Transactions | {s.transaction_count} |",
            "",
        ]

        if s.narrative:
            lines += ["## Analyst Notes", "", s.narrative, ""]

        if s.category_breakdown:
            lines += ["## Spending by Category", "", "| Category | Total | Count | Share |", "|---|---|---|---|"]
            for cat in s.category_breakdown:
                lines.append(f"| {cat.category.value.title()} | ${cat.total:,.2f} | {cat.count} | {cat.percentage}% |")
            lines.append("")

        if s.largest_debit:
            t = s.largest_debit
            lines += [
                "## Notable Transactions",
                "",
                f"**Largest debit:** {t.description_clean or t.description_raw} — ${t.amount:,.2f} on {t.date}",
            ]
        if s.largest_credit:
            t = s.largest_credit
            lines.append(
                f"**Largest credit:** {t.description_clean or t.description_raw} — ${t.amount:,.2f} on {t.date}"
            )
        lines.append("")

        if s.recurring_payments:
            lines += ["## Recurring Payments", ""]
            for t in s.recurring_payments:
                lines.append(f"- {t.description_clean or t.description_raw} — ~${t.amount:,.2f}")
            lines.append("")

    lines += [
        "## Transactions",
        "",
        "| Date | Description | Type | Amount | Balance | Category |",
        "|---|---|---|---|---|---|",
    ]
    for t in result.transactions:
        balance = f"${t.balance:,.2f}" if t.balance is not None else "—"
        cat = t.category.value.title() if t.category else "—"
        lines.append(
            f"| {t.date} | {t.description_clean or t.description_raw} "
            f"| {t.type.value} | ${t.amount:,.2f} | {balance} | {cat} |"
        )

    lines += [
        "",
        "---",
        f"*PII entities masked: {result.pii_entities_masked} · "
        f"Generated: {result.processed_at.strftime('%Y-%m-%d %H:%M UTC')}*",
    ]

    return "\n".join(lines)
