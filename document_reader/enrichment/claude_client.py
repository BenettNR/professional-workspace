"""
Claude API client with prompt caching on the system prompt.
The system prompt is identical across all merchant-decode calls, so it
qualifies for Anthropic's prompt cache and stays warm across a session.
"""

import json
import logging

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)

_MERCHANT_SYSTEM_PROMPT = """\
You are an expert in Australian bank transaction descriptions. Your job is to decode \
cryptic transaction descriptions into clean, human-readable merchant names.

Australian bank transaction conventions you must know:
- EFTPOS / VISA PURCHASE / MASTERCARD PURCHASE prefix: card purchase at the named merchant
- BPAY: bill payment — the description contains the biller name
- SQ *: Square card reader merchant
- PAYPAL *: PayPal payment to named merchant
- Direct Debit / DD: recurring payment to named company
- Direct Credit / DC: incoming payment from named company
- OSKO / PayID / NPP: fast payment system — may include sender/receiver name (treat as PII)
- Salary / Payroll / Wages: employment income
- ATO: Australian Taxation Office
- Descriptions often truncate merchant names to ~18 characters due to legacy EFTPOS limits
- State codes like NSW, VIC, QLD, WA, SA are usually part of the merchant location

Return ONLY valid JSON with no markdown, no explanation:
{"decoded": {"<raw_description>": "<clean_merchant_name>", ...}}

Rules:
1. If the description is already clear, return it as-is (cleaned of prefixes)
2. Expand known abbreviations (e.g. WOOLWORTHS → Woolworths, MCDs → McDonald's)
3. Never invent information not present in the description
4. If genuinely ambiguous, return "Unknown Merchant"
5. Strip EFTPOS/VISA/MASTERCARD prefixes from the output
"""


class ClaudeClient:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def decode_merchants(self, descriptions: list[str]) -> dict[str, str]:
        """Batch-decode up to 50 cryptic descriptions. Returns {raw: decoded}."""
        if not descriptions:
            return {}

        batch = "\n".join(f'- "{d}"' for d in descriptions[:50])
        try:
            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": _MERCHANT_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": f"Decode these transaction descriptions:\n{batch}",
                    }
                ],
            )
            raw = response.content[0].text.strip()
            data = json.loads(raw)
            return data.get("decoded", {})
        except (anthropic.APIError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Claude merchant decode failed: %s", e)
            return {}

    def generate_narrative(self, summary_json: str) -> str:
        """Generate a plain-English paragraph summarising the statement."""
        try:
            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": (
                            "You summarise bank statement data in 2-3 clear sentences for a "
                            "compliance analyst. Focus on cash flow, notable categories, and "
                            "any patterns worth flagging. All PII has already been masked — "
                            "do not reference any personal details."
                        ),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {"role": "user", "content": f"Summarise this statement data:\n{summary_json}"}
                ],
            )
            return response.content[0].text.strip()
        except anthropic.APIError as e:
            logger.warning("Claude narrative generation failed: %s", e)
            return ""
