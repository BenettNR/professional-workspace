"""
PIIMasker applies entity detection and masking to transaction text.
Consistent pseudonymisation: the same raw value always maps to the same token
within a single processing session, so masked reports remain internally coherent.
"""

import re
from typing import Optional

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config.settings import settings
from pii.engine import analyse_text

_anonymizer = AnonymizerEngine()

_ACCOUNT_RE = re.compile(r"\b(\d{6,10})\b")


class PIIMasker:
    def __init__(self) -> None:
        self._seen: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        self.masked_count = 0
        self.audit_log: list[str] = []

    def _token_for(self, entity_type: str, value: str) -> str:
        if value in self._seen:
            return self._seen[value]

        self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
        n = self._counters[entity_type]

        token = self._format_token(entity_type, value, n)
        self._seen[value] = token
        return token

    def _format_token(self, entity_type: str, value: str, n: int) -> str:
        k = settings.pii_keep_last_digits
        match entity_type:
            case "PERSON":
                return f"[PERSON-{n}]"
            case "AU_BSB":
                return "***-***"
            case "AU_TFN":
                return "***-***-***"
            case "AU_ABN":
                return "** *** *** ***"
            case "PHONE_NUMBER":
                digits = re.sub(r"\D", "", value)
                return f"[PHONE-{digits[-k:]}]" if len(digits) >= k else "[PHONE]"
            case "EMAIL_ADDRESS":
                return f"[EMAIL-{n}]"
            case "CREDIT_CARD" | "IBAN_CODE":
                digits = re.sub(r"\D", "", value)
                return f"****{digits[-k:]}" if len(digits) >= k else "****"
            case _:
                return f"[{entity_type}-{n}]"

    def mask(self, text: str, context: Optional[str] = None) -> tuple[str, int]:
        """
        Returns (masked_text, count_of_entities_masked).
        Optionally pass context (e.g. 'account_holder') for audit log entries.
        """
        results = analyse_text(text)

        # Also catch bare account numbers not caught by named recognisers
        for m in _ACCOUNT_RE.finditer(text):
            val = m.group(1)
            digits = re.sub(r"\D", "", val)
            if len(digits) in range(6, 11) and val not in self._seen:
                k = settings.pii_keep_last_digits
                self._seen[val] = f"****{digits[-k:]}"

        if not results:
            return text, 0

        operators = {
            entity_type: OperatorConfig(
                "custom",
                {"lambda": lambda x, et=entity_type: self._token_for(et, x)},
            )
            for entity_type in {r.entity_type for r in results}
        }

        anonymized = _anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
        count = len(results)
        self.masked_count += count

        if context:
            for r in results:
                self.audit_log.append(
                    f"{context}: masked {r.entity_type} at [{r.start}:{r.end}] (score={r.score:.2f})"
                )

        return anonymized.text, count
