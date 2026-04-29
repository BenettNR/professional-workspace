from presidio_analyzer import Pattern, PatternRecognizer


class ABNRecogniser(PatternRecognizer):
    """Detects Australian Business Numbers (11-digit)."""

    PATTERNS = [
        Pattern("ABN_SPACED", r"\b\d{2}\s\d{3}\s\d{3}\s\d{3}\b", 0.75),
        Pattern("ABN_PLAIN", r"\bABN[:\s]+\d{11}\b", 0.95),
        Pattern("ABN_COMPACT", r"\bABN[:\s]+\d{2}\s\d{3}\s\d{3}\s\d{3}\b", 0.95),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="AU_ABN",
            patterns=self.PATTERNS,
            context=["abn", "australian business number", "business number"],
        )
