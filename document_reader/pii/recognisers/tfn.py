from presidio_analyzer import Pattern, PatternRecognizer


class TFNRecogniser(PatternRecognizer):
    """Detects Australian Tax File Numbers (9-digit)."""

    PATTERNS = [
        Pattern("TFN_SPACED", r"\b\d{3}\s\d{3}\s\d{3}\b", 0.75),
        Pattern("TFN_PLAIN", r"\bTFN[:\s]+\d{9}\b", 0.95),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="AU_TFN",
            patterns=self.PATTERNS,
            context=["tfn", "tax file", "tax file number"],
        )
