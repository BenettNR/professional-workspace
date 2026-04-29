from presidio_analyzer import Pattern, PatternRecognizer


class BSBRecogniser(PatternRecognizer):
    """Detects Australian BSB numbers in XXX-XXX format."""

    PATTERNS = [
        Pattern("BSB_HYPHENATED", r"\b\d{3}-\d{3}\b", 0.85),
        Pattern("BSB_PLAIN", r"\bBSB[:\s]+\d{6}\b", 0.9),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="AU_BSB",
            patterns=self.PATTERNS,
            context=["bsb", "bank state branch", "routing"],
        )
