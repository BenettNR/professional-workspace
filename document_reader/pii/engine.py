from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from pii.recognisers import ABNRecogniser, BSBRecogniser, TFNRecogniser

_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "AU_BSB",
    "AU_TFN",
    "AU_ABN",
]


def build_analyzer() -> AnalyzerEngine:
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    registry.add_recognizer(BSBRecogniser())
    registry.add_recognizer(TFNRecogniser())
    registry.add_recognizer(ABNRecogniser())

    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)


# Module-level singleton — loaded once at startup
_analyzer: AnalyzerEngine | None = None


def get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        _analyzer = build_analyzer()
    return _analyzer


def analyse_text(text: str, language: str = "en") -> list:
    return get_analyzer().analyze(text=text, language=language, entities=_ENTITIES)
