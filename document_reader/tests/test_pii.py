"""
PII masking tests — no external dependencies required.
Tests the recognisers and masker logic with synthetic AU banking data.
"""

import pytest
from pii.masker import PIIMasker
from pii.recognisers.bsb import BSBRecogniser
from pii.recognisers.tfn import TFNRecogniser
from pii.recognisers.abn import ABNRecogniser


def test_bsb_pattern_matches():
    rec = BSBRecogniser()
    results = rec.analyze("BSB: 123-456 account 987654321", entities=["AU_BSB"], language="en")
    assert any(r.entity_type == "AU_BSB" for r in results)


def test_tfn_pattern_matches():
    rec = TFNRecogniser()
    results = rec.analyze("TFN: 123456789", entities=["AU_TFN"], language="en")
    assert any(r.entity_type == "AU_TFN" for r in results)


def test_abn_pattern_matches():
    rec = ABNRecogniser()
    results = rec.analyze("ABN: 51 824 753 556", entities=["AU_ABN"], language="en")
    assert any(r.entity_type == "AU_ABN" for r in results)


def test_masker_bsb_is_replaced():
    masker = PIIMasker()
    masked, count = masker.mask("Transfer from BSB 062-000 account 12345678")
    assert "062-000" not in masked
    assert "***-***" in masked


def test_masker_consistent_tokens():
    masker = PIIMasker()
    text1 = "Payment to John Smith reference 12345"
    text2 = "Refund from John Smith"
    masked1, _ = masker.mask(text1)
    masked2, _ = masker.mask(text2)
    # Extract the person token used in each — should be the same
    import re
    token1 = re.findall(r"\[PERSON-\d+\]", masked1)
    token2 = re.findall(r"\[PERSON-\d+\]", masked2)
    assert token1 and token2
    assert token1[0] == token2[0], "Same name should produce the same token across calls"


def test_masker_count_increases():
    masker = PIIMasker()
    _, c1 = masker.mask("Jane Doe BSB 123-456")
    _, c2 = masker.mask("Another Person TFN: 123456789")
    assert masker.masked_count == c1 + c2


def test_masker_clean_text_unchanged():
    masker = PIIMasker()
    text = "EFTPOS WOOLWORTHS SYDNEY"
    masked, count = masker.mask(text)
    assert count == 0
    assert masked == text
