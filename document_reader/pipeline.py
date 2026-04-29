"""
Orchestrates the full processing pipeline for a single document:
  Ingest → Extract → PII mask → Enrich → Analyse → Return ProcessingResult
"""

from pathlib import Path

from analytics.report_generator import build_summary
from enrichment.categoriser import categorise
from enrichment.merchant_decoder import MerchantDecoder
from extraction.models import ProcessingResult, StatementMetadata, Transaction
from ingestion.bank_formats.extractor import extract_statement
from ingestion.router import get_parser
from pii.masker import PIIMasker


def _mask_metadata(metadata: StatementMetadata, masker: PIIMasker) -> StatementMetadata:
    fields_to_mask = ["account_holder_masked", "account_number_masked", "bsb_masked", "bank_name"]
    data = metadata.model_dump()
    for field in fields_to_mask:
        if data.get(field):
            masked, _ = masker.mask(str(data[field]), context=field)
            data[field] = masked
    return StatementMetadata(**data)


def process_document(
    file_path: Path,
    generate_narrative: bool = True,
    use_claude_enrichment: bool = True,
) -> ProcessingResult:
    # 1. Parse
    parser = get_parser(file_path)
    raw_doc = parser.parse(file_path)

    # 2. Extract transactions + metadata
    transactions, metadata = extract_statement(raw_doc)

    # 3. PII masking — single session so tokens are consistent across doc
    masker = PIIMasker()

    metadata = _mask_metadata(metadata, masker)

    masked_transactions: list[Transaction] = []
    for txn in transactions:
        masked_desc, _ = masker.mask(txn.description_raw, context="transaction_description")
        masked_transactions.append(txn.model_copy(update={"description_raw": masked_desc}))

    # 4. Merchant decoding (on masked descriptions — PII already removed)
    decoder = MerchantDecoder(use_claude=use_claude_enrichment)
    descriptions = [t.description_raw for t in masked_transactions]
    decoded = decoder.decode_batch(descriptions)

    enriched: list[Transaction] = []
    for txn in masked_transactions:
        clean = decoded.get(txn.description_raw)
        category = categorise(txn.model_copy(update={"description_clean": clean}))
        enriched.append(txn.model_copy(update={"description_clean": clean, "category": category}))

    # 5. Analytics
    summary = build_summary(enriched, metadata, generate_narrative=generate_narrative)

    return ProcessingResult(
        metadata=metadata,
        transactions=enriched,
        summary=summary,
        pii_entities_masked=masker.masked_count,
        pii_audit_log=masker.audit_log,
    )
