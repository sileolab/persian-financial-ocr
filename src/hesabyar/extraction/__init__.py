"""Information extraction and financial validation module."""

from hesabyar.extraction.llm_parser import LLMDocumentParser
from hesabyar.extraction.rules import RuleBasedExtractor
from hesabyar.extraction.validator import DocumentValidator
from hesabyar.models.schemas import ExtractedDocument


class InformationExtractor:
    """Hybrid extractor: Rule-based regex first, with LLM fallback and Pydantic validation."""

    def __init__(self):
        self.rule_extractor = RuleBasedExtractor()
        self.llm_parser = LLMDocumentParser()
        self.validator = DocumentValidator()

    def extract_and_validate(self, raw_text: str) -> ExtractedDocument:
        """Run complete extraction pipeline with validation and discrepancy checking."""
        # 1. Deterministic Rule-Based Extraction
        doc = self.rule_extractor.extract(raw_text)

        # 2. LLM Fallback if total amount missing or complex items needed
        if doc.total_amount is None:
            llm_doc = self.llm_parser.parse(raw_text)
            if llm_doc and llm_doc.total_amount is not None:
                doc = llm_doc

        # 3. Financial Cross-Validation & Consistency Rules
        validated_doc = self.validator.validate(doc)
        return validated_doc


__all__ = [
    "RuleBasedExtractor",
    "LLMDocumentParser",
    "DocumentValidator",
    "InformationExtractor",
]
