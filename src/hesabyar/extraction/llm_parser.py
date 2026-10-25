"""LLM-based structured extractor fallback for complex receipts and ambiguous invoices."""

import json
from typing import Optional

from hesabyar.core.config import get_settings
from hesabyar.core.exceptions import ExtractionError
from hesabyar.core.logger import get_logger
from hesabyar.models.schemas import CurrencyUnit, ExtractedDocument, FinancialItem

logger = get_logger("extraction.llm_parser")

LLM_SYSTEM_PROMPT = """You are an expert Iranian financial document parser.
Your task is to parse Persian receipts, POS slips, and invoices into structured JSON.
DO NOT guess or invent numbers. If a field cannot be determined with certainty, leave it as null.
Determine amounts in whole numbers. Identify currency as IRR (Rial) or IRT (Toman).

Output MUST be a valid JSON object matching the following schema:
{
  "document_type": "pos_slip" | "invoice" | "receipt" | "bank_transfer",
  "vendor_name": string or null,
  "issue_date": "YYYY/MM/DD" (Solar Hijri) or null,
  "issue_time": "HH:MM:SS" or null,
  "total_amount": integer or null,
  "currency": "IRR" | "IRT",
  "tracking_code": string or null,
  "card_number": string or null,
  "terminal_id": string or null,
  "items": [
    {
      "description": string,
      "quantity": float,
      "unit_price": integer or null,
      "total_price": integer or null
    }
  ]
}
"""


class LLMDocumentParser:
    """Parses complex documents using LLM with structured output."""

    def __init__(self):
        self.settings = get_settings()

    def parse(self, raw_text: str) -> Optional[ExtractedDocument]:
        """
        Send raw OCR text to configured LLM (OpenAI or Gemini) if API keys exist.
        Returns populated ExtractedDocument or None if no API key is available.
        """
        if not raw_text.strip():
            return None

        # Check for OpenAI API Key
        if self.settings.openai_api_key:
            return self._parse_with_openai(raw_text)

        # Check for Gemini API Key
        if self.settings.gemini_api_key:
            return self._parse_with_gemini(raw_text)

        logger.debug("No LLM API keys configured. Skipping LLM fallback.")
        return None

    def _parse_with_openai(self, raw_text: str) -> Optional[ExtractedDocument]:
        """Call OpenAI API with JSON schema."""
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract financial data from this document:\n\n{raw_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                return self._dict_to_document(data)
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            raise ExtractionError(f"LLM extraction error: {e}") from e
        return None

    def _parse_with_gemini(self, raw_text: str) -> Optional[ExtractedDocument]:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(self.settings.llm_model)
            prompt = f"{LLM_SYSTEM_PROMPT}\n\nDocument text:\n{raw_text}"
            res = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            if res.text:
                data = json.loads(res.text)
                return self._dict_to_document(data)
        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            raise ExtractionError(f"LLM extraction error: {e}") from e
        return None

    def _dict_to_document(self, data: dict) -> ExtractedDocument:
        """Convert parsed JSON dict into ExtractedDocument Pydantic model."""
        items = []
        for it in data.get("items", []):
            items.append(
                FinancialItem(
                    description=it.get("description", "Unknown Item"),
                    quantity=float(it.get("quantity", 1.0)),
                    unit_price=it.get("unit_price"),
                    total_price=it.get("total_price"),
                )
            )

        curr_str = data.get("currency", "IRT").upper()
        currency = CurrencyUnit.IRR if curr_str == "IRR" else CurrencyUnit.IRT

        return ExtractedDocument(
            document_type=data.get("document_type", "receipt"),
            vendor_name=data.get("vendor_name"),
            issue_date=data.get("issue_date"),
            issue_time=data.get("issue_time"),
            total_amount=data.get("total_amount"),
            currency=currency,
            tracking_code=data.get("tracking_code"),
            card_number=data.get("card_number"),
            terminal_id=data.get("terminal_id"),
            items=items,
            confidence_score=0.92,
            needs_review=False,
            review_reasons=[],
        )
