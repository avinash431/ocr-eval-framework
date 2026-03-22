"""Azure AI Document Intelligence wrapper."""

from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class AzureADI(BaseOCRModel):
    @property
    def name(self): return "azure_adi"
    @property
    def display_name(self): return "Azure AI Document Intelligence"
    @property
    def model_type(self): return "cloud_api"

    def setup(self):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        cfg = self.config.get("azure", {})
        if not cfg.get("endpoint") or not cfg.get("api_key"):
            raise ValueError("Azure endpoint and api_key required in config.")
        self._client = DocumentIntelligenceClient(
            endpoint=cfg["endpoint"],
            credential=AzureKeyCredential(cfg["api_key"]),
        )
        self._model_id = cfg.get("model_id", "prebuilt-layout")
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        with open(image_path, "rb") as f:
            poller = self._client.begin_analyze_document(
                self._model_id, body=f, content_type="application/octet-stream",
            )
        result = poller.result()

        # Extract text
        lines = []
        for page in result.pages:
            for line in page.lines:
                lines.append(line.content)

        # Extract tables
        tables = []
        if result.tables:
            for table in result.tables:
                t = {"rows": table.row_count, "cols": table.column_count, "cells": []}
                for cell in table.cells:
                    t["cells"].append({
                        "row": cell.row_index, "col": cell.column_index,
                        "text": cell.content, "kind": getattr(cell, "kind", "content"),
                    })
                tables.append(t)

        # Extract key-value pairs
        kv_pairs = {}
        if result.key_value_pairs:
            for pair in result.key_value_pairs:
                key = pair.key.content if pair.key else ""
                val = pair.value.content if pair.value else ""
                if key:
                    kv_pairs[key] = val

        raw_text = "\n".join(lines)
        return OCRResult(
            raw_text=raw_text,
            structured_data={"tables": tables, "key_value_pairs": kv_pairs} if tables or kv_pairs else None,
            metadata={"pages": len(result.pages), "num_tables": len(tables)},
        )

    def estimate_cost(self, num_pages: int) -> float:
        # Azure: ~$1.50 per 1000 pages (layout model), 500 free/month
        free_pages = 500
        billable = max(0, num_pages - free_pages)
        return billable * 0.0015
