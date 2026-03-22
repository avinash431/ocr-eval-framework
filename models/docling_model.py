"""Docling / SmolDocling wrapper."""

from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class DoclingModel(BaseOCRModel):
    @property
    def name(self): return "docling"
    @property
    def display_name(self): return "Docling / SmolDocling"

    def setup(self):
        from docling.document_converter import DocumentConverter
        self._converter = DocumentConverter()
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        result = self._converter.convert(image_path)
        text = result.document.export_to_markdown()
        tables = []
        for table in result.document.tables:
            tables.append(table.export_to_dataframe().to_dict())

        return OCRResult(
            raw_text=text,
            structured_data={"tables": tables} if tables else None,
            metadata={"num_tables": len(tables), "num_pages": getattr(result.document, 'num_pages', 1)},
        )
