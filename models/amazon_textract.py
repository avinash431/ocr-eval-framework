"""Amazon Textract wrapper."""

from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class AmazonTextract(BaseOCRModel):
    @property
    def name(self): return "amazon_textract"
    @property
    def display_name(self): return "Amazon Textract"
    @property
    def model_type(self): return "cloud_api"

    def setup(self):
        import boto3
        cfg = self.config.get("aws", {})
        kwargs = {"region_name": cfg.get("region", "us-east-1")}
        if cfg.get("access_key_id"):
            kwargs["aws_access_key_id"] = cfg["access_key_id"]
            kwargs["aws_secret_access_key"] = cfg["secret_access_key"]
        self._client = boto3.client("textract", **kwargs)
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        # AnalyzeDocument extracts tables + forms + text
        response = self._client.analyze_document(
            Document={"Bytes": img_bytes},
            FeatureTypes=["TABLES", "FORMS"],
        )

        # Extract text lines
        lines = []
        confidences = []
        for block in response["Blocks"]:
            if block["BlockType"] == "LINE":
                lines.append(block["Text"])
                confidences.append(block["Confidence"] / 100)

        # Extract tables
        tables = []
        table_blocks = [b for b in response["Blocks"] if b["BlockType"] == "TABLE"]
        cell_blocks = {b["Id"]: b for b in response["Blocks"] if b["BlockType"] == "CELL"}
        word_blocks = {b["Id"]: b for b in response["Blocks"]
                       if b["BlockType"] in ("WORD", "SELECTION_ELEMENT")}

        for table in table_blocks:
            t = {"cells": []}
            if "Relationships" in table:
                for rel in table["Relationships"]:
                    if rel["Type"] == "CHILD":
                        for cell_id in rel["Ids"]:
                            cell = cell_blocks.get(cell_id, {})
                            cell_text = ""
                            if "Relationships" in cell:
                                for crel in cell["Relationships"]:
                                    if crel["Type"] == "CHILD":
                                        words = [word_blocks.get(wid, {}).get("Text", "")
                                                 for wid in crel["Ids"]]
                                        cell_text = " ".join(w for w in words if w)
                            t["cells"].append({
                                "row": cell.get("RowIndex", 0),
                                "col": cell.get("ColumnIndex", 0),
                                "text": cell_text,
                            })
            tables.append(t)

        # Extract key-value pairs
        kv_pairs = {}
        kv_sets = [b for b in response["Blocks"] if b["BlockType"] == "KEY_VALUE_SET"]
        for kv in kv_sets:
            if "KEY" in kv.get("EntityTypes", []):
                key_text = self._get_text_from_block(kv, response["Blocks"])
                val_text = ""
                if "Relationships" in kv:
                    for rel in kv["Relationships"]:
                        if rel["Type"] == "VALUE":
                            for vid in rel["Ids"]:
                                vblock = next((b for b in response["Blocks"] if b["Id"] == vid), None)
                                if vblock:
                                    val_text = self._get_text_from_block(vblock, response["Blocks"])
                if key_text:
                    kv_pairs[key_text] = val_text

        raw_text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else None

        return OCRResult(
            raw_text=raw_text, confidence=avg_conf,
            structured_data={"tables": tables, "key_value_pairs": kv_pairs} if tables or kv_pairs else None,
            metadata={"lines": len(lines), "num_tables": len(tables)},
        )

    def _get_text_from_block(self, block, all_blocks):
        words = []
        if "Relationships" in block:
            for rel in block["Relationships"]:
                if rel["Type"] == "CHILD":
                    for wid in rel["Ids"]:
                        wb = next((b for b in all_blocks if b["Id"] == wid), None)
                        if wb and wb.get("Text"):
                            words.append(wb["Text"])
        return " ".join(words)

    def estimate_cost(self, num_pages: int) -> float:
        # Textract AnalyzeDocument: ~$15/1000 pages (tables+forms), 1000 free for 3 months
        return num_pages * 0.015
