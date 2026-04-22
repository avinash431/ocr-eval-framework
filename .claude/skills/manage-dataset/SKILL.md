---
name: manage-dataset
description: Download, organize, and manage test datasets for OCR evaluation. Use when the user wants to download documents, add test images, create ground truth annotations, set up a new document category, or manage the test dataset. Triggers on "download dataset", "add documents", "ground truth", "test data", "new category", or any mention of managing evaluation data.
---

# Manage Dataset

Download, organize, and maintain the test document dataset and ground truth annotations.

## Dataset Structure

```
test-dataset/
├── 01_printed_english/
│   ├── invoices/
│   ├── contracts/
│   ├── reports/
│   └── letters/
├── 02_complex_tables/
│   ├── financial/
│   ├── multi_column/
│   ├── nested_merged/
│   └── forms/
├── 03_handwritten/
│   ├── english/
│   ├── hindi/
│   └── other_indic/
├── 04_indian_languages/
│   ├── hindi/
│   ├── tamil/
│   ├── bengali/
│   ├── telugu/
│   └── mixed_bilingual/
├── 05_low_quality_scans/
│   ├── faded/
│   ├── skewed/
│   └── noisy/
├── 06_mixed_content/
│   ├── text_and_images/
│   ├── equations/
│   └── receipts/
├── 07_org_documents/
│   └── manual/
└── ground_truth/
    └── <mirrors above structure with _gt.txt files>
```

## Downloading the Dataset

```bash
# Download with default 5 samples per category
python tools/download_dataset.py --output-dir ./test-dataset --samples 5

# Download more samples
python tools/download_dataset.py --output-dir ./test-dataset --samples 20
```

The script downloads from public datasets:
- **FUNSD**: Form understanding documents → `02_complex_tables/forms/`
- **OmniDocBench**: Complex layouts → tables, multi-column, mixed content
- **Omni OCR Benchmark**: 1000-doc benchmark distributed across categories
- **IAM Handwriting**: English handwritten text → `03_handwritten/english/`
- **Indic Text Recognition**: Indian language scripts → `04_indian_languages/`

## Adding Custom Documents

### Add images to an existing category

1. Place image files (.jpg, .png, .pdf) in the appropriate category folder
2. Create matching ground truth:
   ```
   test-dataset/ground_truth/<category>/<filename_without_ext>_gt.txt
   ```

### Add a new document category

1. Create the folder under `test-dataset/`:
   ```bash
   mkdir -p test-dataset/08_new_category/subcategory
   mkdir -p test-dataset/ground_truth/08_new_category/subcategory
   ```
2. Follow the naming convention: `NN_descriptive_name/`
3. Update `download_dataset.py` if you want auto-download support
4. Update `utils/helpers.py` `get_document_category()` if the prefix pattern differs

### Ground Truth Formats

The framework supports multiple ground truth formats (checked in this order):

1. **`<stem>_gt.txt`** — Plain text, preferred format
2. **`<stem>.txt`** — Plain text, alternate naming
3. **`<stem>.md`** — Markdown formatted text
4. **`<stem>.json`** — JSON with `form` key containing word objects (FUNSD format)

For best results, ground truth should contain the exact text content of the document with natural line breaks.

## Validating the Dataset

Before running evaluations, verify the dataset is complete:

```python
from utils.helpers import load_config, find_documents, get_ground_truth

config = load_config()
docs = find_documents(config["paths"]["dataset_dir"])
gt_dir = config["paths"]["ground_truth_dir"]

print(f"Total documents: {len(docs)}")

missing_gt = []
for doc in docs:
    gt = get_ground_truth(str(doc), gt_dir)
    if not gt:
        missing_gt.append(str(doc))

print(f"Documents with ground truth: {len(docs) - len(missing_gt)}")
print(f"Missing ground truth: {len(missing_gt)}")
for m in missing_gt[:10]:
    print(f"  - {m}")
```

Documents without ground truth will still be processed by OCR models, but no accuracy metrics will be computed for them.
