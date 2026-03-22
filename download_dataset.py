#!/usr/bin/env python3
"""
OCR Evaluation — Test Dataset Downloader & Organizer
=====================================================
Downloads sample documents from public OCR benchmark datasets
and organizes them into the evaluation folder structure.

Usage:
    pip install requests huggingface_hub tqdm
    python download_test_dataset.py [--output-dir ./ocr-eval-test-dataset] [--samples 5]

Notes:
    - Some datasets require manual download (marked below)
    - The script downloads SAMPLES, not full datasets
    - Run with --samples N to control how many files per source
    - Organization documents (07_org_documents/) must be added manually by your team
"""

import os
import sys
import json
import random
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "./test-dataset"
DEFAULT_SAMPLES = 5  # samples per sub-category from each source
LOG_FILE = "download_log.txt"

FOLDER_STRUCTURE = [
    "01_printed_english/invoices",
    "01_printed_english/contracts",
    "01_printed_english/reports",
    "01_printed_english/letters_emails",
    "02_complex_tables/financial",
    "02_complex_tables/multi_column",
    "02_complex_tables/nested_merged",
    "02_complex_tables/forms",
    "03_handwritten/english",
    "03_handwritten/hindi_devanagari",
    "03_handwritten/other_indic",
    "04_indian_languages/hindi",
    "04_indian_languages/tamil",
    "04_indian_languages/bengali",
    "04_indian_languages/telugu",
    "04_indian_languages/mixed_bilingual",
    "04_european_languages/spanish",
    "04_european_languages/french",
    "04_european_languages/german",
    "05_low_quality_scans/faded",
    "05_low_quality_scans/skewed",
    "05_low_quality_scans/noisy",
    "06_mixed_content/text_and_images",
    "06_mixed_content/equations_formulas",
    "06_mixed_content/receipts",
    "07_org_documents",
    "ground_truth/01_printed_english",
    "ground_truth/02_complex_tables",
    "ground_truth/03_handwritten",
    "ground_truth/04_indian_languages",
    "ground_truth/04_european_languages",
    "ground_truth/05_low_quality_scans",
    "ground_truth/06_mixed_content",
    "ground_truth/07_org_documents",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg, log_fp=None):
    """Print and optionally write to log file."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if log_fp:
        log_fp.write(line + "\n")
        log_fp.flush()


def ensure_pip_package(package, import_name=None):
    """Install a pip package if not already available."""
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--quiet",
             "--break-system-packages"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def create_folders(base_dir):
    """Create the full folder structure."""
    for folder in FOLDER_STRUCTURE:
        Path(base_dir, folder).mkdir(parents=True, exist_ok=True)


def safe_download(url, dest_path, timeout=60):
    """Download a file with error handling."""
    import requests
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    ⚠ Failed to download {url}: {e}")
        return False


def copy_samples(src_dir, dest_dir, extensions=None, max_files=5):
    """Copy a random sample of files from src to dest."""
    extensions = extensions or [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"]
    files = []
    for ext in extensions:
        files.extend(Path(src_dir).rglob(f"*{ext}"))
    if not files:
        return 0
    sample = random.sample(files, min(max_files, len(files)))
    for f in sample:
        shutil.copy2(f, Path(dest_dir) / f.name)
    return len(sample)


# ---------------------------------------------------------------------------
# Dataset Downloaders
# ---------------------------------------------------------------------------

def download_funsd(base_dir, n_samples, log_fp):
    """
    FUNSD — Form Understanding in Noisy Scanned Documents
    Source: https://guillaumejaume.github.io/FUNSD/
    Auto-downloadable: YES
    """
    log("=" * 60, log_fp)
    log("📄 FUNSD — Annotated Forms Dataset", log_fp)
    log("=" * 60, log_fp)

    import requests, zipfile, io

    url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
    tmp_dir = Path(base_dir) / "_tmp_funsd"
    tmp_dir.mkdir(exist_ok=True)

    log("  Downloading FUNSD dataset...", log_fp)
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(tmp_dir)
        log("  ✅ Downloaded and extracted.", log_fp)
    except Exception as e:
        log(f"  ❌ Failed: {e}", log_fp)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # Find image files
    images = list(tmp_dir.rglob("*.png"))
    if not images:
        log("  ⚠ No images found in archive.", log_fp)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # Copy samples to forms folder
    sample = random.sample(images, min(n_samples, len(images)))
    dest = Path(base_dir) / "02_complex_tables" / "forms"
    for img in sample:
        shutil.copy2(img, dest / img.name)
    log(f"  ✅ Copied {len(sample)} form samples → 02_complex_tables/forms/", log_fp)

    # Copy corresponding annotations as ground truth
    gt_dest = Path(base_dir) / "ground_truth" / "02_complex_tables"
    annotations_dir = tmp_dir / "dataset"
    for img in sample:
        json_name = img.stem + ".json"
        for json_file in annotations_dir.rglob(json_name):
            shutil.copy2(json_file, gt_dest / json_name)
            break

    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_omnidocbench(base_dir, n_samples, log_fp):
    """
    OmniDocBench — Comprehensive Document Parsing Benchmark
    Source: HuggingFace (opendatalab/OmniDocBench)
    Auto-downloadable: YES (via huggingface_hub)
    """
    log("=" * 60, log_fp)
    log("📄 OmniDocBench — Complex Layouts & Tables", log_fp)
    log("=" * 60, log_fp)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        log("  ⚠ huggingface_hub not installed. Skipping.", log_fp)
        return

    repo_id = "opendatalab/OmniDocBench"
    tmp_dir = Path(base_dir) / "_tmp_omnidocbench"
    tmp_dir.mkdir(exist_ok=True)

    try:
        log("  Listing files in OmniDocBench repo...", log_fp)
        all_files = list_repo_files(repo_id, repo_type="dataset")
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            log("  ⚠ No image files found. Trying to download annotation JSON...", log_fp)
            json_files = [f for f in all_files if f.endswith('.json')]
            if json_files:
                dl_path = hf_hub_download(repo_id, json_files[0], repo_type="dataset",
                                          local_dir=str(tmp_dir))
                log(f"  ℹ Downloaded annotation file: {json_files[0]}", log_fp)
                log("  ℹ Images may need separate download. Check repo README.", log_fp)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        sample_files = random.sample(image_files, min(n_samples * 3, len(image_files)))
        downloaded = []
        for fname in sample_files:
            try:
                dl_path = hf_hub_download(repo_id, fname, repo_type="dataset",
                                          local_dir=str(tmp_dir))
                downloaded.append(Path(dl_path))
            except Exception:
                continue

        # Distribute to relevant folders
        table_dest = Path(base_dir) / "02_complex_tables" / "financial"
        multicol_dest = Path(base_dir) / "02_complex_tables" / "multi_column"
        mixed_dest = Path(base_dir) / "06_mixed_content" / "equations_formulas"

        for i, img in enumerate(downloaded):
            if i < n_samples:
                shutil.copy2(img, table_dest / f"omnidoc_{img.name}")
            elif i < n_samples * 2:
                shutil.copy2(img, multicol_dest / f"omnidoc_{img.name}")
            else:
                shutil.copy2(img, mixed_dest / f"omnidoc_{img.name}")

        log(f"  ✅ Distributed {len(downloaded)} samples across tables/multi-column/mixed content", log_fp)

    except Exception as e:
        log(f"  ❌ Failed: {e}", log_fp)

    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_omni_benchmark(base_dir, n_samples, log_fp):
    """
    Omni OCR Benchmark (getomni)
    Source: HuggingFace (getomni/ocr-benchmark)
    Auto-downloadable: YES
    """
    log("=" * 60, log_fp)
    log("📄 Omni OCR Benchmark — 1000 Document Benchmark", log_fp)
    log("=" * 60, log_fp)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        log("  ⚠ huggingface_hub not installed. Skipping.", log_fp)
        return

    repo_id = "getomni/ocr-benchmark"
    tmp_dir = Path(base_dir) / "_tmp_omni"
    tmp_dir.mkdir(exist_ok=True)

    try:
        log("  Listing files...", log_fp)
        all_files = list_repo_files(repo_id, repo_type="dataset")
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]

        if not image_files:
            log("  ⚠ No image files found directly. Check repo structure.", log_fp)
            log(f"  ℹ Available files (first 20): {all_files[:20]}", log_fp)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        sample_files = random.sample(image_files, min(n_samples * 3, len(image_files)))
        downloaded = []
        for fname in sample_files:
            try:
                dl_path = hf_hub_download(repo_id, fname, repo_type="dataset",
                                          local_dir=str(tmp_dir))
                downloaded.append(Path(dl_path))
            except Exception:
                continue

        # Also try to get ground truth markdown
        md_files = [f for f in all_files if f.endswith('.md') and 'readme' not in f.lower()]

        # Distribute across categories
        invoice_dest = Path(base_dir) / "01_printed_english" / "invoices"
        report_dest = Path(base_dir) / "01_printed_english" / "reports"
        receipt_dest = Path(base_dir) / "06_mixed_content" / "receipts"

        for i, img in enumerate(downloaded):
            if i < n_samples:
                shutil.copy2(img, invoice_dest / f"omni_{img.name}")
            elif i < n_samples * 2:
                shutil.copy2(img, report_dest / f"omni_{img.name}")
            else:
                shutil.copy2(img, receipt_dest / f"omni_{img.name}")

        log(f"  ✅ Distributed {len(downloaded)} samples across invoices/reports/receipts", log_fp)

    except Exception as e:
        log(f"  ❌ Failed: {e}", log_fp)

    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_score_bench(base_dir, n_samples, log_fp):
    """
    SCORE-Bench — Real-world documents with expert annotations
    Source: HuggingFace (unstructured-io/SCORE-Bench)
    Auto-downloadable: YES
    """
    log("=" * 60, log_fp)
    log("📄 SCORE-Bench — Enterprise Document Benchmark", log_fp)
    log("=" * 60, log_fp)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        log("  ⚠ huggingface_hub not installed. Skipping.", log_fp)
        return

    repo_id = "unstructured-io/SCORE-Bench"
    tmp_dir = Path(base_dir) / "_tmp_score"
    tmp_dir.mkdir(exist_ok=True)

    try:
        log("  Listing files...", log_fp)
        all_files = list_repo_files(repo_id, repo_type="dataset")
        doc_files = [f for f in all_files
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff'))]

        if not doc_files:
            log(f"  ⚠ No document files found. Available: {all_files[:15]}", log_fp)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        sample_files = random.sample(doc_files, min(n_samples * 2, len(doc_files)))
        downloaded = []
        for fname in sample_files:
            try:
                dl_path = hf_hub_download(repo_id, fname, repo_type="dataset",
                                          local_dir=str(tmp_dir))
                downloaded.append(Path(dl_path))
            except Exception:
                continue

        nested_dest = Path(base_dir) / "02_complex_tables" / "nested_merged"
        lowq_dest = Path(base_dir) / "05_low_quality_scans" / "noisy"

        for i, doc in enumerate(downloaded):
            if i < n_samples:
                shutil.copy2(doc, nested_dest / f"score_{doc.name}")
            else:
                shutil.copy2(doc, lowq_dest / f"score_{doc.name}")

        log(f"  ✅ Distributed {len(downloaded)} samples → nested_merged + low_quality", log_fp)

    except Exception as e:
        log(f"  ❌ Failed: {e}", log_fp)

    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_ocrbench(base_dir, n_samples, log_fp):
    """
    OCRBench v2 — Multimodal OCR Evaluation
    Source: github.com/Yuliang-Liu/MultimodalOCR
    Auto-downloadable: Partial (via HuggingFace)
    """
    log("=" * 60, log_fp)
    log("📄 OCRBench v2 — Multimodal OCR Benchmark", log_fp)
    log("=" * 60, log_fp)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        log("  ⚠ huggingface_hub not installed. Skipping.", log_fp)
        return

    # OCRBench data is on HuggingFace
    repo_id = "echo840/OCRBench"
    tmp_dir = Path(base_dir) / "_tmp_ocrbench"
    tmp_dir.mkdir(exist_ok=True)

    try:
        log("  Listing files...", log_fp)
        all_files = list_repo_files(repo_id, repo_type="dataset")
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            log(f"  ⚠ No images found. Available: {all_files[:15]}", log_fp)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        sample_files = random.sample(image_files, min(n_samples * 2, len(image_files)))
        downloaded = []
        for fname in sample_files:
            try:
                dl_path = hf_hub_download(repo_id, fname, repo_type="dataset",
                                          local_dir=str(tmp_dir))
                downloaded.append(Path(dl_path))
            except Exception:
                continue

        mixed_dest = Path(base_dir) / "06_mixed_content" / "text_and_images"
        printed_dest = Path(base_dir) / "01_printed_english" / "letters_emails"

        for i, img in enumerate(downloaded):
            if i < n_samples:
                shutil.copy2(img, mixed_dest / f"ocrbench_{img.name}")
            else:
                shutil.copy2(img, printed_dest / f"ocrbench_{img.name}")

        log(f"  ✅ Distributed {len(downloaded)} samples → mixed_content + printed_english", log_fp)

    except Exception as e:
        log(f"  ❌ Failed: {e}", log_fp)

    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_devanagari_handwritten(base_dir, n_samples, log_fp):
    """
    Devanagari Handwritten Character Dataset
    Source: UCI ML Repository
    Auto-downloadable: YES
    """
    log("=" * 60, log_fp)
    log("📄 Devanagari Handwritten Character Dataset", log_fp)
    log("=" * 60, log_fp)

    import requests, zipfile, io

    url = "https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip"
    tmp_dir = Path(base_dir) / "_tmp_devanagari"
    tmp_dir.mkdir(exist_ok=True)

    log("  Downloading Devanagari dataset (may take a minute)...", log_fp)
    try:
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(tmp_dir)
        log("  ✅ Downloaded and extracted.", log_fp)
    except Exception as e:
        log(f"  ❌ Failed: {e}", log_fp)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    images = list(tmp_dir.rglob("*.png")) + list(tmp_dir.rglob("*.jpg"))
    if not images:
        log("  ⚠ No images found.", log_fp)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # Pick samples from different character classes
    by_class = {}
    for img in images:
        cls = img.parent.name
        by_class.setdefault(cls, []).append(img)

    dest = Path(base_dir) / "03_handwritten" / "hindi_devanagari"
    count = 0
    classes = list(by_class.keys())
    random.shuffle(classes)
    for cls in classes:
        if count >= n_samples * 2:
            break
        sample = random.sample(by_class[cls], min(2, len(by_class[cls])))
        for img in sample:
            shutil.copy2(img, dest / f"devanagari_{cls}_{img.name}")
            count += 1

    log(f"  ✅ Copied {count} samples → 03_handwritten/hindi_devanagari/", log_fp)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_indicphotoocr_samples(base_dir, n_samples, log_fp):
    """
    IndicPhotoOCR / Bharat Scene Text Dataset
    Source: github.com/Bhashini-IITJ/IndicPhotoOCR
    Auto-downloadable: Partial (test images from repo)
    """
    log("=" * 60, log_fp)
    log("📄 IndicPhotoOCR — Indian Language Scene Text", log_fp)
    log("=" * 60, log_fp)

    import requests

    # Download test images from the GitHub repo
    repo_raw = "https://raw.githubusercontent.com/Bhashini-IITJ/IndicPhotoOCR/main/test_images"

    # Known test images in the repo
    test_images = [
        "image_141.jpg", "image_142.jpg", "image_143.jpg",
        "image_144.jpg", "image_145.jpg", "image_146.jpg",
        "image_147.jpg", "image_148.jpg", "image_149.jpg",
        "image_150.jpg",
    ]

    hindi_dest = Path(base_dir) / "04_indian_languages" / "hindi"
    telugu_dest = Path(base_dir) / "04_indian_languages" / "telugu"
    mixed_dest = Path(base_dir) / "04_indian_languages" / "mixed_bilingual"

    downloaded = 0
    for img_name in test_images[:n_samples * 2]:
        url = f"{repo_raw}/{img_name}"
        # Distribute across Indian language folders
        if downloaded < n_samples:
            dest = hindi_dest / f"indic_{img_name}"
        elif downloaded < n_samples + 3:
            dest = telugu_dest / f"indic_{img_name}"
        else:
            dest = mixed_dest / f"indic_{img_name}"

        if safe_download(url, dest):
            downloaded += 1

    if downloaded > 0:
        log(f"  ✅ Downloaded {downloaded} scene text samples → indian_languages/", log_fp)
    else:
        log("  ⚠ Could not download from repo. Try cloning manually:", log_fp)
        log("    git clone https://github.com/Bhashini-IITJ/IndicPhotoOCR.git", log_fp)


def download_sroie_info(base_dir, n_samples, log_fp):
    """
    SROIE — Scanned Receipts OCR and Information Extraction
    Auto-downloadable: NO (requires competition registration)
    """
    log("=" * 60, log_fp)
    log("📄 SROIE — Receipt Dataset (MANUAL DOWNLOAD REQUIRED)", log_fp)
    log("=" * 60, log_fp)
    log("  ℹ SROIE requires registration at:", log_fp)
    log("    https://rrc.cvc.uab.es/?ch=13", log_fp)
    log("  ℹ Alternative: search 'SROIE dataset' on Kaggle or HuggingFace", log_fp)
    log(f"  → Place {n_samples}-8 receipt images in: 06_mixed_content/receipts/", log_fp)
    log(f"  → Place ground truth in: ground_truth/06_mixed_content/", log_fp)

    # Try HuggingFace mirror
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        # Some community uploads exist
        for repo in ["mychen76/sroie2019", "darentang/sroie"]:
            try:
                files = list_repo_files(repo, repo_type="dataset")
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
                if image_files:
                    log(f"  ✅ Found SROIE mirror on HuggingFace: {repo}", log_fp)
                    sample = random.sample(image_files, min(n_samples, len(image_files)))
                    tmp_dir = Path(base_dir) / "_tmp_sroie"
                    tmp_dir.mkdir(exist_ok=True)
                    dest = Path(base_dir) / "06_mixed_content" / "receipts"
                    count = 0
                    for fname in sample:
                        try:
                            dl_path = hf_hub_download(repo, fname, repo_type="dataset",
                                                      local_dir=str(tmp_dir))
                            shutil.copy2(dl_path, dest / f"sroie_{Path(fname).name}")
                            count += 1
                        except Exception:
                            continue
                    log(f"  ✅ Downloaded {count} receipt samples from mirror", log_fp)
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return
            except Exception:
                continue
    except ImportError:
        pass

    log("  ⚠ No auto-download mirror found. Please download manually.", log_fp)


def download_iiithw_info(base_dir, n_samples, log_fp):
    """
    IIIT-HW — Handwritten Word Dataset for Indian Languages
    Auto-downloadable: NO (requires institutional access)
    """
    log("=" * 60, log_fp)
    log("📄 IIIT-HW — Indian Handwriting (MANUAL DOWNLOAD REQUIRED)", log_fp)
    log("=" * 60, log_fp)
    log("  ℹ The IIIT Handwritten Word dataset must be requested from:", log_fp)
    log("    CVIT, IIIT Hyderabad — https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data", log_fp)
    log("  ℹ Languages available: Hindi, Tamil, Telugu, Bengali, Malayalam, Gujarati, Kannada", log_fp)
    log(f"  → Place Hindi samples in: 03_handwritten/hindi_devanagari/", log_fp)
    log(f"  → Place Tamil/Bengali/Telugu/Malayalam in: 03_handwritten/other_indic/", log_fp)
    log(f"  → Place Telugu printed/scene text in: 04_indian_languages/telugu/", log_fp)
    log(f"  → Place ground truth .txt files in corresponding ground_truth/ folders", log_fp)


def download_iam_info(base_dir, n_samples, log_fp):
    """
    IAM Handwriting Database — English Handwritten Text
    Auto-downloadable: NO (requires registration)
    """
    log("=" * 60, log_fp)
    log("📄 IAM Handwriting — English Handwriting (MANUAL DOWNLOAD REQUIRED)", log_fp)
    log("=" * 60, log_fp)
    log("  ℹ Register at: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database", log_fp)
    log("  ℹ Free for research use after registration", log_fp)
    log(f"  → Place {n_samples}-7 samples in: 03_handwritten/english/", log_fp)
    log(f"  → Place ground truth in: ground_truth/03_handwritten/", log_fp)


def download_rvlcdip_samples(base_dir, n_samples, log_fp):
    """
    RVL-CDIP — 400K Document Classification Dataset
    Source: HuggingFace (rvl_cdip or aharley/rvl_cdip)
    Auto-downloadable: YES (subset via HuggingFace)
    """
    log("=" * 60, log_fp)
    log("📄 RVL-CDIP — Document Classification Dataset", log_fp)
    log("=" * 60, log_fp)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        log("  ⚠ huggingface_hub not installed. Skipping.", log_fp)
        return

    repo_id = "aharley/rvl_cdip"
    tmp_dir = Path(base_dir) / "_tmp_rvlcdip"
    tmp_dir.mkdir(exist_ok=True)

    try:
        log("  Listing files (this is a large dataset, fetching index)...", log_fp)
        all_files = list_repo_files(repo_id, repo_type="dataset")

        # Look for image files or parquet data
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]
        parquet_files = [f for f in all_files if f.lower().endswith('.parquet')]

        if image_files:
            sample_files = random.sample(image_files, min(n_samples * 3, len(image_files)))
            downloaded = []
            for fname in sample_files:
                try:
                    dl_path = hf_hub_download(repo_id, fname, repo_type="dataset",
                                              local_dir=str(tmp_dir))
                    downloaded.append(Path(dl_path))
                except Exception:
                    continue

            contract_dest = Path(base_dir) / "01_printed_english" / "contracts"
            report_dest = Path(base_dir) / "01_printed_english" / "reports"
            letter_dest = Path(base_dir) / "01_printed_english" / "letters_emails"

            for i, img in enumerate(downloaded):
                if i < n_samples:
                    shutil.copy2(img, contract_dest / f"rvl_{img.name}")
                elif i < n_samples * 2:
                    shutil.copy2(img, report_dest / f"rvl_{img.name}")
                else:
                    shutil.copy2(img, letter_dest / f"rvl_{img.name}")

            log(f"  ✅ Distributed {len(downloaded)} samples across printed_english/", log_fp)
        elif parquet_files:
            log("  ℹ Dataset uses Parquet format. Use HuggingFace datasets library to load:", log_fp)
            log("    from datasets import load_dataset", log_fp)
            log("    ds = load_dataset('aharley/rvl_cdip', split='test[:50]')", log_fp)
            log("    Then save individual images to the appropriate folders.", log_fp)
        else:
            log(f"  ⚠ Unexpected format. Files: {all_files[:10]}", log_fp)

    except Exception as e:
        log(f"  ❌ Failed: {e}", log_fp)

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Summary & Manifest
# ---------------------------------------------------------------------------

def generate_summary(base_dir, log_fp):
    """Count files in each folder and generate a summary."""
    log("\n" + "=" * 60, log_fp)
    log("📊 DOWNLOAD SUMMARY", log_fp)
    log("=" * 60, log_fp)

    total = 0
    manifest = {}
    for folder in FOLDER_STRUCTURE:
        if folder.startswith("ground_truth"):
            continue
        full_path = Path(base_dir) / folder
        files = [f for f in full_path.iterdir() if f.is_file()] if full_path.exists() else []
        count = len(files)
        total += count
        manifest[folder] = [f.name for f in files]
        status = "✅" if count > 0 else "⬜"
        log(f"  {status} {folder}: {count} files", log_fp)

    log(f"\n  Total documents downloaded: {total}", log_fp)
    log(f"  Target: 100-150 documents", log_fp)

    if total < 100:
        log(f"\n  ⚠ {100 - total} more documents needed to reach minimum target.", log_fp)
        log("  Suggestions:", log_fp)
        log("  - Download manual datasets (SROIE, IAM, IIIT-HW) — see instructions above", log_fp)
        log("  - Add 10-15 organization internal documents to 07_org_documents/", log_fp)
        log("  - Re-run script with --samples 8 for more per source", log_fp)

    # Save manifest
    manifest_path = Path(base_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"generated": datetime.now().isoformat(), "total_files": total,
                    "folders": manifest}, f, indent=2)
    log(f"\n  📄 Manifest saved to: {manifest_path}", log_fp)

    # Reminder about manual steps
    log("\n" + "=" * 60, log_fp)
    log("📋 MANUAL STEPS REMAINING", log_fp)
    log("=" * 60, log_fp)
    log("  1. Download SROIE receipts → 06_mixed_content/receipts/", log_fp)
    log("  2. Download IAM handwriting → 03_handwritten/english/", log_fp)
    log("  3. Request IIIT-HW dataset → 03_handwritten/ (Hindi, Tamil, Telugu, Bengali)", log_fp)
    log("  4. ⭐ Add 10-15 INTERNAL org documents → 07_org_documents/", log_fp)
    log("  5. Create ground truth for all documents without existing annotations", log_fp)
    log("  6. Log everything in the OCR_Test_Dataset_Tracker.xlsx spreadsheet", log_fp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and organize OCR evaluation test datasets"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Samples per category per source (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip specific downloaders (e.g., --skip funsd devanagari)")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.output_dir)
    n_samples = args.samples

    print(f"""
╔══════════════════════════════════════════════════════════╗
║     OCR Evaluation — Test Dataset Downloader            ║
╠══════════════════════════════════════════════════════════╣
║  Output:  {base_dir:<47}║
║  Samples: {n_samples} per category per source{' ' * 27}║
╚══════════════════════════════════════════════════════════╝
    """)

    # Install dependencies
    ensure_pip_package("requests")
    ensure_pip_package("huggingface_hub", "huggingface_hub")
    ensure_pip_package("tqdm")

    # Create folder structure
    create_folders(base_dir)
    print(f"✅ Created folder structure in {base_dir}\n")

    # Open log file
    log_path = Path(base_dir) / LOG_FILE
    log_fp = open(log_path, "w")

    # Run downloaders
    downloaders = {
        "funsd": download_funsd,
        "omnidocbench": download_omnidocbench,
        "omni": download_omni_benchmark,
        "score": download_score_bench,
        "ocrbench": download_ocrbench,
        "devanagari": download_devanagari_handwritten,
        "indicphoto": download_indicphotoocr_samples,
        "rvlcdip": download_rvlcdip_samples,
        "sroie": download_sroie_info,
        "iiithw": download_iiithw_info,
        "iam": download_iam_info,
    }

    skip = set(s.lower() for s in args.skip)
    for name, func in downloaders.items():
        if name in skip:
            log(f"⏭ Skipping {name} (user requested)", log_fp)
            continue
        try:
            func(base_dir, n_samples, log_fp)
        except Exception as e:
            log(f"❌ Error in {name}: {e}", log_fp)
        print()

    # Generate summary
    generate_summary(base_dir, log_fp)
    log_fp.close()

    print(f"\n📄 Full log saved to: {log_path}")
    print(f"📂 Dataset folder: {base_dir}")
    print("\nDone! Next step: review downloaded files and fill in the tracker spreadsheet.")


if __name__ == "__main__":
    main()
