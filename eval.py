#!/usr/bin/env python3
"""
Odia OCR Evaluation Script
==========================

Evaluate fine-tuned Qwen2.5-VL model on Odia OCR task.

Computes Character Error Rate (CER) and Word Error Rate (WER).
Includes post-processing to extract clean Odia text.

Dataset: shantipriya/odia-ocr-merged
Model: shantipriya/odia-ocr-qwen-finetuned
"""

import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import logging
from datetime import datetime
import json
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "shantipriya/odia-ocr-qwen-finetuned"
DATASET_NAME = "OdiaGenAIOCR/synthetic_data"
MAX_SAMPLES = None  # None for all samples

logger.info("="*70)
logger.info("📊 ODIA OCR EVALUATION")
logger.info("="*70)
logger.info(f"Model: {MODEL_ID}")
logger.info(f"Dataset: {DATASET_NAME}")
logger.info("="*70)

# Load model and processor
logger.info("\n📦 Loading model and processor...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# Load dataset
logger.info("\n📥 Loading dataset...")
try:
    dataset = load_dataset(DATASET_NAME, split="train")
    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    logger.info(f"✅ Loaded {len(dataset):,} samples")
except Exception as e:
    logger.error(f"❌ Failed to load dataset: {e}")
    raise


def extract_odia_text(text):
    """Extract Odia Unicode characters (U+0B00-U+0B7F)."""
    odia_chars = [char for char in text if '\u0B00' <= char <= '\u0B7F']
    return ''.join(odia_chars)


def character_error_rate(reference, hypothesis):
    """Calculate Character Error Rate (CER)."""
    if not reference:
        return 100.0 if hypothesis else 0.0

    insertions = 0
    deletions = 0
    substitutions = 0

    i, j = 0, 0
    while i < len(reference) and j < len(hypothesis):
        if reference[i] == hypothesis[j]:
            i += 1
            j += 1
        else:
            substitutions += 1
            i += 1
            j += 1

    deletions = len(reference) - i
    insertions = len(hypothesis) - j

    errors = insertions + deletions + substitutions
    cer = (errors / len(reference)) * 100 if reference else 0

    return min(100.0, cer)


def word_error_rate(reference, hypothesis):
    """Calculate Word Error Rate (WER)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 100.0 if hyp_words else 0.0

    insertions = 0
    deletions = 0
    substitutions = 0

    i, j = 0, 0
    while i < len(ref_words) and j < len(hyp_words):
        if ref_words[i] == hyp_words[j]:
            i += 1
            j += 1
        else:
            substitutions += 1
            i += 1
            j += 1

    deletions = len(ref_words) - i
    insertions = len(hyp_words) - j

    errors = insertions + deletions + substitutions
    wer = (errors / len(ref_words)) * 100 if ref_words else 0

    return min(100.0, wer)


# Evaluation
logger.info("\n🔄 Running evaluation...")
results = {
    "total_samples": 0,
    "successful": 0,
    "failed": 0,
    "cer_scores": [],
    "wer_scores": [],
    "exact_matches": 0,
    "samples": []
}

with torch.no_grad():
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            image = example.get("image")
            reference_text = example.get("text", "")

            # Validate inputs
            if image is None or not reference_text:
                results["failed"] += 1
                continue

            # Ensure image is PIL Image
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif hasattr(image, "convert"):
                image = image.convert("RGB")
            else:
                results["failed"] += 1
                continue

            # Generate prediction
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=256)
            raw_output = processor.decode(output[0], skip_special_tokens=True)

            # Extract clean Odia text
            predicted_text = extract_odia_text(raw_output)

            # Calculate metrics
            cer = character_error_rate(reference_text, predicted_text)
            wer = word_error_rate(reference_text, predicted_text)
            exact_match = 1 if predicted_text == reference_text else 0

            results["cer_scores"].append(cer)
            results["wer_scores"].append(wer)
            results["exact_matches"] += exact_match
            results["successful"] += 1
            results["total_samples"] += 1

            # Store first 5 samples for inspection
            if len(results["samples"]) < 5:
                results["samples"].append({
                    "index": idx,
                    "reference": reference_text[:100],
                    "predicted": predicted_text[:100],
                    "cer": round(cer, 2),
                    "wer": round(wer, 2),
                    "exact_match": exact_match
                })

        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            results["failed"] += 1

# Calculate statistics
if results["cer_scores"]:
    avg_cer = sum(results["cer_scores"]) / len(results["cer_scores"])
    avg_wer = sum(results["wer_scores"]) / len(results["wer_scores"])
    exact_match_rate = (results["exact_matches"] / results["total_samples"]) * 100
else:
    avg_cer = avg_wer = exact_match_rate = 0

# Log results
logger.info("\n" + "="*70)
logger.info("📊 EVALUATION RESULTS")
logger.info("="*70)
logger.info(f"\n✅ Processed: {results['successful']} samples")
logger.info(f"❌ Failed: {results['failed']} samples")
logger.info(f"\n📈 Metrics:")
logger.info(f"   Average CER: {avg_cer:.2f}%")
logger.info(f"   Average WER: {avg_wer:.2f}%")
logger.info(f"   Exact Match Rate: {exact_match_rate:.2f}%")
logger.info(f"\n📋 Sample Results:")
for sample in results["samples"]:
    logger.info(f"   Sample {sample['index']}: CER={sample['cer']}% WER={sample['wer']}%")

# Save results
output_file = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "model": MODEL_ID,
            "dataset": DATASET_NAME,
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "total_samples": results["total_samples"],
            "successful": results["successful"],
            "failed": results["failed"],
            "avg_cer": round(avg_cer, 2),
            "avg_wer": round(avg_wer, 2),
            "exact_match_rate": round(exact_match_rate, 2)
        },
        "samples": results["samples"]
    }, f, ensure_ascii=False, indent=2)

logger.info(f"\n💾 Results saved to: {output_file}")
logger.info("\n✨ Evaluation complete!")
