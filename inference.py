"""
HunyuanOCR SAMPLE TEST — validates the full pipeline on a tiny subset
before launching full training.

Changes vs hunyuan_odia_ocr_train_v2.py:
  - Only 100 rows from each dataset (fast normalization, ~seconds)
  - MAX_STEPS = 20  (just to confirm loss goes down)
  - EVAL_SAMPLES = 10
  - LABELS BUG FIXED: labels derived from input_ids, not separate tokenization
  - No HF push on sample run

Run on server:
  cd /root/hunyuan_odia && source venv/bin/activate
  python hunyuan_sample_test.py
"""

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_MODEL    = "tencent/HunyuanOCR"
OUTPUT_DIR    = "/root/hunyuan_odia/output_sample"

DATASET_SYNTHETIC = "OdiaGenAIOCR/synthetic_data"
DATASET_MERGED    = "OdiaGenAIOCR/odia-ocr-merged"
OCR_PROMPT = "Extract all Odia text from this image. Return only the Odia text, nothing else."

SAMPLE_ROWS   = 100   # rows from each dataset
MAX_STEPS     = 20
EVAL_STEPS    = 10
SAVE_STEPS    = 20
BATCH_SIZE    = 1
GRAD_ACCUM    = 2
LEARNING_RATE = 1e-4
WARMUP_STEPS  = 5
MAX_NEW_TOKENS = 128
EVAL_SAMPLES  = 10
MAX_SEQ_LEN   = 1024

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# HUNYUAN IMPORT
# ──────────────────────────────────────────────
try:
    from transformers import HunYuanVLForConditionalGeneration
    print("[OK] HunYuanVLForConditionalGeneration available.")
except ImportError:
    os.system(
        'pip install -q "git+https://github.com/huggingface/transformers'
        '@82a06db03535c49aa987719ed0746a76093b1ec4"'
    )
    from transformers import HunYuanVLForConditionalGeneration

from transformers import AutoProcessor, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model

# ──────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────
MAX_IMG_SIZE = 512  # Resize images to avoid HunyuanVL processor off-by-one on image tokens

def ensure_pil(img, max_size=MAX_IMG_SIZE):
    """Convert to RGB PIL image and resize so the longest edge <= max_size."""
    pil = None
    if img is None:
        return None
    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    elif isinstance(img, bytes):
        from io import BytesIO
        pil = Image.open(BytesIO(img)).convert("RGB")
    elif isinstance(img, str) and os.path.exists(img):
        pil = Image.open(img).convert("RGB")
    else:
        try:
            import numpy as np
            if isinstance(img, np.ndarray):
                pil = Image.fromarray(img).convert("RGB")
        except Exception:
            pass
    if pil is None:
        return None
    # Resize so longest edge <= max_size to limit image token count
    w, h = pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil


def compute_metrics(predictions, references):
    try:
        import jiwer
        cer = round(jiwer.cer(references, predictions), 4)
        wer = round(jiwer.wer(references, predictions), 4)
    except Exception:
        cer = wer = None
    exact = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    result = {"exact_match": round(exact / max(len(predictions), 1), 4), "n": len(predictions)}
    if cer is not None:
        result["CER"] = cer
        result["WER"] = wer
    return result


# ──────────────────────────────────────────────
# LOAD DATASETS (tiny subset)
# ──────────────────────────────────────────────
def load_sample_datasets():
    print(f"\n[DATA] Loading {SAMPLE_ROWS} rows from each dataset...")

    ds1 = load_dataset(DATASET_SYNTHETIC, split=f"train[:{SAMPLE_ROWS}]")
    ds1 = ds1.map(
        lambda x: {"image": x["image"], "gt_text": x["extracted_text"]},
        remove_columns=ds1.column_names,
        desc="Normalize synthetic_data",
    )

    ds2 = load_dataset(DATASET_MERGED, split=f"train[:{SAMPLE_ROWS}]")
    ds2 = ds2.map(
        lambda x: {"image": x["image"], "gt_text": x["text"]},
        remove_columns=ds2.column_names,
        desc="Normalize odia-ocr-merged",
    )

    combined = concatenate_datasets([ds1, ds2]).shuffle(seed=42)
    # Filter out rows with missing image or empty text
    combined = combined.filter(lambda x: x["gt_text"] is not None and len((x["gt_text"] or "").strip()) > 0)
    n = len(combined)
    val_size = max(10, int(n * 0.1))
    train_ds = combined.select(range(0, n - val_size))
    val_ds   = combined.select(range(n - val_size, n))
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


# ──────────────────────────────────────────────
# DATA COLLATOR  (labels bug fixed)
# ──────────────────────────────────────────────
@dataclass
class OdiaOCRCollator:
    processor: Any
    max_length: int = MAX_SEQ_LEN

    def __call__(self, batch: List[Dict]) -> Dict:
        valid_images, texts_input = [], []

        for sample in batch:
            image = ensure_pil(sample.get("image"))
            gt    = (sample.get("gt_text") or "").strip()
            if image is None or not gt:
                continue

            # FIX: Include GT text as assistant response so the model trains on it
            messages = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  OCR_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": gt,
                },
            ]
            try:
                full_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                valid_images.append(image)
                texts_input.append(full_text)
            except Exception as e:
                print(f"[WARN] Skipping sample in collator (apply_chat_template): {e}")
                continue

        if not valid_images:
            return {}

        # Try processing; if the HunyuanVL image-token indexing fails,
        # shrink images by 50% and retry (up to 3 times).
        for attempt in range(3):
            try:
                inputs = self.processor(
                    text=texts_input,
                    images=valid_images,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                break  # success
            except (IndexError, Exception) as e:
                print(f"[WARN] Processor attempt {attempt+1} failed: {e}. Shrinking images.")
                valid_images = [img.resize((img.width // 2, img.height // 2), Image.LANCZOS) for img in valid_images]
        else:
            return {}

        # Build labels — mask prompt tokens, train only on GT text + hy_Assistant token.
        # HunyuanOCR chat template structure:
        #   [BOS][image_tokens][OCR_PROMPT]<｜hy_User｜>[GT_TEXT]<｜hy_Assistant｜>
        # Everything up to and including <｜hy_User｜> (ID 120006) is prompt → masked.
        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id

        HY_USER_ID = 120006  # <｜hy_User｜> ends the user/prompt turn
        for i in range(labels.shape[0]):
            ids = labels[i].tolist()
            found = False
            for j in range(len(ids) - 1, -1, -1):
                if ids[j] == HY_USER_ID:
                    labels[i, : j + 1] = -100
                    found = True
                    break
            if not found:
                print(f"[WARN] sample {i}: <|hy_User|> not found, masking all tokens")
                labels[i, :] = -100

        # Also mask padding tokens
        if pad_id is not None:
            labels[labels == pad_id] = -100
        inputs["labels"] = labels
        return dict(inputs)

# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────
def run_evaluation(model, processor, dataset, num_samples=EVAL_SAMPLES, tag="eval"):
    model.eval()
    predictions, references = [], []
    for idx in tqdm(range(min(num_samples, len(dataset))), desc=f"[EVAL] {tag}"):
        sample = dataset[idx]
        image  = ensure_pil(sample.get("image"))
        gt     = (sample.get("gt_text") or "").strip()
        if image is None or not gt:
            continue
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  OCR_PROMPT},
            ]},
        ]
        texts  = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        inp    = processor(text=texts, images=[image], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_ids = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inp["input_ids"], gen_ids)]
        pred    = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        predictions.append(pred)
        references.append(gt)

    metrics = compute_metrics(predictions, references)
    print(f"\n[EVAL] {tag}: {metrics}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"eval_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "samples": [{"pred": p, "ref": r} for p, r in zip(predictions, references)]}, f, ensure_ascii=False, indent=2)
    return metrics


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print("HunyuanOCR SAMPLE TEST (100 rows, 20 steps)")
    print(f"Device: {DEVICE}  |  dtype: {DTYPE}")
    print(f"{'='*60}\n")

    train_ds, val_ds = load_sample_datasets()

    # Load processor + model
    print(f"\n[MODEL] Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=False)

    print(f"[MODEL] Loading HunYuanVLForConditionalGeneration...")
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        attn_implementation="eager",
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.config.use_cache = False

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Baseline eval
    print("\n[EVAL] Baseline (before training)...")
    run_evaluation(model, processor, val_ds, num_samples=EVAL_SAMPLES, tag="baseline")

    # Training
    collator = OdiaOCRCollator(processor=processor)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=SAVE_STEPS,
        eval_strategy="no",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    print(f"\n[TRAIN] Starting ({MAX_STEPS} steps)...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()
    print("\n[TRAIN] Sample run COMPLETE. Check loss went down.")
    print(f"Logs at: {OUTPUT_DIR}")
    print("\nIf successful, run the full training with hunyuan_odia_ocr_train_v2.py (with labels bug fixed).")


if __name__ == "__main__":
    main()
