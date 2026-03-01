---
language:
- or
license: apache-2.0
tags:
- ocr
- odia
- hunyuan
- fine-tuned
- lora
base_model: tencent/HunyuanOCR
datasets:
- OdiaGenAIOCR/odia-ocr-merged
---

# HunyuanOCR Fine-tuned for Odia OCR

Fine-tuned [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) on the
[OdiaGenAIOCR/odia-ocr-merged](https://huggingface.co/datasets/OdiaGenAIOCR/odia-ocr-merged)
dataset using LoRA (r=64, alpha=128).

**GitHub:** [shantipriyap/hunyuan_odia_ocr](https://github.com/shantipriyap/hunyuan_odia_ocr)

---

## Evaluation Results

| Checkpoint | Steps | CER↓ | WER↓ | Notes |
|---|---|---|---|---|
| Baseline (zero-shot) | 0 | 0.9111 | 0.9467 | HunyuanOCR, no fine-tuning |
| v5 (r=32) | 1000 | **0.7577** | **0.846** | Best word-level CER so far |
| v7 (r=32) | 3200 | 0.7909 | 0.941 | r=32 capacity ceiling |
| v8 baseline | 0 | 1.1188 | 1.4385 | Before v8 training |
| **v8 ckpt-3250** (latest) | 3250 | *in training* | — | Loss ~0.93 best; 67% done |

> **Note on evaluation:** Training uses word-level crops (`OdiaGenAIOCR/odia-ocr-merged`). The `Iftesha/odia-ocr-benchmark` dataset contains paragraph-level images — a different domain where this model scores CER ~0.99 (expected, not trained on paragraphs).

---

## Inference Samples *(checkpoint-3250, step 67% of training)*

Samples from [Iftesha/odia-ocr-benchmark](https://huggingface.co/datasets/Iftesha/odia-ocr-benchmark) — paragraph-level images. Evaluated at ckpt-3250.

> ⚠️ This model was trained on **word-level crops**. Paragraph images are out-of-training-domain, so CER is high. The model correctly outputs valid Odia words but cannot yet handle full paragraphs.

| Image | Ground Truth (truncated) | Prediction | CER | Category |
|:---:|:---|:---|:---:|:---:|
| <img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples3/benchmark_01.jpg" width="220"/> | ପରିଶ୍ରମ ହେ ଅଳସୁଆ, ପିମ୍ପୁଡ଼ିକୁ ଯାଅ; ତା'ର ମାର୍ଗ ବିଷୟରେ ଚିନ୍ତା କର ... | ପରିବେଷଣ | 0.98 | Digital |
| <img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples3/benchmark_02.jpg" width="220"/> | ଆଜି ସନ୍ଧ୍ୟାର 20 ବଡ଼ ଖବର ସ୍କୁଲ ଖୋଲିବା ଓ ବନ୍ଦ ସମୟ ବଦଳିଲା ... | ବିଶ୍ୱାସ | 0.97 | Digital |
| <img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples3/benchmark_03.jpg" width="220"/> | ଆଜି ୧୭ ମଇ ଶୁକ୍ରବାର ଚାଷୀଙ୍କୁ ୫୦୦୦୦ ଟଙ୍କା କ୍ଷତିପୂରଣ ... | ବାଲିକ | 0.97 | Digital |
| <img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples3/benchmark_04.jpg" width="220"/> | ଓଁ ଶ୍ରୀପରମାତ୍ମନେ ନମଃ ଶ୍ରୀମଦ୍ଭଗବଦ୍‌ଗୀତା ଅଥ କରନ୍ୟାସଃ ... | ବାହାରିଥିଲେ | 0.99 | Book |

---

## Training Loss Curve (v8, r=64)

| Step | Loss |
|---|---|
| 10 | 2.3695 |
| 500 | ~1.18 |
| 910 | 1.0948 |
| 1500 | ~1.11 |
| **2100** | **0.9964** ← first sub-1.0 |
| 2580 | 0.9339 ← best so far |
| 2750 | 1.0291 |
| 3000 | ~0.979 |
| **3250** | **~0.979** (67% done, in training) |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | tencent/HunyuanOCR |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Learning rate | 2e-4 |
| Warmup steps | 100 |
| Max steps | 5000 |
| Batch size | 1 (grad_accum=4) |
| Max seq len | 2048 |

---

## Quick Start

```python
import torch
from PIL import Image
from transformers import HunYuanVLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE  = "tencent/HunyuanOCR"
CKPT  = "shantipriya/hunyuan-ocr-odia"

base  = HunYuanVLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16,
    attn_implementation="eager", device_map="auto")
model = PeftModel.from_pretrained(base, CKPT)
model.eval()
proc  = AutoProcessor.from_pretrained(BASE, use_fast=False)

img   = Image.open("odia_image.jpg").convert("RGB")
msgs  = [
    {"role": "system", "content": ""},        # required
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": "Extract all Odia text from this image. Return only the Odia text."},
    ]},
]
text   = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
result = proc.batch_decode(
    [gen[0][inputs["input_ids"].shape[1]:]], skip_special_tokens=True
)[0].strip()
print(result)
```

> **Note:** The empty `system` message is **required** — omitting it causes a `position_ids` dimension error.

---

## License

Apache 2.0
