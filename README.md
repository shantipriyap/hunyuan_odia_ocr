# Hunyuan Odia OCR

Fine-tuning [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) on the Odia script using LoRA.

**Model on HuggingFace:** [shantipriya/hunyuan-ocr-odia](https://huggingface.co/shantipriya/hunyuan-ocr-odia)  
**Dataset:** [OdiaGenAIOCR/odia-ocr-merged](https://huggingface.co/datasets/OdiaGenAIOCR/odia-ocr-merged)

---

## Repository Contents

| File | Description |
|---|---|
| `hunyuan_odia_ocr_train_v8.py` | Latest training script (LoRA r=64, 5000 steps) |
| `inference.py` | Run inference with a fine-tuned checkpoint |
| `eval.py` | Evaluate CER/WER on test split |
| `requirements.txt` | Python dependencies |
| `setup_hunyuanocr_mac.sh` | macOS dev environment setup |

---

## Results

| Checkpoint | Steps | CER | WER | Notes |
|---|---|---|---|---|
| Baseline (zero-shot) | 0 | 0.9111 | 0.9467 | No fine-tuning |
| v5 (r=32) | 1000 | **0.7577** | **0.846** | Best CER so far |
| v7 (r=32) | 3200 | 0.7909 | 0.941 | r=32 capacity ceiling |
| v8 (r=64) | 5000 | *in training* | — | Current run |

---

## Inference Samples *(checkpoint-1250, step 25% of training)*

> Images are from the `OdiaGenAIOCR/odia-ocr-merged` test split.  
> Full evaluation at step 5000 will update the CER column above.

<table>
  <tr>
    <th>Image</th>
    <th>Ground Truth</th>
    <th>Prediction (ckpt-1250)</th>
    <th>CER</th>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples/sample_01.jpg" width="220"/></td>
    <td>ଚଳେଇବାପାଇଁ</td>
    <td>ବାହୁଡ଼ା</td>
    <td>0.90</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples/sample_02.jpg" width="220"/></td>
    <td>ପୁନର୍ନଭା</td>
    <td>ବାହୁ</td>
    <td>1.00</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples/sample_03.jpg" width="220"/></td>
    <td>ଟ୍ରଷ୍ଟର</td>
    <td>ବାହୁ</td>
    <td>1.00</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples/sample_04.jpg" width="220"/></td>
    <td>ସୀମାର</td>
    <td>E151182</td>
    <td>1.40</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples/sample_05.jpg" width="220"/></td>
    <td>ଟ୍ୟୁବରକୁଲୋସିସ</td>
    <td>ବାହୁଡ଼ା</td>
    <td>0.85</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/shantipriya/hunyuan-ocr-odia/resolve/main/samples/sample_06.jpg" width="220"/></td>
    <td>ଟ୍ରାନ୍ସଭର୍ସସ</td>
    <td>ବେକ୍ଟର</td>
    <td>0.83</td>
  </tr>
</table>

*Model outputs valid Odia script at step 1250. Loss dropped to 0.9964 by step 2100 — quality expected to improve significantly by step 5000.*

---

## Training

### Requirements

- Python 3.12+
- NVIDIA GPU (A100 80GB recommended)
- CUDA 12+

### Setup

```bash
git clone https://github.com/shantipriyap/hunyuan_odia_ocr.git
cd hunyuan_odia_ocr
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run Training

```bash
python hunyuan_odia_ocr_train_v8.py
```

Key hyperparameters (edit at the top of the script):

| Parameter | Value |
|---|---|
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Learning rate | 2e-4 |
| Warmup steps | 100 |
| Max steps | 5000 |
| Max seq len | 2048 |

---

## Inference

```python
import torch
from PIL import Image
from transformers import HunYuanVLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE = "tencent/HunyuanOCR"
CKPT = "shantipriya/hunyuan-ocr-odia"   # or path to local checkpoint

base  = HunYuanVLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16,
    attn_implementation="eager", device_map="auto")
model = PeftModel.from_pretrained(base, CKPT)
model.eval()
proc  = AutoProcessor.from_pretrained(BASE, use_fast=False)

img  = Image.open("odia_image.jpg").convert("RGB")
msgs = [
    {"role": "system", "content": ""},   # required -- do not omit
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

> **Note:** The empty `system` message is **required** to avoid a `position_ids` shape error
> in HunyuanOCR's attention layer.

---

## macOS Dev Setup

```bash
bash setup_hunyuanocr_mac.sh
```

---

## License

Apache 2.0
