# Reading Text in Images for Visual Question Answering

Compare **OCR-first** and **vision-language (VLM)** approaches on **TextVQA**, with
analysis of failure cases where OCR is noisy or incomplete.

## Problem

Many VQA systems are good at recognizing objects and scenes but fail when the answer
depends on reading text inside the image — a street sign, a product label, a jersey
number. This project measures exactly when OCR helps, when a frozen vision-language
model can read the text on its own, and when combining both wins.

## What this project does

1. **OCR-first pipeline** — extract text tokens from each image (EasyOCR / Tesseract),
   feed `[question + OCR tokens]` to a text reasoner, and answer.
2. **VLM pipeline** — answer directly from `[image + question]` using a frozen
   multimodal model (BLIP-2), zero-shot, no OCR involved.
3. **Hybrid pipeline** — give the VLM the OCR tokens as an auxiliary hint.
4. **Evaluation** — official TextVQA accuracy (soft-voting over 10 human answers) + ANLS.
5. **Error analysis** — bucket results by OCR coverage of the ground-truth answer to
   show *where* and *why* each approach wins or loses.

## Results

Setup: 200 questions from the TextVQA 0.5.1 validation split, EasyOCR for text
extraction, `Salesforce/blip2-flan-t5-xl` as the frozen VLM, `google/flan-t5-xl` as the
text-only reader, run on a Colab T4 GPU. All three pipelines are scored on the same
200 questions.

### Overall

| Approach                                | TextVQA accuracy | ANLS  |
| --------------------------------------- | ---------------- | ----- |
| OCR-first (EasyOCR → Flan-T5-XL)        | 24.90%           | 0.373 |
| VLM (BLIP-2, zero-shot, no OCR)         | 20.65%           | 0.302 |
| **Hybrid (BLIP-2 + OCR tokens as hint)** | **28.55%**       | **0.419** |

### By OCR coverage of the ground-truth answer

| OCR coverage of GT answer          | Questions | OCR-first | Hybrid |
| ---------------------------------- | --------- | --------- | ------ |
| full (every answer token present)  | 65        | 58.8%     | 54.5%  |
| partial (some answer tokens)       | 12        | 2.5%      | 5.0%   |
| none (OCR missed the answer)       | 123       | 9.2%      | 17.2%  |

The VLM pipeline never sees OCR output, so `02_eval.py` does not assign its
predictions to coverage buckets; only its overall score (20.65%) is reported.

### What the numbers show

- **Hybrid is the best overall**: +3.65 points over OCR-first and +7.9 points over the
  VLM on its own.
- **OCR quality is the bottleneck.** EasyOCR captured the complete answer for only 65
  of 200 questions (32.5%) and none of it for 123 (61.5%).
- **When OCR has the answer, text alone is enough.** OCR-first reaches 58.8% in the
  `full` bucket; adding the image does not improve on that (54.5%).
- **When OCR misses, the image is the fallback.** OCR-first drops to 9.2% in the `none`
  bucket, while the hybrid nearly doubles that (17.2%) by reading from the image.
- The `partial` bucket has only 12 questions, so its numbers are too noisy to interpret.

### Failure cases

| Question                       | OCR-first       | VLM              | Ground truth     | What happened                    |
| ------------------------------ | --------------- | ---------------- | ---------------- | -------------------------------- |
| what word is handwritten?      | Urban ✗         | jesus ✓          | jesus            | OCR missed the handwriting       |
| what is the name of the vodka? | GRFAT CHASE ✗   | chase ✓          | chase            | OCR misread a character          |
| who was the photographer?      | Philippe Molitor ✓ | stefan savchenko ✗ | philippe molitor | VLM hallucinated a name          |
| are these switches on or off?  | off ✓           | on ✗             | off              | VLM guessed instead of reading   |

These results come from a 200-question subset, so treat them as indicative rather than
as benchmark numbers for the full validation split.

## Repo structure

```
Reading-Text-in-Images-for-Visual-Question-Answering/
├── README.md
└── Reading Text in Images for Visual Question Answering/
    ├── Reading-Text-in-Images-for-Visual-Question-Answering.ipynb   # end-to-end Colab run, with outputs
    ├── requirements.txt
    ├── configs/
    │   └── default.yaml
    ├── src/
    │   ├── config.py               # config loader + device selection
    │   ├── dataset.py              # TextVQA loader
    │   ├── ocr.py                  # EasyOCR / Tesseract wrappers
    │   ├── normalize.py            # official answer normalization
    │   ├── metrics.py              # TextVQA soft accuracy + ANLS
    │   ├── ocr_quality.py          # OCR-vs-GT-answer coverage score for analysis
    │   └── models/
    │       ├── ocr_first.py        # OCR -> text reasoner
    │       ├── vlm.py              # frozen BLIP-2 VLM
    │       └── hybrid.py           # VLM + OCR hint
    └── scripts/
        ├── 00_download.md          # how to get the data
        ├── 01_run_ocr.py           # cache OCR for all images
        ├── 02_eval.py              # run one pipeline + score it
        └── 03_analysis.py          # buckets, plots, comparison table
```

`data/` (TextVQA json + images) and `outputs/` (predictions, scores, figures) are
created when you run the pipeline and are not checked in.

## Quick start — Colab (easiest)

Open `Reading-Text-in-Images-for-Visual-Question-Answering.ipynb` in Colab with a GPU
runtime and run top to bottom. It installs dependencies, downloads TextVQA, runs all
three pipelines, and produces the comparison table + plots.

## Quick start — local / CLI

```bash
git clone https://github.com/saiprasad-29/Reading-Text-in-Images-for-Visual-Question-Answering.git
cd "Reading-Text-in-Images-for-Visual-Question-Answering/Reading Text in Images for Visual Question Answering"
pip install -r requirements.txt
pip install easyocr pytesseract opencv-python

# 1. download data (see scripts/00_download.md), put under data/
# 2. cache OCR once (slow):
python scripts/01_run_ocr.py --split val --engine easyocr --limit 200
# 3. evaluate each approach:
python scripts/02_eval.py --approach ocr_first --split val --limit 200
python scripts/02_eval.py --approach vlm       --split val --limit 200
python scripts/02_eval.py --approach hybrid    --split val --limit 200
# 4. compare + analyze:
python scripts/03_analysis.py --split val
```

Set `LIMIT = 0` (in the notebook) or drop `--limit` (CLI) to run the full validation
split instead of the 200-question subset.

## Why three pipelines

The interesting result isn't a single accuracy number — it's the **interaction
between OCR quality and answer source**. Splitting every question by how much of the
correct answer the OCR actually captured shows that the two sources of evidence fail
in different places: OCR-first is strong when the text was read correctly and nearly
useless when it wasn't, while the image-based model is weaker overall but does not
depend on OCR. The hybrid gets most of the benefit of both.

## Notes on the environment

Downloading TextVQA (~25 GB) and running OCR/VLM inference needs a machine with
internet and ideally a GPU. The code runs on CPU for small `--limit` smoke tests and
scales up on GPU for the full split. In Colab, set an optional `HF_TOKEN` to avoid
Hugging Face download rate limits.
