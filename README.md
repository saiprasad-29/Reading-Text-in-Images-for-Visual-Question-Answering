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

## Repo structure

```
MLProject/
├── README.md
├── requirements.txt
├── MLProject_final.ipynb      # run everything end-to-end in Colab
├── configs/
│   └── default.yaml
├── data/                      # TextVQA json + images go here (gitignored)
├── src/
│   ├── dataset.py              # TextVQA loader
│   ├── ocr.py                  # EasyOCR / Tesseract wrappers
│   ├── normalize.py            # official answer normalization
│   ├── metrics.py               # TextVQA soft accuracy + ANLS
│   ├── ocr_quality.py          # OCR-vs-GT-answer coverage score for analysis
│   └── models/
│       ├── ocr_first.py        # OCR -> text reasoner
│       ├── vlm.py               # frozen BLIP-2 VLM
│       └── hybrid.py            # VLM + OCR hint
├── scripts/
│   ├── 00_download.md          # how to get the data (network-gated)
│   ├── 01_run_ocr.py           # cache OCR for all images
│   ├── 02_eval.py              # run one pipeline + score it
│   └── 03_analysis.py          # buckets, plots, comparison table
└── outputs/                    # predictions, scores, figures (gitignored)
```

## Quick start — Colab (easiest)

Open `MLProject_final.ipynb` in Colab and run top to bottom. It clones this repo,
installs dependencies, downloads TextVQA, runs all three pipelines, and produces the
comparison table + plots.

## Quick start — local / CLI

```bash
git clone https://github.com/Prajwala15/MLProject.git
cd MLProject
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
split instead of a smoke test.

## Why three pipelines

The interesting result isn't a single accuracy number — it's the **interaction
between OCR quality and answer source**. `03_analysis.py` produces a table like:

| OCR coverage of GT answer | OCR-first acc | VLM acc | Hybrid acc |
|---|---|---|---|
| full (answer token present) | high | mid | high |
| partial | mid | mid | high |
| none (OCR missed it) | low | depends on VLM's own reading | mid |

## Notes on the environment

Downloading TextVQA (~25 GB) and running OCR/VLM inference needs a machine with
internet and ideally a GPU. The code runs on CPU for small `--limit` smoke tests and
scales up on GPU for the full split. In Colab, set an optional `HF_TOKEN` to avoid
Hugging Face download rate limits.
