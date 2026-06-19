# CRAFT-GC Research Repository

**Authors:** Natnael Abayneh, Abiy Alemu — MSc AI, Addis Ababa University

Training-free geo-cultural fairness steering for Stable Diffusion v1.5 (CRAFT-GC).

**Repository:** https://github.com/Natiabay/Dynamic-Open-Set-Post-Processing-for-Intersectional-Fairness-in-Text-to-Image-Diffusion-Models

---

## Project structure

```
craft_gc/           CAD, BDE, CAS (768-d SD steering), pipeline, metrics
craft_gc/benchmark/ GCFairBench (500 prompts, 5 regions)
scripts/            Evaluation and Colab runners
craft-gc-paper/     LaTeX manuscript + figures
app.py              Gradio demo for Hugging Face Spaces
CRAFT_GC_Colab.ipynb  Full publishable experiment (Trial 2)
results/            Generated outputs (after Colab)
```

---

## Publishable experiment (Colab Trial 2)

### 1. Open `CRAFT_GC_Colab.ipynb` on Colab (T4 GPU)

### 2. Run cells in order

| Cell | Output |
|------|--------|
| Setup | Clone repo, install craft_gc |
| Steering check | Confirms Base ≠ CRAFT-GC images |
| Stage A | `results/embedding_eval.csv` |
| Stage B | 100 prompts (20/region), 2000 images |
| Download | `craft_gc_results.zip` |

### 3. On your PC after download

```bash
cd ~/Desktop/Research
unzip ~/Downloads/craft_gc_results.zip -d results/
python scripts/summarize_results.py
python scripts/update_image_table.py
```

---

## Trial 1 vs Trial 2

| | Trial 1 (pilot) | Trial 2 (publishable) |
|--|-----------------|----------------------|
| Prompts | 50 (SSA only) | 100 stratified (20 × 5 regions) |
| CAS | OpenCLIP 512-d (no effect) | SD text encoder 768-d |
| Status | Appendix / limitation | Main paper results |

---

## Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install open-clip-torch pandas matplotlib diffusers transformers accelerate
```

---

## Hugging Face Space demo

1. Create Space at huggingface.co/new-space (Gradio, **GPU**)
2. Point to this repo; set `app.py` and `requirements-space.txt`
3. Add secret `HF_TOKEN`
4. Upload `results/pilot_images/` from Colab for benchmark gallery tab

See `README_SPACE.md`.

---

## GCFairBench prompt criteria

Each prompt combines:
- **Region locative** (e.g. in Addis Ababa, in Mumbai)
- **Category template** (profession, social role, daily activity, cultural practice, public setting)
- **Cultural caption** for CFS evaluation

500 prompts = 5 regions × 100 prompts. Stage B uses stratified sampling (`--per-region 20`).

---

## Citation

```bibtex
@article{abayneh2026craftgc,
  title={CRAFT-GC: Cross-Attention Fairness Steering with Geo-Cultural Knowledge Integration for Training-Free Bias Mitigation in Text-to-Image Diffusion Models},
  author={Abayneh, Natnael and Alemu, Abiy},
  year={2026},
  institution={Addis Ababa University}
}
```
