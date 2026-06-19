# CRAFT-GC Research Repository

**Repository:** https://github.com/Natiabay/Dynamic-Open-Set-Post-Processing-for-Intersectional-Fairness-in-Text-to-Image-Diffusion-Models

**Authors:** Natnael Abayneh, Abiy Alemu — MSc AI, Addis Ababa University

Cross-Attention Fairness Steering with Geo-Cultural Knowledge Integration (CRAFT-GC) for training-free bias mitigation in Stable Diffusion.

## Status

| Stage | Description | Status |
|-------|-------------|--------|
| **A** | CLIP ViT-B/32 embedding eval (500 prompts) | Complete |
| **B** | SD v1.5 image generation + CoFS/CFS/FID | Run on [Colab notebook](notebooks/CRAFT_GC_Colab.ipynb) |
| **Paper** | LaTeX + figures + Springer template | Draft ready |

## Quick start (local CPU — Stage A)

```bash
cd ~/Desktop/Research
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch pandas matplotlib pyyaml tqdm

python -m craft_gc.benchmark.gcfairbench
python scripts/evaluate_embeddings.py --device cpu
python scripts/merge_paper_results.py
python scripts/plot_figures.py
```

## GPU experiments (Colab — Stage B)

1. Open `notebooks/CRAFT_GC_Colab.ipynb` on Colab (GPU runtime)
2. Add `HF_TOKEN` in Colab Secrets
3. Run all cells — Stage B takes **2–4 hours** for 50 prompts
4. Download `results/` and run `python scripts/update_image_table.py` locally

```bash
# After Colab:
python scripts/update_image_table.py
```

## Push to GitHub

```bash
bash scripts/push_to_github.sh
# or: git push origin main
```

## Structure

```
craft_gc/           # CAD, BDE, CAS, pipeline, metrics, GCFairBench
scripts/            # Evaluation, plotting, colab_run_stage_b.py
notebooks/          # CRAFT_GC_Colab.ipynb
craft-gc-paper/     # main.tex, figures/, references.bib
results/            # Generated outputs
```

## Citation

```bibtex
@article{abayneh2026craftgc,
  title={CRAFT-GC: Cross-Attention Fairness Steering with Geo-Cultural Knowledge Integration for Training-Free Bias Mitigation in Text-to-Image Diffusion Models},
  author={Abayneh, Natnael and Alemu, Abiy},
  year={2026},
  institution={Addis Ababa University}
}
```
