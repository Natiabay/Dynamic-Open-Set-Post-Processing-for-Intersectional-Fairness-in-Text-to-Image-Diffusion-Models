# CRAFT-GC Research Repository

**Repository:** https://github.com/Natiabay/Dynamic-Open-Set-Post-Processing-for-Intersectional-Fairness-in-Text-to-Image-Diffusion-Models

**Authors:** Natnael Abayneh, Abiy Alemu — MSc AI, Addis Ababa University

Cross-Attention Fairness Steering with Geo-Cultural Knowledge Integration (CRAFT-GC) for training-free bias mitigation in Stable Diffusion.

## Status

| Stage | Description | Status |
|-------|-------------|--------|
| **A** | CLIP ViT-B/32 embedding eval (500 prompts) | ✅ Complete (CPU) |
| **B** | SD v1.5 image generation + CoFS/CFS/FID | ⏳ Run on [Colab notebook](notebooks/CRAFT_GC_Colab.ipynb) |
| **Paper** | LaTeX + figures + Springer template | ✅ Draft ready |

## Quick start (local CPU — Stage A)

```bash
cd ~/Desktop/Research
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch pandas matplotlib pyyaml tqdm

python -m craft_gc.benchmark.gcfairbench          # 500 prompts
python scripts/evaluate_embeddings.py --device cpu # ~14 min
python scripts/merge_paper_results.py
python scripts/plot_figures.py
```

## GPU experiments (Colab — Stage B)

1. Zip this folder and upload to Google Colab
2. Open `notebooks/CRAFT_GC_Colab.ipynb`
3. Set `HF_TOKEN` in Colab Secrets (copy from `.env.example` — **never commit `.env`**)
4. Run all cells (~2–4 h for 50-prompt pilot)
5. Download `results/` and run `python scripts/update_image_table.py` to fill LaTeX Table 2

```bash
# After Colab, locally:
python scripts/update_image_table.py
cd craft-gc-paper && pdflatex main && bibtex main && pdflatex main
```

## Compile paper

```bash
cd craft-gc-paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For Springer journals, see [craft-gc-paper/SPRINGER.md](craft-gc-paper/SPRINGER.md).

## Structure

```
craft_gc/           # CAD, BDE, CAS, pipeline, metrics, GCFairBench
scripts/            # Evaluation, plotting, Colab helpers
notebooks/          # CRAFT_GC_Colab.ipynb
craft-gc-paper/     # main.tex, figures/, references.bib
results/            # embedding_eval.csv, paper_results.json (generated)
```

## Real results (Stage A, n=500)

| Method | CoFS | CFS |
|--------|------|-----|
| Base | 0.9971 | 0.0000 |
| PromptAug | 0.9974 | 0.0000 |
| FairImagen-E | 0.9971 | -0.0015 |
| CRAFT-GC-E | 0.9971 | -0.0021 |

Image-level Stage B numbers go in Table 2 after Colab run.

## Citation

```bibtex
@article{abayneh2026craftgc,
  title={CRAFT-GC: Cross-Attention Fairness Steering with Geo-Cultural Knowledge Integration for Training-Free Bias Mitigation in Text-to-Image Diffusion Models},
  author={Abayneh, Natnael and Alemu, Abiy},
  year={2026},
  institution={Addis Ababa University}
}
```
