# CRAFT-GC Paper

Compile with `pdflatex` + `bibtex` (or `latexmk`):

```bash
cd craft-gc-paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Figures are generated from project root:

```bash
python3 scripts/evaluate_hash_embedding.py   # Table 1 numbers
python3 scripts/run_ablation.py              # Ablation table
python3 scripts/plot_figures.py              # figures/cofs_comparison.pdf
```

## Submission checklist

- [x] GCFairBench (500 prompts) in `craft_gc/benchmark/prompts/`
- [x] Stage A embedding pilot results in Section 5
- [x] Ablation table and CoFS figure
- [ ] Stage B SD image results (run on GPU: `scripts/run_pilot_images.py`)
- [ ] CLIP embedding eval (run with torch: `scripts/evaluate_embeddings.py`)
- [ ] Replace architecture placeholder figure (Figure 1)
- [ ] Human review / advisor sign-off before journal submission

**Honesty note:** Stage A uses a hash-embedding proxy, not FairFace on generated images. Do not describe Table 1 as Stable Diffusion image results.
