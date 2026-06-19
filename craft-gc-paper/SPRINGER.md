# Springer Journal Submission (sn-jnl)

CRAFT-GC targets Springer Nature journals (e.g., *AI and Ethics*) using the **sn-jnl** LaTeX class.

## Overleaf (recommended)

1. Open [Springer Nature LaTeX template](https://www.overleaf.com/latex/templates/springer-nature-latex-template/myxmhdsbznyt)
2. Upload `main.tex`, `references.bib`, and `figures/` from this folder
3. Replace the template body with our `main.tex` content (or use `main-springer.tex` below)
4. Set `\documentclass[pdflatex,sn-mathphys-num,Numbered]{sn-jnl}`

## Local compile

Download `sn-jnl.cls` and bst files from Springer Author Guidelines, place in `craft-gc-paper/`, then:

```bash
pdflatex main-springer && bibtex main-springer && pdflatex main-springer && pdflatex main-springer
```

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Full manuscript (generic article class) |
| `main-springer.tex` | Springer wrapper (requires sn-jnl.cls) |
| `figures/` | Generated plots (run `scripts/plot_figures.py`) |
