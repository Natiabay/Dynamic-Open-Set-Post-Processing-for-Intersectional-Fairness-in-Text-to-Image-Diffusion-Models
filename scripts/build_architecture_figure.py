#!/usr/bin/env python3
"""Compile TikZ architecture diagram to PDF/PNG for the paper."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fig_dir = ROOT / "craft-gc-paper" / "figures"
tex = fig_dir / "architecture.tex"

try:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(fig_dir), str(tex.name)],
        check=True,
        cwd=fig_dir,
        capture_output=True,
        text=True,
    )
    print(f"Built {fig_dir / 'architecture.pdf'}")
except FileNotFoundError:
    print("pdflatex not installed; compile architecture.tex on Overleaf or after apt install texlive-latex-extra.")
except subprocess.CalledProcessError as e:
    print((e.stdout or "")[-800:])
