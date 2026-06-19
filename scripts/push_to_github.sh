#!/bin/bash
# Push Research repo to GitHub. Run once after authenticating.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Git status ==="
git status -sb
git log origin/main..HEAD --oneline 2>/dev/null || git log -1 --oneline

echo ""
echo "=== Pushing to origin main ==="

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh auth setup-git
  git push -u origin main
elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
  git push "https://Natiabay:${GITHUB_TOKEN}@github.com/Natiabay/Dynamic-Open-Set-Post-Processing-for-Intersectional-Fairness-in-Text-to-Image-Diffusion-Models.git" main
else
  echo "No GitHub auth found. Choose one:"
  echo "  1) gh auth login   (then re-run this script)"
  echo "  2) export GITHUB_TOKEN=ghp_... && bash scripts/push_to_github.sh"
  exit 1
fi

echo "Done. Verify: https://github.com/Natiabay/Dynamic-Open-Set-Post-Processing-for-Intersectional-Fairness-in-Text-to-Image-Diffusion-Models"
