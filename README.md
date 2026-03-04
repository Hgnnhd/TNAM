# TNAM Minimal Reproduction Package

This package only keeps the required code to reproduce **TNAM** training/evaluation.
It excludes visualization scripts, comparison-model code, rebuttal scripts, logs, and experiment artifacts.

## Included
- `main.py`: TNAM-only training entry.
- `MODEL/TNAM.py`: TNAM implementation.
- `utils.py`: seed setup and class-balancing sampler.
- `evaluate_sepsis_score.py`: metric calculation.
- `config/*.yaml`: TNAM configs for datasets.

## Run
```bash
pip install -r requirements.txt
python main.py
```
