# TNAM Minimal Reproduction Package
This package only keeps the required code to reproduce **TNAM** training/evaluation.
## Included
- `main.py`: TNAM-only training entry.
- `MODEL/TNAM.py`: TNAM implementation.
- `utils.py`: utility helpers. Currently it provides `set_seed` for reproducible training across Python, NumPy, and PyTorch.
- `evaluate_sepsis_score.py`: metric calculation.
- `config/*.yaml`: TNAM configs for datasets.

## Run
```bash
pip install -r requirements.txt
python main.py
```
