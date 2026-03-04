# TNAM Minimal Reproduction Package

This package only keeps the required code to reproduce **TNAM** training/evaluation.
It excludes visualization scripts, comparison-model code, rebuttal scripts, logs, and experiment artifacts.

## Included
- `main.py`: TNAM-only training entry.
- `MODEL/TNAM.py`: TNAM implementation.
- `utils.py`: seed setup and class-balancing sampler.
- `evaluate_sepsis_score.py`: metric calculation.
- `config/*.yaml`: TNAM configs for datasets (`A`, `B`, `CD`, `TJ`, `2012`, `M`).

## Expected data layout
Put data under:
- `data/trainA/data.pickle`, `data/trainA/label.pickle`
- `data/trainB/data.pickle`, `data/trainB/label.pickle`
- `data/trainCD/data.pickle`, `data/trainCD/label.pickle`
- `data/trainTJ/data.pickle`, `data/trainTJ/label.pickle`
- `data/train2012/data.pickle`, `data/train2012/label.pickle`
- `data/trainM/data.pickle`, `data/trainM/label.pickle`

## Run
```bash
pip install -r requirements.txt
python main.py --config config/B.yaml
```
