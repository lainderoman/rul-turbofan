# CLAUDE.md

## Project context

- University group project for **Emerging Technologies in Aviation Engineering** (TSI, prof. Dmitry Pavlyuk).
- **Deadline: 17 Dec 2026.**

## Goal

Predict the **Remaining Useful Life (RUL)** of turbofan engines using the NASA **C-MAPSS** dataset (subsets FD001–FD004). The raw `.txt` files live in `data/raw/`.

## Baseline vs. our contribution

- **Baseline:** the course notebook — classic ML models trained on FD001, best RMSE ≈ 26.
- **Our contribution:**
  - RUL capping (piecewise-linear target, cap ≈ 125 cycles)
  - Time windows over the sensor sequences
  - Sequence models: LSTM and the course's TransLSTM
  - Validation split **by engine** (no engine appears in both train and validation)
  - Evaluation with RMSE + NASA scoring function
  - SHAP explainability
  - A maintenance decision rule built on the RUL predictions

## Repository structure

- `notebooks/` — numbered notebooks: `01_eda`, `02_baselines`, ...
- `src/` — reusable functions imported by the notebooks
- `results/` — figures, metrics, saved outputs
- `docs/meetings/` — meeting notes
- `data/raw/` — original C-MAPSS `.txt` files

## Rules

- All code, comments and docs in **English**.
- **Python 3.12**, virtual environment in `.venv`.
- Keep code **simple and well commented**: we are students and must be able to explain every line in an oral Q&A. Prefer clear, explicit code over clever one-liners.
