# Seeding Calculator

A Streamlit app that calculates seeding densities for adherent cell lines based on doubling time, vessel type, and time to confluence. Uses a logistic growth model with attachment lag and plating efficiency corrections.

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run Script.py
```

The app opens at `http://localhost:8501`.

## Features

- Preset doubling times for common adherent lines (HeLa, HEK293, MCF7, A549, and more), with a custom entry option
- Supports 11 vessel formats from 384-well to 15 cm dish
- Standard timepoints (24 h – 144 h) plus a custom timepoint field
- Advanced options: attachment lag and plating efficiency
- Interactive logistic growth curve (Plotly)
- Practical seeding floor to guard against Poisson noise at very low densities

## Dependencies

| Package | Version |
|---|---|
| streamlit | 1.57.0 |
| plotly | 6.7.0 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
