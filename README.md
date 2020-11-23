# Robust detection and attribution

RobustDA is a Python package for testing detection and attribution.

## Installation

Install the virtual environment with conda and activate it:

```bash
conda env create -f environment.yml
conda activate robustDA
```

Install robustDA in the virtual environment:

```bash
pip install -e .
```

Run using python3 with the parameters: target, anchor, gamma, nonlinear_anchors

```bash
python3 main.py --target aerosols --anchor co2 --gamma 1 1000 --nonlinear_anchors square abs
```

Run the jupyter notebook:

```bash
jupyter notebook
```

To deactivate the environment use:
```bash
conda deactivate
```


## Structure

## References

## Data