# HSF-India 2026 hackathon: Differentiable Programming for HEP

This repository contains the hackathon notebook `zprime_differentiable_hackathon.ipynb`, in which you tune the selection of a Z' → ttbar search by gradient descent instead of scanning cut values by hand. The event samples it uses, skimmed from CMS Open Data, are in `data/`.

## Option 1: MyBinder (no installation)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MoAly98/hsf-india-mandi-diffprog-hackathon/main?urlpath=lab/tree/zprime_differentiable_hackathon.ipynb)

Click on the Binder badge above or go to the url below to launch a JupyterLab session in your browser with the notebook and all dependencies pre-installed using MyBinder:

```
https://mybinder.org/v2/gh/MoAly98/hsf-india-mandi-diffprog-hackathon/main?urlpath=lab/tree/zprime_differentiable_hackathon.ipynb
```

## Option 2: run locally with `pixi`

First, clone this repository.

```shell
git clone https://github.com/MoAly98/hsf-india-mandi-diffprog-hackathon.git
cd hsf-india-mandi-diffprog-hackathon
```

Make sure you've [installed pixi](https://pixi.sh/latest/installation/) on your computer.

Then you can install the environment and start a local JupyterLab session with:

```shell
pixi run start
```

## Option 3: run locally with `uv` or a standard Python `venv`

Make sure you've [installed uv](https://docs.astral.sh/uv/getting-started/installation/) on your computer.

Then you can install the environment and start a local JupyterLab session with uv:

```shell
uv venv --python=3.14
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab
```

or with a standard Python `venv`:

```shell
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

JAX does not provide wheels for Intel Macs, so it is skipped there.

## Option 4: run locally with `conda`/`mamba`

Make sure you've [installed Miniforge](https://conda-forge.org/download/) on your computer.

Then you can install the environment and start a local JupyterLab session with:

```shell
conda env create -f environment.yml
conda activate hsf-india-diffprog-hackathon
jupyter lab
```

JAX is installed with `pip` inside the environment, because conda-forge has no Windows build of it, and it is skipped on Intel Macs, which it does not provide wheels for.
