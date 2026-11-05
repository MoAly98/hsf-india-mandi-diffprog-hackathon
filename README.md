# HSF-India 2026 hackathon: Differentiable Programming for HEP

This repository contains the hackathon notebooks, in which you tune the selection of a Z' → ttbar search by gradient descent instead of scanning cut values by hand, on samples skimmed from CMS Open Data.

## The notebooks

Work through them in order:

1. **`zprime_differentiable_hackathon.ipynb`**: the core idea on easy data. The input is a small pre-made flat table (`data/simple/`, one npz file per sample) with three variables per event. You learn the whole differentiable-analysis trick in its simplest form: why hard cuts have no gradient, soft cuts and soft histograms (bKDE), gradient ascent, optimising several cuts at once, and the honest train/test evaluation. The statistics is a simple s/√b, and the signal is scaled up so every effect is clearly visible.

2. **`zprime_nanoaod_hackathon.ipynb`**: the same search, done like a real analysis. The input is 18 million skimmed CMS Open Data NanoAOD events (`data/nanoaod/`, jagged per-object arrays): you build the flat table yourself with `uproot` and `awkward` (object masks, structural cuts, four-vectors, padding). New physics content: the full set of hard→soft transformations, including cuts that *count* objects above a tunable threshold, a neural network pretrained as a classifier and then fine-tuned inside the analysis against the physics objective, and one systematic uncertainty (jet energy scale). The signal carries its real cross-section, so expect honest, small significances.

## The data

The part-1 npz tables (`data/simple/`) are small and ship with this repository, so there is nothing to do.

The part-2 NanoAOD skims (`data/nanoaod/`, a few GB) are distributed separately:

- **During the hackathon (2–6 November)**, for people using BinderHub as in Option 0 below, they are already available under `data/mandi26/diffprog-hack`.
- **After the school**, download them from CERNBox:

```
https://cernbox.cern.ch/s/g1NTIeLDrMUxNPX
```

Place them so that `data/nanoaod/` sits at the top of the repository, next to `data/simple/`.

## Option 0: BinderHub (no installation)

Go to the url below to launch a JupyterLab session in your browser, with no GPU, 4 CPUs and 4 GB of memory, using BinderHub:

```
https://binderhub.ssl-hep.org/v2/gh/research-software-collaborations/courses-hsf-india-October2026/binder?gpuModel=&gpuCount=0&cudaMajor=undefined&cudaMinor=undefined&site=nrp&memory=4.0&cpu=4&qos=Guaranteed
```

<p align="center"><img src="https://github.com/jpivarski-talks/2024-05-22-hsf-india-tutorial-delhi/assets/1852447/9f57768b-7d3d-4b74-8364-c066fe352c39" width="75%"></p>

This session comes with all dependencies pre-installed, but not with the notebook.
To get it, bring up a terminal by pressing the "Terminal" button:

<p align="center"><img src="https://github.com/jpivarski-talks/2024-05-22-hsf-india-tutorial-delhi/assets/1852447/9efd21b4-1cd7-45b7-80d0-a20bb0c36ac4" width="75%"></p>

Then clone this repository in that terminal with:

```shell
git clone https://github.com/MoAly98/hsf-india-mandi-diffprog-hackathon.git
```

<p align="center"><img src="https://github.com/jpivarski-talks/2024-05-22-hsf-india-tutorial-delhi/assets/1852447/89b5625b-c1db-44f7-8ce7-774ed592580e" width="75%"></p>

Now there is a `hsf-india-mandi-diffprog-hackathon` directory in the file browser that you can double-click to open the notebook.

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
