# VOICE_app

VOICE_app is a napari plugin for synthetic organelle LR/SR dataset generation.

This repository is a public installable plugin package. VOICE_app can generate
virtual datasets for five types of organelles: microtubules, outer mitochondrial
membranes (OMM), clathrin-coated pits (CCPs), nuclear pore complexes (NPCs), and
the endoplasmic reticulum (ER).

## Compatibility

This package currently targets CPython 3.10 because the bundled `.pyc` files were
compiled with Python 3.10. Use a Python 3.10 napari environment.

## Install From GitHub

Use this method if you want to install the latest uploaded version directly from
GitHub.

```bash
python -m pip install git+https://github.com/AmyYangPKU/Virtual-Organelle-Imaging-Computational-Engine-VOICE-.git
```

Then start napari:

```bash
napari
```

Open the plugin from:

```text
Plugins > VOICE_app > VOICE_app
```

## Install From Downloaded Code (recommended)

Use this method if you downloaded the repository as a ZIP file or cloned it to
your computer.

First create and activate a Python 3.10 conda environment:

```bash
conda create -n voice_app python=3.10 -y
conda activate voice_app


Then enter the downloaded project folder:


```bash
cd /path/to/VOICE_app
```

Install the plugin:

```bash
python -m pip install .
```

Then start napari:

```bash
napari
```

Open the plugin from:

```text
Plugins > VOICE_app > VOICE_app
```

## Editable Local Install

Use this only if you are editing the visible plugin wrapper files and want local
changes to take effect without reinstalling every time.

```bash
cd /path/to/VOICE_app
python -m pip install -e .
napari
```

## Build Wheel

```bash
python -m build --wheel --no-isolation
```

The wheel will be created in `dist/`.

## Source Visibility Note

Shipping `.pyc` hides the plain `.py` files from this repository, but bytecode is
not strong IP protection. For strict protection, use a private repository or move
the core algorithm to a server-side API.
