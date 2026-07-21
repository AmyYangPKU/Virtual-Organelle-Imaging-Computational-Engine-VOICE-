# VOICE_app

VOICE_app is a napari plugin for synthetic organelle LR/SR dataset generation.

This repository is a public installable plugin package. The UI/plugin shell is
included as Python source, while these core algorithm modules are distributed as
compiled CPython bytecode only:

- `generate_mt.pyc`
- `generate_mito.pyc`
- `generate_ccp.pyc`
- `generate_npc.pyc`
- `generate_er.pyc`
- `resolution_tools.pyc`

No `.py` source files for those six modules are included.

The OMM profile data is embedded inside the compiled `generate_mito.pyc`
module, so no separate `.mat` data files are exposed in the repository.
The compiled `resolution_tools.pyc` module includes the OMM 3D PSF helper
ported from `AutoPSF3D`.
OMM image generation uses a 3D volume blur for GT/SR, fits `sigma_xy` in the
`[0.5, 2.0]` range with `sigma_z = 1.2 + rand * 0.8`, then creates LR by
downsampling GT/SR before applying the fitted 2D `generate_psf` lambda.

## Compatibility

This package currently targets CPython 3.10 because the bundled `.pyc` files were
compiled with Python 3.10. Use a Python 3.10 napari environment.

## Install From GitHub

```bash
python -m pip install git+https://github.com/YOUR_GITHUB_NAME/VOICE_app.git
```

Then start napari:

```bash
napari
```

Open the plugin from:

```text
Plugins > VOICE_app > VOICE_app
```

## Local Install

```bash
python -m pip install .
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
