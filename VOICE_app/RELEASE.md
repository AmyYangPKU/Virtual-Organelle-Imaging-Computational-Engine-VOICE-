# Release Notes

## v0.1.0

- Adds the `VOICE_app` napari widget.
- Keeps the plugin UI/source wrapper visible.
- Ships the six core algorithm modules as `.pyc` only:
  `generate_mt`, `generate_mito`, `generate_ccp`, `generate_npc`,
  `generate_er`, and `resolution_tools`.
- Embeds OMM profile data inside the compiled `generate_mito.pyc` module.
- Adds the OMM 3D PSF helper ported from `AutoPSF3D` inside
  `resolution_tools.pyc`.
- Updates OMM convolution so GT/SR is generated from 3D volume blur with fitted
  `sigma_xy`, while LR is generated from downsampled GT/SR using the fitted 2D
  `generate_psf` lambda.

Install:

```bash
python -m pip install git+https://github.com/YOUR_GITHUB_NAME/VOICE_app.git
```
