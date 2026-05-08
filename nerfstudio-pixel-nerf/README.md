# nerfstudio-pixel-nerf

Unofficial pixelNeRF integration for Nerfstudio. A framework for training and rendering few-shot Neural Radiance Fields using image-conditioned feature extraction.

## Install

```bash
uv pip install -e .
```

## Verify method registration

```bash
ns-train --help | grep -i pixel-nerf
```

## Dataset Format
This integration uses a custom pipeline to inject source views dynamically into the RayBundle metadata. It is compatible with the standard `NerfstudioDataParserConfig`. Point the dataparser to a scene root containing a valid `transforms.json`.

## Configure PixelNeRF
Use the `pixel-nerf` method and set your data paths from the CLI:

```bash
ns-train pixel-nerf --data /path/to/scene_root
```

Or with extended configuration and TensorBoard visualization:

```
ns-train pixel-nerf \
  --output-dir /path/to/outputs \
  --vis tensorboard \
  --pipeline.datamanager.dataparser.data /path/to/scene_root
```
