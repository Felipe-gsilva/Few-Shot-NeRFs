# nerfstudio-gnt

Unofficial GNT (Generalizable NeRF Transformer) integration for Nerfstudio.

## Install

```bash
uv pip install -e .
```

## Verify method registration

```bash
ns-train --help | grep -i gnt
```

## Dataset format

This integration uses Nerfstudio's native `VanillaDataManager` with `NerfstudioDataParserConfig`.
Point the dataparser to a scene root containing `transforms.json`.

## Configure GNT

Use the `gnt` method and set datamanager options from CLI:

```bash
ns-train gnt --pipeline.datamanager.data-root /path/to/datasets
```

```bash
ns-train gnt \
  --output-dir /path/to/outputs \
  --vis tensorboard \
  --pipeline.datamanager.dataparser.data /path/to/scene_root
```

## Smoke check

```bash
python test.py /path/to/scene_root/transforms.json
```

This runs one train step through `GNTPipeline.get_train_loss_dict` and checks source-view tensor shapes.
