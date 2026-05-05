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

## Dataset root (`data_root`)

Upstream GNT loaders expect paths like:

- `.../data/real_iconic_noface` (llff)
- `.../data/nerf_synthetic`
- `.../data/google_scanned_objects`

In this integration, `data_root` should point to either:

1. The parent directory of `data/` (recommended), or
2. The `data/` directory itself.

The datamanager normalizes `data_root` to the GNT `rootdir` convention and validates expected dataset folders before training.

## Configure GNT

Use the `gnt` method and set datamanager options from CLI:

```bash
ns-train gnt --pipeline-config.datamanager-config.data-root /path/to/datasets
```

For generic Nerfstudio scenes (`images/` + `transforms.json`), use:

```bash
ns-train gnt \
  --pipeline-config.datamanager-config.train-dataset nerfstudio \
  --pipeline-config.datamanager-config.eval-dataset nerfstudio \
  --pipeline-config.datamanager-config.data-root /path/to/scene_root
```

## Smoke check

```bash
python test.py
```

This validates that the exported method specification can be imported and resolved.
