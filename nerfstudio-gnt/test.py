import argparse
from pathlib import Path

import torch
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

from GNTModel import GNTModelConfig
from GNTPipeline import GNTPipelineConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for GNT + VanillaDataManager integration."
    )
    parser.add_argument(
        "transforms_json",
        type=Path,
        help="Path to transforms.json",
    )
    parser.add_argument("--num-rays", type=int, default=128)
    parser.add_argument("--num-source-views", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    transforms_json = args.transforms_json.expanduser().resolve()
    if transforms_json.name != "transforms.json":
        raise ValueError("Expected a path to transforms.json.")
    if not transforms_json.exists():
        raise FileNotFoundError(f"Could not find transforms.json at '{transforms_json}'.")

    scene_root = transforms_json.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline_config = GNTPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(data=scene_root),
            train_num_rays_per_batch=args.num_rays,
            eval_num_rays_per_batch=args.num_rays,
        ),
        model=GNTModelConfig(
            num_source_views=args.num_source_views,
            N_importance=0,
            single_net=True,
        ),
    )
    pipeline = pipeline_config.setup(device=device, test_mode="val", world_size=1, local_rank=0)

    ray_bundle, batch = pipeline.datamanager.next_train(step=0)
    target_idx = pipeline._extract_target_image_idx(batch)
    src_rgbs, src_cameras = pipeline._sample_source_views(target_idx, split="train")
    print("src_rgbs shape:", tuple(src_rgbs.shape))
    print("src_cameras shape:", tuple(src_cameras.shape))
    assert src_rgbs.ndim == 5 and src_rgbs.shape[0] == 1
    assert src_cameras.ndim == 3 and src_cameras.shape[-1] == 34
    assert src_rgbs.dtype == torch.float32
    assert float(src_rgbs.min().item()) >= 0.0 and float(src_rgbs.max().item()) <= 1.0

    outputs, loss_dict, metrics_dict = pipeline.get_train_loss_dict(step=0)
    coarse_rgb = outputs["outputs_coarse"]["rgb"]
    print("coarse rgb shape:", tuple(coarse_rgb.shape))
    print("loss keys:", sorted(loss_dict.keys()))
    print("metric keys:", sorted(metrics_dict.keys()))


if __name__ == "__main__":
    main()

