from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler

from GNTModel import GNTModelConfig


def cameras_to_gnt_format(
    cameras: Cameras, image_idx: int, device: torch.device
) -> torch.Tensor:
    """Convert one Nerfstudio camera entry to GNT's 34D camera vector.

    Layout matches GNT projector expectations exactly:
    [H, W, intrinsics(4x4 flattened), c2w(4x4 flattened)].
    """

    idx = int(image_idx)
    dtype = torch.float32

    h = cameras.height[idx].reshape(-1)[0].to(device=device, dtype=dtype)
    w = cameras.width[idx].reshape(-1)[0].to(device=device, dtype=dtype)
    fx = cameras.fx[idx].reshape(-1)[0].to(device=device, dtype=dtype)
    fy = cameras.fy[idx].reshape(-1)[0].to(device=device, dtype=dtype)
    cx = cameras.cx[idx].reshape(-1)[0].to(device=device, dtype=dtype)
    cy = cameras.cy[idx].reshape(-1)[0].to(device=device, dtype=dtype)

    intrinsics = torch.eye(4, device=device, dtype=dtype)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy

    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, :4] = cameras.camera_to_worlds[idx].to(device=device, dtype=dtype)

    return torch.cat(
        [torch.stack([h, w], dim=0), intrinsics.reshape(-1), c2w.reshape(-1)], dim=0
    )


@dataclass
class GNTPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: GNTPipeline)
    datamanager: VanillaDataManagerConfig = field(
        default_factory=lambda: VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
        )
    )
    model: GNTModelConfig = field(default_factory=GNTModelConfig)


class GNTPipeline(VanillaPipeline):
    config: GNTPipelineConfig
    datamanager: VanillaDataManager

    def _extract_target_image_idx(self, batch: Dict) -> int:
        if "image_idx" in batch:
            image_idx = batch["image_idx"]
            return int(
                image_idx.reshape(-1)[0].item()
                if torch.is_tensor(image_idx)
                else image_idx
            )

        if "indices" in batch:
            indices = batch["indices"]
            if torch.is_tensor(indices):
                return int(indices.reshape(-1, indices.shape[-1])[0, 0].item())

        raise KeyError(
            "Could not find target image index in batch; expected 'image_idx' or 'indices'."
        )

    @profiler.time_function
    def _sample_source_views(
        self, target_image_idx: int, split: Literal["train", "eval"]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset = (
            self.datamanager.train_dataset
            if split == "train"
            else self.datamanager.eval_dataset
        )
        cameras = dataset.cameras
        device = next(self.model.parameters()).device
        num_images = int(cameras.camera_to_worlds.shape[0])
        if num_images < 2:
            raise ValueError("Need at least 2 images to sample source views.")

        all_indices = torch.arange(num_images, device=device)
        target_idx = int(target_image_idx)
        source_candidates = all_indices[all_indices != target_idx]
        if source_candidates.numel() == 0:
            raise ValueError("No source views available after excluding target image.")

        num_src = min(
            int(self.model.config.num_source_views), int(source_candidates.numel())
        )
        perm = torch.randperm(source_candidates.numel(), device=device)[:num_src]
        source_indices = source_candidates[perm]

        src_rgbs = []
        src_cameras = []
        for src_idx in source_indices.tolist():
            sample = dataset[int(src_idx)]
            if "image" not in sample:
                raise KeyError("Dataset sample must include an 'image' tensor.")

            src_rgb = sample["image"]
            if not torch.is_tensor(src_rgb):
                src_rgb = torch.as_tensor(src_rgb)
            src_rgb = src_rgb[..., :3].to(device=device, dtype=torch.float32)
            if src_rgb.max() > 1.0:
                src_rgb = src_rgb / 255.0
            src_rgbs.append(src_rgb)

            src_cameras.append(cameras_to_gnt_format(cameras, src_idx, device=device))

        src_rgbs_tensor = torch.stack(src_rgbs, dim=0).unsqueeze(
            0
        )  # (1, N_src, H, W, 3)
        src_cameras_tensor = torch.stack(src_cameras, dim=0).unsqueeze(
            0
        )  # (1, N_src, 34)
        return src_rgbs_tensor, src_cameras_tensor

    @profiler.time_function
    def _inject_gnt_metadata(
        self, ray_bundle: RayBundle, batch: Dict, split: Literal["train", "eval"]
    ) -> Dict:
        device = next(self.model.parameters()).device
        target_idx = self._extract_target_image_idx(batch)
        src_rgbs, src_cameras = self._sample_source_views(target_idx, split=split)

        dataset = (
            self.datamanager.train_dataset
            if split == "train"
            else self.datamanager.eval_dataset
        )
        camera = cameras_to_gnt_format(
            dataset.cameras, target_idx, device=device
        ).unsqueeze(0)  # (1, 34)
        depth_range = (
            torch.stack([ray_bundle.nears.min(), ray_bundle.fars.max()], dim=0)
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
        )  # (1, 2)

        metadata = dict(ray_bundle.metadata) if ray_bundle.metadata is not None else {}
        metadata["src_rgbs"] = src_rgbs
        metadata["src_cameras"] = src_cameras
        metadata["camera"] = camera
        metadata["depth_range"] = depth_range
        ray_bundle.metadata = metadata

        if "image" in batch and "rgb" not in batch:
            batch["rgb"] = batch["image"].to(device=device, dtype=torch.float32)
        elif "rgb" in batch:
            batch["rgb"] = batch["rgb"].to(device=device, dtype=torch.float32)
        return batch

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        batch = self._inject_gnt_metadata(ray_bundle, batch, split="train")
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        self.eval()
        with torch.no_grad():
            ray_bundle, batch = self.datamanager.next_eval(step)
            batch = self._inject_gnt_metadata(ray_bundle, batch, split="eval")
            model_outputs = self.model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict
