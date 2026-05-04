from torch.utils.data import DataLoader
from GNT.gnt.sample_ray import RaySamplerSingleImage
from GNT.utils import cycle
import torch
import numpy as np
import copy

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from GNT.gnt.data_loaders.create_training_dataset import create_training_dataset
from GNT.gnt.data_loaders import dataset_dict


@dataclass
class GNTDataManagerConfig(DataManagerConfig):
    """Configuration for the GNT data manager.

    Args:
        _target: The target class to instantiate, in this case, GNTDataManager
    """

    _target: Type = field(default_factory=lambda: GNTDataManager, init=False)
    """The target class to instantiate, in this case, GNTDataManager."""
    center_ratio: float = 0.5
    """Ratio of rays to sample from the center of the image. Only used if sample_mode is 'center'."""
    sample_mode: Literal["all", "center"] = "all"
    """Mode for sampling rays. 'all' samples rays uniformly across the image, while 'center' samples a portion of rays from the center of the image based on center_ratio."""
    train_num_rays_per_batch: int = 1024
    """Number of rays to sample per batch during training."""
    eval_scenes: Optional[List[str]] = None
    """List of scenes to use for evaluation. If None, all scenes in the eval dataset will be used."""
    num_source_views: int = 10
    """Number of source views to use for each target view. Only used for training."""
    render_stride: int = 1
    """Stride for rendering during evaluation. A stride of 1 means rendering every pixel, while a stride of 2 means rendering every other pixel, etc."""
    dataset_weights = dataset_dict
    """Dictionary mapping dataset names to their corresponding weights for sampling during training. Only used if multiple datasets are used for training."""
    distributed = False
    """Whether to use distributed training. If True, the data manager will use a DistributedSampler for the training dataset."""
    train_dataset: str = "llff"
    """Default GNT implementation for creating training datasets. If None, the data manager will load datasets normally without GNT's custom dataset creation function."""
    eval_dataset: str = "llff"
    """Default GNT implementation for creating evaluation datasets. If None, the data manager will use the same dataset as train_dataset for evaluation."""
    train_scenes = None
    """List of scenes to use for training. If None, all scenes in the training dataset will be used."""
    rootdir = Path("data/")
    """Root directory for the dataset. Only used if train_dataset is not None."""
    local_rank = 0
    """Local rank for distributed training. Only used if distributed is True."""


class GNTDataManager(DataManager):
    config: GNTDataManagerConfig

    def __init__(
        self,
        config: GNTDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.test_mode = test_mode
        self.world_size = world_size
        self.local_rank = local_rank
        self.config.local_rank = local_rank
        self.config.distributed = world_size > 1

        self.setup_train()
        if self.test_mode in ["test", "val"]:
            self.setup_eval()

    def setup_train(self):
        if self.config.train_dataset is None:
            raise ValueError(
                "train_dataset must be set in GNTDataManagerConfig. "
                f"Available datasets: {list(dataset_dict.keys())}"
            )
        train_dataset, train_sampler = create_training_dataset(self.config)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            worker_init_fn=lambda _: np.random.seed(),
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler,
            shuffle=train_sampler is None,
        )
        self.train_loader_iter = iter(cycle(self.train_loader))

    def setup_eval(self):
        eval_config = copy.copy(self.config)
        eval_config.train_dataset = self.config.eval_dataset
        eval_dataset, _ = create_training_dataset(eval_config)
        self.eval_loader = DataLoader(eval_dataset, batch_size=1)
        self.eval_loader_iter = iter(cycle(self.eval_loader))

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        train_data = next(self.train_loader_iter)
        ray_sampler = RaySamplerSingleImage(train_data, self.device)
        ray_batch = ray_sampler.random_sample(
            self.config.train_num_rays_per_batch,
            sample_mode=self.config.sample_mode,
            center_ratio=self.config.center_ratio,
        )
        ray_bundle = RayBundle(
            origins=ray_batch["ray_o"],
            directions=ray_batch["ray_d"],
            pixel_area=torch.ones_like(ray_batch["ray_o"][..., :1]),
            nears=ray_batch["near"],
            fars=ray_batch["far"],
        )
        batch = {
            "src_rgbs": ray_batch["src_rgbs"],
            "src_cameras": ray_batch["src_cameras"],
            "camera": ray_batch["camera"],
            "rgb": ray_batch["rgb"],
        }
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        val_data = next(self.eval_loader_iter)
        ray_sampler = RaySamplerSingleImage(
            val_data, self.device, render_stride=self.config.render_stride
        )
        ray_batch = ray_sampler.get_all()
        ray_bundle = RayBundle(
            origins=ray_batch["ray_o"],
            directions=ray_batch["ray_d"],
            pixel_area=torch.ones_like(ray_batch["ray_o"][..., :1]),
            nears=ray_batch["near"],
            fars=ray_batch["far"],
            metadata={
                "src_rgbs": ray_batch["src_rgbs"],
                "src_cameras": ray_batch["src_cameras"],
            },
        )
        batch = {
            "camera": ray_batch["camera"],
            "rgb": ray_batch["rgb"],
        }
        return ray_bundle, batch

    def iter_train(self):
        return self.train_loader_iter

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        # eval renders full images, so H*W — return a sentinel if unknown
        return self.config.train_num_rays_per_batch
