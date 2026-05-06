from types import SimpleNamespace
from torch.utils.data import DataLoader
from GNT.gnt.sample_ray import RaySamplerSingleImage
from GNT.gnt.utils import cycle
import torch
import numpy as np
import copy
import json
import imageio.v2 as imageio

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from GNT.gnt.data_loaders.create_training_dataset import create_training_dataset
from GNT.gnt.data_loaders import dataset_dict
from GNT.gnt.data_loaders.data_utils import get_nearest_pose_ids

DATASET_SUBDIRS = {
    "spaces": Path("data/spaces_dataset/data/800"),
    "google_scanned": Path("data/google_scanned_objects"),
    "realestate": Path("data/RealEstate10K-subset"),
    "deepvoxels": Path("data/deepvoxels"),
    "nerf_synthetic": Path("data/nerf_synthetic"),
    "llff": Path("data/real_iconic_noface"),
    "ibrnet_collected": Path("data/ibrnet_collected_1"),
    "llff_test": Path("data/nerf_llff_data"),
    "shiny": Path("data/shiny"),
    "llff_render": Path("data/nerf_llff_data"),
    "shiny_render": Path("data/shiny"),
    "nerf_synthetic_render": Path("data/nerf_synthetic"),
    "nmr": Path("data/nmr"),
}


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
    dataset_weights: List[float] = field(default_factory=list)
    """Per-dataset sampling weights for train_dataset strings joined by '+'."""
    distributed: bool = False
    """Whether to use distributed training. If True, the data manager will use a DistributedSampler for the training dataset."""
    train_dataset: str = "llff"
    """Dataset backend for training. Use 'nerfstudio' for transforms.json scenes."""
    eval_dataset: str = "llff"
    """Dataset backend for evaluation. Use 'nerfstudio' for transforms.json scenes."""
    train_scenes: Optional[List[str]] = None
    """List of scenes to use for training. If None, all scenes in the training dataset will be used."""
    data_root: Path = Path(".")
    """User-facing dataset root. Upstream GNT loaders append 'data/<dataset>' to this root."""
    rootdir: Path = Path("data/")
    """Root directory for the dataset. Only used if train_dataset is not None."""
    local_rank: int = 0
    """Local rank for distributed training. Only used if distributed is True."""
    nerfstudio_near_plane: float = 0.1
    """Near depth for transforms.json scenes."""
    nerfstudio_far_plane: float = 6.0
    """Far depth for transforms.json scenes."""
    nerfstudio_eval_stride: int = 8
    """Every Nth frame is used for eval in transforms.json scenes."""


class NerfstudioTransformsDataset:
    def __init__(
        self,
        args: GNTDataManagerConfig,
        mode: Literal["train", "val", "test"] = "train",
    ) -> None:
        self.args = args
        self.mode = mode
        self.scene_root = args.data_root.expanduser().resolve()
        transforms_path = self.scene_root / "transforms.json"
        if not transforms_path.exists():
            raise FileNotFoundError(
                f"Expected transforms.json at '{transforms_path}' for train_dataset='nerfstudio'."
            )

        with transforms_path.open("r", encoding="utf-8") as f:
            transforms = json.load(f)
        frames = transforms.get("frames")
        if not isinstance(frames, list) or len(frames) < 2:
            raise ValueError("transforms.json must contain at least 2 frame entries.")

        self.global_w = transforms.get("w")
        self.global_h = transforms.get("h")
        self.global_fl_x = transforms.get("fl_x")
        self.global_fl_y = transforms.get("fl_y")
        self.global_cx = transforms.get("cx")
        self.global_cy = transforms.get("cy")
        self.global_camera_angle_x = transforms.get("camera_angle_x")

        self.image_paths: List[Path] = []
        self.poses: List[np.ndarray] = []
        self.intrinsics: List[np.ndarray] = []

        for frame in frames:
            if not isinstance(frame, dict):
                continue
            file_path = frame.get("file_path")
            transform_matrix = frame.get("transform_matrix")
            if not isinstance(file_path, str) or transform_matrix is None:
                continue
            image_path = Path(file_path)
            if not image_path.is_absolute():
                image_path = self.scene_root / image_path
            if not image_path.exists():
                continue
            pose = np.array(transform_matrix, dtype=np.float32)
            if pose.shape != (4, 4):
                continue
            h, w, intrinsics = self._resolve_intrinsics(frame, image_path)
            if intrinsics is None:
                continue
            self.image_paths.append(image_path)
            self.poses.append(pose)
            self.intrinsics.append(intrinsics)

        if len(self.image_paths) < 2:
            raise ValueError(
                "Need at least 2 valid frames in transforms.json to select source views."
            )

        all_ids = np.arange(len(self.image_paths))
        eval_ids = all_ids[:: max(1, args.nerfstudio_eval_stride)]
        if len(eval_ids) == 0:
            eval_ids = all_ids[-1:]
        eval_set = set(eval_ids.tolist())
        train_ids = np.array(
            [idx for idx in all_ids.tolist() if idx not in eval_set], dtype=np.int64
        )
        if len(train_ids) == 0:
            train_ids = all_ids

        if mode == "train":
            self.indices = train_ids
        else:
            self.indices = eval_ids
        self.depth_range = torch.tensor(
            [args.nerfstudio_near_plane, args.nerfstudio_far_plane], dtype=torch.float32
        )
        # VanillaPipeline expects datasets to expose scene_box and metadata.
        camera_centers = np.stack([pose[:3, 3] for pose in self.poses], axis=0)
        mins = torch.from_numpy(camera_centers.min(axis=0)).float()
        maxs = torch.from_numpy(camera_centers.max(axis=0)).float()
        pad = torch.maximum((maxs - mins) * 0.5, torch.tensor([1.0, 1.0, 1.0]))
        self.scene_box = SceneBox(aabb=torch.stack([mins - pad, maxs + pad], dim=0))
        self.metadata: Dict[str, torch.Tensor] = {}

    def _resolve_intrinsics(
        self, frame: dict, image_path: Path
    ) -> Tuple[int, int, Optional[np.ndarray]]:
        image = imageio.imread(str(image_path))
        if image.ndim != 3:
            return 0, 0, None
        h, w = int(image.shape[0]), int(image.shape[1])

        fl_x = frame.get("fl_x", self.global_fl_x)
        fl_y = frame.get("fl_y", self.global_fl_y)
        cx = frame.get("cx", self.global_cx)
        cy = frame.get("cy", self.global_cy)
        camera_angle_x = frame.get("camera_angle_x", self.global_camera_angle_x)

        if fl_x is None and camera_angle_x is not None:
            fl_x = 0.5 * w / np.tan(0.5 * float(camera_angle_x))
        if fl_y is None and fl_x is not None:
            fl_y = fl_x
        if cx is None:
            cx = w * 0.5
        if cy is None:
            cy = h * 0.5
        if fl_x is None or fl_y is None:
            return h, w, None

        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[0, 0] = float(fl_x)
        intrinsic[1, 1] = float(fl_y)
        intrinsic[0, 2] = float(cx)
        intrinsic[1, 2] = float(cy)
        return h, w, intrinsic

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        render_id = int(self.indices[idx])
        rgb = (
            imageio.imread(str(self.image_paths[render_id])).astype(np.float32) / 255.0
        )
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]

        render_pose = self.poses[render_id]
        render_intrinsics = self.intrinsics[render_id]
        h, w = rgb.shape[:2]
        camera = np.concatenate(
            ([h, w], render_intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        candidate_ids = (
            self.indices if self.mode == "train" else np.arange(len(self.image_paths))
        )
        candidate_poses = np.stack([self.poses[int(i)] for i in candidate_ids], axis=0)
        target_pose = self.poses[render_id]
        target_local_id = (
            int(np.where(candidate_ids == render_id)[0][0])
            if render_id in candidate_ids
            else -1
        )

        if self.mode == "train":
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            max_candidates = min(
                self.args.num_source_views * subsample_factor, len(candidate_ids) - 1
            )
            nearest_local_ids = get_nearest_pose_ids(
                target_pose,
                candidate_poses,
                max(1, max_candidates),
                tar_id=target_local_id,
                angular_dist_method="dist",
            )
            num_select = min(
                len(nearest_local_ids),
                max(1, self.args.num_source_views + np.random.randint(low=-2, high=3)),
            )
        else:
            nearest_local_ids = get_nearest_pose_ids(
                target_pose,
                candidate_poses,
                min(self.args.num_source_views, len(candidate_ids) - 1),
                tar_id=target_local_id,
                angular_dist_method="dist",
            )
            num_select = min(len(nearest_local_ids), max(1, self.args.num_source_views))

        if len(nearest_local_ids) == 0:
            raise ValueError(
                "No source views available; transforms scene needs at least 2 valid frames."
            )
        selected_local_ids = np.random.choice(
            nearest_local_ids, num_select, replace=False
        )
        selected_global_ids = [
            int(candidate_ids[int(local_id)])
            for local_id in selected_local_ids.tolist()
        ]

        src_rgbs: List[np.ndarray] = []
        src_cameras: List[np.ndarray] = []
        for src_id in selected_global_ids:
            src_rgb = (
                imageio.imread(str(self.image_paths[src_id])).astype(np.float32) / 255.0
            )
            if src_rgb.shape[-1] > 3:
                src_rgb = src_rgb[..., :3]
            src_h, src_w = src_rgb.shape[:2]
            src_intrinsics = self.intrinsics[src_id]
            src_pose = self.poses[src_id]
            src_camera = np.concatenate(
                ([src_h, src_w], src_intrinsics.flatten(), src_pose.flatten())
            ).astype(np.float32)
            src_rgbs.append(src_rgb)
            src_cameras.append(src_camera)

        return {
            "rgb": torch.from_numpy(rgb),
            "camera": torch.from_numpy(camera),
            "rgb_path": str(self.image_paths[render_id]),
            "src_rgbs": torch.from_numpy(np.stack(src_rgbs, axis=0)),
            "src_cameras": torch.from_numpy(np.stack(src_cameras, axis=0)),
            "depth_range": self.depth_range.clone(),
        }


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
        self.config.rootdir = self._resolve_rootdir(self.config.data_root)
        self._populate_default_dataset_weights()
        super().__init__()

        self.setup_train()
        if self.test_mode in ["test", "val"]:
            self.setup_eval()

    @staticmethod
    def _resolve_rootdir(data_root: Path) -> Path:
        resolved = data_root.expanduser().resolve()
        return resolved.parent if resolved.name == "data" else resolved

    def _populate_default_dataset_weights(self) -> None:
        if "+" not in self.config.train_dataset:
            return
        if self.config.dataset_weights:
            return
        dataset_count = len(self.config.train_dataset.split("+"))
        self.config.dataset_weights = [1.0 / dataset_count] * dataset_count

    def _validate_dataset_path(self, dataset_name: str) -> None:
        if dataset_name == "nerfstudio":
            transforms_path = (
                self.config.data_root.expanduser().resolve() / "transforms.json"
            )
            if transforms_path.exists():
                return
            raise FileNotFoundError(
                f"Dataset 'nerfstudio' expects transforms.json at '{transforms_path}'."
            )
        dataset_subdir = DATASET_SUBDIRS.get(dataset_name)
        if dataset_subdir is None:
            return
        expected_dir = self.config.rootdir / dataset_subdir
        if expected_dir.exists():
            return
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' expects directory '{expected_dir}'. "
            f"Set data_root to the parent of 'data/' (or to '.../data')."
        )

    def setup_train(self):
        if self.config.train_dataset is None:
            raise ValueError(
                "train_dataset must be set in GNTDataManagerConfig. "
                f"Available datasets: {list(dataset_dict.keys())}"
            )
        for dataset_name in self.config.train_dataset.split("+"):
            self._validate_dataset_path(dataset_name)
        if self.config.train_dataset == "nerfstudio":
            train_dataset = NerfstudioTransformsDataset(self.config, mode="train")
            train_sampler = None
        else:
            train_dataset, train_sampler = create_training_dataset(self.config)
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
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
        for dataset_name in eval_config.train_dataset.split("+"):
            self._validate_dataset_path(dataset_name)
        if eval_config.train_dataset == "nerfstudio":
            eval_dataset = NerfstudioTransformsDataset(eval_config, mode="val")
        else:
            eval_dataset, _ = create_training_dataset(eval_config)
        self.eval_dataset = eval_dataset
        self.eval_loader = DataLoader(eval_dataset, batch_size=1)
        self.eval_loader_iter = iter(cycle(self.eval_loader))

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        train_data = next(self.train_loader_iter)
        ray_sampler = RaySamplerSingleImage(train_data, self.device)
        ray_batch = ray_sampler.random_sample(
            self.config.train_num_rays_per_batch,
            sample_mode="uniform"
            if self.config.sample_mode == "all"
            else self.config.sample_mode,
            center_ratio=self.config.center_ratio,
        )
        depth_range = ray_batch["depth_range"]  # (1, 2)
        N = ray_batch["ray_o"].shape[0]
        near = depth_range[:, 0:1].expand(N, 1)  # (N, 1)
        far = depth_range[:, 1:2].expand(N, 1)  # (N, 1)
        # Squeeze the DataLoader batch-size-1 leading dim from scene-context tensors.
        # Wrap them in a SimpleNamespace so TensorDataclass._get_dict_batch_shapes
        # skips them entirely (it only recurses into plain dicts, not arbitrary objects).
        ctx = SimpleNamespace(
            depth_range=depth_range.squeeze(0),  # (2,)
            camera=ray_batch["camera"].squeeze(0),  # (34,)
            src_rgbs=ray_batch["src_rgbs"].squeeze(0),  # (K, H, W, 3)
            src_cameras=ray_batch["src_cameras"].squeeze(0),  # (K, 34)
        )
        ray_bundle = RayBundle(
            origins=ray_batch["ray_o"],
            directions=ray_batch["ray_d"],
            pixel_area=torch.ones_like(ray_batch["ray_o"][..., :1]),  # (N, 1)
            nears=near,
            fars=far,
            metadata={"ctx": ctx},
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
        depth_range = ray_batch["depth_range"]  # (1, 2)
        N = ray_batch["ray_o"].shape[0]
        near = depth_range[:, 0:1].expand(N, 1)  # (N, 1)
        far = depth_range[:, 1:2].expand(N, 1)  # (N, 1)
        ctx = SimpleNamespace(
            depth_range=depth_range.squeeze(0),
            camera=ray_batch["camera"].squeeze(0),
            src_rgbs=ray_batch["src_rgbs"].squeeze(0),
            src_cameras=ray_batch["src_cameras"].squeeze(0),
        )
        ray_bundle = RayBundle(
            origins=ray_batch["ray_o"],
            directions=ray_batch["ray_d"],
            pixel_area=torch.ones_like(ray_batch["ray_o"][..., :1]),  # (N, 1)
            nears=near,
            fars=far,
            metadata={"ctx": ctx},
        )
        batch = {
            "camera": ray_batch["camera"],
            "rgb": ray_batch["rgb"],
        }
        return ray_bundle, batch

    def iter_train(self):
        return self.train_loader_iter

    def iter_eval(self):
        return self.eval_loader_iter

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        # eval renders full images, so H*W — return a sentinel if unknown
        return self.config.train_num_rays_per_batch
