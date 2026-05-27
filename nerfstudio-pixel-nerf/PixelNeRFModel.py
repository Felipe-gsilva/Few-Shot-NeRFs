import dataclasses
from pathlib import Path
import sys

production_pixelnerf_src = str(Path(__file__).parent / "pixelnerf" / "src")
if production_pixelnerf_src not in sys.path:
    sys.path.insert(0, production_pixelnerf_src)

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Tuple, cast
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from torch.nn import Parameter
from pixelnerf.src.model.models import PixelNeRFNet
from pixelnerf.src.render import NeRFRenderer
from pyhocon import ConfigFactory
from dotmap import DotMap

import os
import torch
import nerfstudio.utils.profiler as profiler


def download_pretrained_pixelnerf_weights():
    import gdown

    url = "https://drive.google.com/file/d/1UO_rL201guN6euoWkCOn-XpqR2e8o6ju"
    output = "pixelnerf_pretrained.pth"
    if not os.path.exists(output):
        print("Downloading pretrained PixelNeRF weights...")
        gdown.download(url, output, quiet=False)

    else:
        print("Pretrained PixelNeRF weights already downloaded.")

    unzipped_path = "pixelnerf_pretrained"
    if not os.path.exists(unzipped_path):
        print("Unzipping pretrained weights...")
        import zipfile

        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(unzipped_path)
        print(f"Pretrained weights downloaded and unzipped to {unzipped_path}")


@dataclass
class PixelNeRFModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: PixelNeRFModel, init=False)
    ckpt_path: Optional[str] = None
    """Path to a .pth checkpoint file to load the network weights from. If not provided, will look for .pth files in the output directory and load the latest one."""
    pretrained_ckpt_path: Optional[str] = None
    """Path to a .pth checkpoint file containing pretrained weights to initialize the network with. This is different from ckpt_path in that it will only be used to initialize the network weights and will not be reloaded during training. If not provided, will use the official pretrained weights from the paper (https://drive.google.com/file/d/1UO_rL201guN6euoWkCOn-XpqR2e8o6ju)"""
    transfer_learning: bool = False
    """Whether to use the pretrained weights for transfer learning."""
    no_reload: bool = False
    """If True, will not attempt to load from any checkpoint and will always train from scratch."""
    out_dir: str = "outputs"
    """Subdirectory of the output directory to save checkpoints and logs for this model. If not provided, will use 'default_exp'."""
    exp_name: str = "default_exp"
    """Name of the experiment, used as a subdirectory under out_dir to save checkpoints and logs. If not provided, will use 'default_exp'."""
    encoder: Dict[str, Any] = field(
        default_factory=lambda: {
            "backbone": "resnet34",
            "pretrained": True,
            "num_layers": 4,
        },
        metadata={
            "help": "Configuration for the pixelNeRF encoder. Currently using the paper default configuration"
        },
    )
    mlp_coarse: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "resnet",
            "n_blocks": 3,
            "d_hidden": 512,
        },
        metadata={
            "help": "Configuration for the pixelNeRF coarse MLP. Currently using the paper default configuration"
        },
    )

    mlp_fine: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "resnet",
            "n_blocks": 4,
            "d_hidden": 512,
        },
        metadata={
            "help": "Configuration for the pixelNeRF fine MLP. Currently using the paper default configuration, which is the same as the coarse MLP but with one additional block."
        },
    )
    renderer: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_coarse": 64,
            "n_fine": 32,
            "n_fine_depth": 64,
            "depth_std": 0.01,
            "white_bkgd": False,
        },
        metadata={
            "help": "Configuration for the pixelNeRF renderer. Currently using the paper default configuration"
        },
    )
    loss: Dict[str, Any] = field(
        default_factory=lambda: {
            "rgb": {"use_l1": False},
            "rgb_fine": {"use_l1": False},
            "alpha": {"lambda_alpha": 0.0, "clamp_alpha": 100, "init_epoch": 5},
        },
        metadata={
            "help": "Configuration for the pixelNeRF loss. Currently using the paper default configuration "
        },
    )
    lindisp: bool = False
    """Whether to sample linearly in disparity (inverse depth) rather than depth. Paper defines it troughout dataset preprocessing, so we keep it as a config option but set it to False by default since it's not commonly used in nerf implementations."""


class PixelNeRFModel(Model):
    config: PixelNeRFModelConfig

    def __init__(self, config, scene_box=None, num_train_data=0, **kwargs):
        if scene_box is None:
            scene_box = SceneBox(
                aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)
            )
        super().__init__(
            config=config, scene_box=scene_box, num_train_data=num_train_data, **kwargs
        )

    def populate_modules(self):
        super().populate_modules()

        if dataclasses.is_dataclass(self.config) and not isinstance(self.config, type):
            conf_dict = dataclasses.asdict(self.config)
        else:
            conf_dict = vars(self.config)

        pixelnerf_conf = ConfigFactory.from_dict(conf_dict)

        self.net = PixelNeRFNet(pixelnerf_conf)
        self.renderer = NeRFRenderer.from_conf(pixelnerf_conf)

        if torch.cuda.is_available():
            print(f"Using {torch.cuda.device_count()} GPUs for parallelization")
            self.renderer = self.renderer.bind_parallel(
                self.net, gpus=list(range(torch.cuda.device_count()))
            ).eval()

        if self.config.no_reload:
            print("Not loading from ckpt, training from scratch...")
        else:
            if self.config.transfer_learning:
                self.load_from_ckpt(self.config.pretrained_ckpt_path, force_latest=True)
                print("Using pretrained weights for transfer learning...")
                self.freeze_net()
            else:
                print("Loading from ckpt if available...")
                self.load_from_ckpt(self.config.out_dir)

    def freeze_net(self):
        for param in self.net.encoder.parameters():
            param.requires_grad = False

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return_map = {}
        return_map["encoder"] = (
            list(self.net.encoder.parameters())
            if self.net.encoder.requires_grad_ == True
            else []
        )
        return_map["nerf"] = list(self.net.mlp_coarse.parameters()) + list(
            self.net.mlp_fine.parameters() if self.net.mlp_fine is not None else []
        )
        return return_map

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """The paper calcs loss"""
        loss = torch.nn.functional.mse_loss(outputs["rgb_coarse"], batch["rgb"])
        if "rgb_fine" in outputs:
            loss = loss + torch.nn.functional.mse_loss(
                outputs["rgb_fine"], batch["rgb"]
            )
        return {"rgb_loss": loss}

    def get_metrics(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """The paper only reports PSNR, but you can add more metrics here if you want."""
        pred = outputs.get("rgb_fine", outputs["rgb_coarse"])
        gt = batch["rgb"].to(pred.device)
        psnr = -10.0 * torch.log10(torch.mean((pred - gt) ** 2).clamp_min(1e-10))
        return {"psnr": psnr}

    def get_image_metrics_and_images(
        self, outputs, batch
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        pred = outputs.get("rgb_fine", outputs["rgb_coarse"])
        gt = batch["rgb"].to(pred.device)
        psnr = -10.0 * torch.log10(torch.mean((pred - gt) ** 2).clamp_min(1e-10))
        return {"psnr": float(psnr.item())}, {"rgb": pred, "rgb_gt": gt}

    def load_from_ckpt(self, out_folder, force_latest=False):
        if not os.path.exists(out_folder):
            print("No ckpts found, training from scratch...")
            return 0
        ckpts = sorted(
            [
                os.path.join(out_folder, f)
                for f in os.listdir(out_folder)
                if f.endswith(".pth")
            ]
        )
        if self.config.ckpt_path and not force_latest:
            if os.path.isfile(self.config.ckpt_path):
                ckpts = [self.config.ckpt_path]
        if ckpts and not self.config.no_reload:
            fpath = ckpts[-1]
            self.net.load_state_dict(torch.load(fpath, map_location="cpu"))
            print(f"Reloading from {fpath}")
            return int(fpath[-10:-4])
        print("No ckpts found, training from scratch...")
        return

    @profiler.time_function
    def get_outputs(
        self, ray_bundle: RayBundle | Cameras
    ) -> Dict[str, torch.Tensor | List]:
        assert isinstance(ray_bundle, RayBundle)
        device = next(self.net.parameters()).device
        metadata = ray_bundle.metadata or {}

        for key in ("src_rgbs", "src_cameras", "focal", "c"):
            if key not in metadata:
                raise KeyError(
                    f"Missing metadata key '{key}' — pipeline must inject source views"
                )

        src_images = (
            metadata["src_rgbs"].squeeze(0).permute(0, 3, 1, 2).to(device)
        )  # (NS, 3, H, W)
        src_poses = metadata["src_cameras"].squeeze(0).to(device)  # (NS, 4, 4)

        focal = ray_bundle.metadata["focal"][0].unsqueeze(0).to(device)
        c = ray_bundle.metadata["c"][0].unsqueeze(0).to(device)

        self.net.encode(
            src_images.unsqueeze(0),
            src_poses.unsqueeze(0),
            focal,
            c=c,
        )

        if ray_bundle.nears is None:
            ray_bundle.nears = torch.zeros_like(ray_bundle.origins[..., :1])
        if ray_bundle.fars is None:
            ray_bundle.fars = torch.ones_like(ray_bundle.origins[..., :1])

        rays = torch.cat(
            [
                ray_bundle.origins.to(device),
                ray_bundle.directions.to(device),
                ray_bundle.directions.to(device),
                ray_bundle.nears.to(device),
                ray_bundle.fars.to(device),
            ],
            dim=-1,
        ).unsqueeze(0)

        render_dict = DotMap(self.renderer(rays, want_weights=True))

        outputs: Dict[str, torch.Tensor | List] = {
            "rgb_coarse": render_dict.coarse.rgb.squeeze(0),
            "depth_coarse": render_dict.coarse.depth.squeeze(0),
            "weights_coarse": render_dict.coarse.weights.squeeze(0),
        }
        if len(render_dict.fine) > 0:
            outputs["rgb_fine"] = render_dict.fine.rgb.squeeze(0)
            outputs["depth_fine"] = render_dict.fine.depth.squeeze(0)
            outputs["weights_fine"] = render_dict.fine.weights.squeeze(0)

        outputs["rgb"] = outputs.get("rgb_fine", outputs["rgb_coarse"])
        outputs["accumulation"] = outputs.get(
            "weights_fine", outputs["weights_coarse"]
        ).sum(dim=-1)
        outputs["depth"] = outputs.get("depth_fine", outputs["depth_coarse"])

        return cast(Dict[str, torch.Tensor | List], outputs)
