from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from torch.nn import Parameter
from pixelnerf.src.model import make_model
from pyhocon import ConfigFactory

import os
import torch
import nerfstudio.utils.profiler as profiler


@dataclass
class PixelNeRFModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: PixelNeRFModel, init=False)
    conf_path: str = "pixelnerf/conf/resnet_fine_mv.conf"
    """Path to the PyHocon .conf file that configures the pixelNeRF network."""
    num_source_views: int = 3
    N_samples: int = 64
    N_importance: int = 64
    white_bkgd: bool = False
    inv_uniform: bool = False
    ckpt_path: Optional[str] = None
    no_reload: bool = False
    out_dir: str = "outputs"
    exp_name: str = "default_exp"


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
        conf = ConfigFactory.parse_file(self.config.conf_path)

        self.net = make_model(conf["model"])
        self.renderer = NeRFRenderer.from_conf(
            conf["renderer"],
            lindisp=self.config.lindisp,
        )
        # bind_parallel expects gpu ids — use [0] or derive from device later
        # don't call it here since device isn't available yet

        self.z_near = self.config.z_near
        self.z_far = self.config.z_far
        self.load_from_ckpt()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        # pixelNeRF typically uses same lr for everything
        return {"network": list(self.net.parameters())}

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        loss = torch.nn.functional.mse_loss(outputs["rgb_coarse"], batch["rgb"])
        if "rgb_fine" in outputs:
            loss = loss + torch.nn.functional.mse_loss(
                outputs["rgb_fine"], batch["rgb"]
            )
        return {"rgb_loss": loss}

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        return {}

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
        return 0

    @profiler.time_function
    def get_outputs(
        self, ray_bundle: RayBundle | Cameras
    ) -> Dict[str, torch.Tensor | List]:
        assert isinstance(ray_bundle, RayBundle)
        device = next(self.net.parameters()).device
        metadata = ray_bundle.metadata or {}

        src_images = (
            metadata["src_rgbs"].squeeze(0).permute(0, 3, 1, 2)
        )  # (NS, 3, H, W)
        src_poses = metadata["src_cameras"].squeeze(0)  # (NS, 4, 4)
        focal = metadata["focal"].to(device)  # (NS, 2) or scalar
        c = metadata["c"].to(device)  # (NS, 2) or None

        self.net.encode(
            src_images.unsqueeze(0),  # (1, NS, 3, H, W) — SB=1
            src_poses.unsqueeze(0),  # (1, NS, 4, 4)
            focal,
            c=c,
        )

        # Build pixelNeRF ray format: (SB, N_rays, 8)
        # 8 = [origin(3), direction(3), near(1), far(1)]
        rays = (
            torch.cat(
                [
                    ray_bundle.origins,  # (N, 3)
                    ray_bundle.directions,  # (N, 3)
                    ray_bundle.nears,  # (N, 1)
                    ray_bundle.fars,  # (N, 1)
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .to(device)
        )  # (1, N, 8)

        render_par = self.renderer.bind_parallel(self.net, [device.index or 0]).eval()
        render_dict = DotMap(render_par(rays, want_weights=True))

        out = {"rgb_coarse": render_dict.coarse.rgb.squeeze(0)}  # (N, 3)
        if len(render_dict.fine) > 0:
            out["rgb_fine"] = render_dict.fine.rgb.squeeze(0)  # (N, 3)
        return cast(Dict[str, torch.Tensor | List], out)
