"""
This is a GNTModel implementation for NeRFStudio. You can see I moslty copied the code from @GNT.gnt.model.py and modified it to fit the NeRFStudio framework. Credits belong moslty to https://github.com/VITA-Group/GNT.
"""

import os
import re
import torch

from types import SimpleNamespace
from nerfstudio.data.scene_box import SceneBox
from dataclasses import dataclass, field
from nerfstudio.cameras.cameras import Cameras
from torch.nn import Parameter
from typing import Dict, List, Optional, Type, cast
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from GNT.gnt.feature_network import ResUNet
from GNT.gnt.projection import Projector
from GNT.gnt.transformer_network import GNT
from GNT.gnt.render_ray import render_rays


def download_pretrained_gnt_model(path):
    from gdown import download

    url = "https://drive.google.com/file/d/1YvOJXa5eGpKgoMYcxC2ma7prB1n5UwRn/"  # Replace with actual URL
    output = path
    download(url, output, quiet=False)
    print(f"Pretrained GNT model downloaded to {output}")


@dataclass
class GNTModelConfig(ModelConfig):
    """Config for the GNTModel. This is where you can set hyperparameters for your model, such as the number of layers, hidden dimensions, etc."""

    _target: Type = field(default_factory=lambda: GNTModel, init=False)
    """The target class for this config. This is used by NeRFStudio to instantiate the model when you specify this config in your pipeline config."""
    transfer_learning: bool = False
    """Whether to use transfer learning from a pretrained model. If True, the model will load from the checkpoint specified in ckpt_path and will not train the feature net."""
    coarse_feat_dim: int = 48
    """The number of feature channels for the coarse MLP."""
    fine_feat_dim: int = 48
    """The number of feature channels for the fine MLP."""
    single_net: bool = False
    """Whether to use a single MLP for both coarse and fine sampling. If True, only net_coarse will be used and net_fine will be set to None."""
    N_samples: int = 64
    """The number of samples to take along each ray for the coarse MLP."""
    N_importance: int = 64
    """The number of additional samples to take along each ray for the fine MLP. If 0, no fine sampling will be done."""
    num_source_views: int = 10
    """The number of source views to use for feature extraction. This is used to determine the input dimensions for the feature MLP."""
    sample_mode: str = "center"
    """The mode for sampling rays during training. Options are 'all' for uniform sampling across the image and 'center' for sampling a portion of rays from the center of the image."""
    inv_uniform: bool = False
    """Whether to use inverse uniform sampling for the coarse MLP. If True, samples will be taken more densely near the camera and less densely further away."""
    white_bkgd: bool = False
    """Whether to use a white background for rendering. If False, a black background will be used."""
    out_dir: str = "./outputs"
    """The directory to save outputs such as checkpoints and rendered images."""
    exp_name: str = "gnt_exp"
    """The name of the experiment. This is used to create a subdirectory in out_dir for saving outputs."""
    netwidth: int = 256
    """The width of the MLPs for both the coarse and fine networks."""
    transdepth: int = 4
    """The number of layers in the MLPs for both the coarse and fine networks."""
    ckpt_path: Optional[str] = None
    """The path to a checkpoint to load the model from. If None, the model will be initialized from scratch."""
    pretrained_ckpt_path: Optional[str] = None
    """The path to a pretrained checkpoint for transfer learning. If None and transfer_learning is True, the model will attempt to download a pretrained checkpoint from a hardcoded URL."""
    no_reload: bool = False
    """If True, will not load from existing checkpoints even if they exist."""


class GNTModel(Model):
    config: GNTModelConfig
    """Set the model config so that Python gives you typed autocomplete!"""
    net_coarse: torch.nn.Module
    """The coarse MLP that takes in the ray samples and outputs the RGB and density values"""
    net_fine: Optional[torch.nn.Module]
    """The fine MLP that takes in the ray samples and outputs the RGB and density values"""
    feature_net: torch.nn.Module
    """The feature MLP that takes in the ray samples and outputs the features for the ray samples"""
    optimizer: torch.optim.Optimizer
    """The optimizer for training the model"""
    scheduler: torch.optim.lr_scheduler._LRScheduler
    """The learning rate scheduler for training the model"""
    start_step: int
    """The starting step for training. This is useful for resuming training from a checkpoint."""
    projector: Projector
    """The projector for projecting the 3D points to 2D image space. This is used for sampling the features from the feature net."""

    def __init__(
        self,
        config: GNTModelConfig,
        scene_box: Optional[SceneBox] = None,
        num_train_data: int = 0,
        **kwargs,
    ):
        if scene_box is None:
            scene_box = SceneBox(
                aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)
            )

        super().__init__(
            config=config, scene_box=scene_box, num_train_data=num_train_data, **kwargs
        )

        if self.config.transfer_learning:
            ckpt = self.config.pretrained_ckpt_path or "gnt_pretrained_model.ckpt"
            if not os.path.exists(ckpt):
                download_pretrained_gnt_model(ckpt)
            self.config.pretrained_ckpt_path = ckpt

    def _load_pretrained(self, ckpt_path):
        assert os.path.isfile(ckpt_path), (
            f"Checkpoint path {ckpt_path} does not exist or is not a file."
        )

        print(f"Loading pretrained weights for Transfer Learning from: {ckpt_path}")

        weights = torch.load(ckpt_path, map_location="cpu")

        def load_weights_safely(module, weight_entry):
            if isinstance(weight_entry, torch.nn.Module):
                module.load_state_dict(weight_entry.state_dict())
            else:
                module.load_state_dict(weight_entry)

        if "feature_net" in weights:
            load_weights_safely(self.feature_net, weights["feature_net"])
        if "net_coarse" in weights:
            load_weights_safely(self.net_coarse, weights["net_coarse"])
        if "net_fine" in weights and self.net_fine is not None:
            load_weights_safely(self.net_fine, weights["net_fine"])

        # freeze feat network
        self._freeze_feature_net()

    def _freeze_feature_net(self):
        print("Transfer Learning Active: Freezing Feature Network.")
        for param in self.feature_net.parameters():
            param.requires_grad = False

        self.feature_net.eval()

    def populate_modules(self):
        super().populate_modules()

        args = SimpleNamespace(
            netwidth=self.config.netwidth,
            trans_depth=self.config.transdepth,
        )

        self.net_coarse = GNT(
            args,
            in_feat_ch=self.config.coarse_feat_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            ret_alpha=self.config.N_importance > 0,
        )

        self.net_fine = (
            None
            if self.config.single_net
            else GNT(
                args,
                in_feat_ch=self.config.fine_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=True,
            )
        )

        self.feature_net = ResUNet(
            coarse_out_ch=self.config.coarse_feat_dim,
            fine_out_ch=self.config.fine_feat_dim,
            single_net=self.config.single_net,
        )

        self.projector = Projector(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.config.transfer_learning and self.config.ckpt_path is not None:
            self._load_pretrained(self.config.ckpt_path)

        # resume checkpoint
        out_folder = os.path.join(self.config.out_dir, self.config.exp_name, "ckpts")
        self.start_step = self.load_from_ckpt(out_folder)

        # re-freeze (load_from_ckpt resets requires_grad)
        if self.config.transfer_learning:
            self._freeze_feature_net()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups passed downstream to NeRFStudio's optimizers."""
        param_groups = {}

        # Transformer/rendering networks always train
        param_groups["network"] = list(self.net_coarse.parameters())
        if self.net_fine is not None:
            param_groups["network"] += list(self.net_fine.parameters())

        if not self.config.transfer_learning:
            param_groups["feature_net"] = list(self.feature_net.parameters())
        else:
            param_groups["feature_net"] = []

        return param_groups

    def train(self, mode: bool = True):
        """Override the standard PyTorch train mode toggle to keep feature_net frozen."""
        super().train(mode)
        if self.config.transfer_learning:
            self.feature_net.eval()

        return self

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks, such as updating a density grid for Instant NGP."""
        return []

    def get_outputs(
        self, ray_bundle: RayBundle | Cameras
    ) -> Dict[str, torch.Tensor | List]:
        assert isinstance(ray_bundle, RayBundle), (
            "GNTModel only supports RayBundle, not Cameras"
        )
        device = next(self.parameters()).device
        self.projector.device = device

        metadata = ray_bundle.metadata or {}
        required = ("src_rgbs", "src_cameras", "camera", "depth_range")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise KeyError(
                f"Missing GNT metadata in ray bundle: {missing}. "
                "GNTPipeline must inject source views and camera context."
            )

        src_rgbs = cast(torch.Tensor, metadata["src_rgbs"]).to(
            device=self.device, dtype=torch.float32
        )  # (1, K, H, W, 3)
        src_cameras = cast(torch.Tensor, metadata["src_cameras"]).to(
            device=self.device, dtype=torch.float32
        )  # (1, K, 34)
        camera = cast(torch.Tensor, metadata["camera"]).to(
            device=self.device, dtype=torch.float32
        )  # (1, 34)
        depth_range = cast(torch.Tensor, metadata["depth_range"]).to(
            device=self.device, dtype=torch.float32
        )  # (1, 2)

        featmaps = self.feature_net(src_rgbs.squeeze(0).permute(0, 3, 1, 2))

        ray_batch = {
            "ray_o": ray_bundle.origins,
            "ray_d": ray_bundle.directions,
            "depth_range": depth_range,  # (1, 2)
            "src_rgbs": src_rgbs,  # (1, K, H, W, 3)
            "src_cameras": src_cameras,  # (1, K, 34)
            "camera": camera,  # (1, 34)
        }
        ret = render_rays(
            ray_batch=ray_batch,
            model=self,
            projector=self.projector,
            featmaps=featmaps,
            N_samples=self.config.N_samples,
            inv_uniform=self.config.inv_uniform,
            N_importance=self.config.N_importance,
            det=not self.training,
            white_bkgd=self.config.white_bkgd,
            ret_alpha=self.config.N_importance > 0,
            single_net=self.config.single_net,
        )

        if "outputs_fine" in ret and ret["outputs_fine"] is not None:
            ret["rgb"] = ret["outputs_fine"]["rgb"]
            ret["depth"] = (
                ret["outputs_fine"]["depth"]
                if "depth" in ret["outputs_fine"]
                else ret["outputs_fine"]["z_vals"]
            )
        elif "outputs_coarse" in ret and ret["outputs_coarse"] is not None:
            ret["rgb"] = ret["outputs_coarse"]["rgb"]
            ret["depth"] = (
                ret["outputs_coarse"]["depth"]
                if "depth" in ret["outputs_coarse"]
                else ret["outputs_coarse"]["z_vals"]
            )
        else:
            raise ValueError(
                "render_rays must return at least 'outputs_coarse' with 'rgb' and 'depth' or 'z_vals'."
            )

        return cast(Dict[str, torch.Tensor | List], ret)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        pred_key = (
            "outputs_fine"
            if "outputs_fine" in outputs and outputs["outputs_fine"] is not None
            else "outputs_coarse"
        )

        predicted_rgb = outputs[pred_key]["rgb"]
        batch_rgb = batch.get("rgb", batch.get("image"))
        if batch_rgb is None:
            raise KeyError(
                "Batch must contain either 'rgb' or 'image' key for ground truth RGB values."
            )

        gt_rgb = batch_rgb.to(predicted_rgb.device)

        # I could use MSELoss or L1Loss from nerfstudio, but since I only need MSE for PSNR calculation, I'll just compute it directly here to avoid unnecessary overhead.
        # future test can be done
        mse = torch.nn.functional.mse_loss(predicted_rgb, gt_rgb)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
        metrics_dict["psnr"] = psnr
        return metrics_dict

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        pred_key = (
            "outputs_fine"
            if "outputs_fine" in outputs and outputs["outputs_fine"] is not None
            else "outputs_coarse"
        )

        loss = torch.nn.functional.mse_loss(
            outputs[pred_key]["rgb"], batch.get("rgb", batch.get("image"))
        )
        return {"rgb_loss": loss}

    def get_image_metrics_and_images(self, outputs, batch):
        assert "rgb" in outputs, (
            "render_rays must return 'rgb' in outputs for metric calculation."
        )

        pred_key = (
            "outputs_fine"
            if "outputs_fine" in outputs and outputs["outputs_fine"] is not None
            else "outputs_coarse"
        )

        predicted_rgb = outputs[pred_key]["rgb"]  # (N, 3)
        batch_rgb = batch.get("rgb", batch.get("image"))
        if batch_rgb is None:
            raise KeyError(
                "Batch must contain either 'rgb' or 'image' key for ground truth RGB values."
            )

        gt_rgb = batch_rgb.to(predicted_rgb.device)  # (N, 3)

        mse = torch.mean((predicted_rgb - gt_rgb) ** 2).clamp_min(1e-10)
        psnr = -10.0 * torch.log10(mse)

        return {"psnr": float(psnr.item())}, {"rgb": predicted_rgb, "rgb_gt": gt_rgb}

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        if self.net_fine is not None:
            self.net_fine.train()
        if self.config.transfer_learning:
            self.feature_net.eval()
        else:
            self.feature_net.train()

    def save_model(self, filename):
        to_save = {
            "net_coarse": self.net_coarse.state_dict(),
            "feature_net": self.feature_net.state_dict(),
        }
        if self.net_fine is not None:
            to_save["net_fine"] = self.net_fine.state_dict()
        torch.save(to_save, filename)

    def load_model(self, filename):
        to_load = torch.load(filename, map_location="cpu")
        self.net_coarse.load_state_dict(to_load["net_coarse"])
        self.feature_net.load_state_dict(to_load["feature_net"])
        if self.net_fine is not None and "net_fine" in to_load:
            self.net_fine.load_state_dict(to_load["net_fine"])

    def load_from_ckpt(self, out_folder, force_latest_ckpt=False):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        if self.config.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.config.ckpt_path):
                ckpts = [self.config.ckpt_path]

        if ckpts and not self.config.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath)
            m = re.search(r"(\d+)\.pth$", fpath)
            step = int(m.group(1)) if m else 0
            print(f"Reloading from {fpath}, starting at step={step}")
            return step

        print("No ckpts found, training from scratch...")
        return 0
