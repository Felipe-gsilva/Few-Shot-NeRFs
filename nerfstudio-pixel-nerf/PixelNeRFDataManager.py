from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

import random
import torch


@dataclass
class PixelNeRFDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the GNT data manager.

    Args:
        _target: The target class to instantiate, in this case, GNTDataManager
    """

    _target: Type = field(default_factory=lambda: PixelNeRFDataManager, init=False)
    num_source_views: int = 3
    """Number of source views to sample from the dataset for conditioning the pixelNeRF model. The paper typically uses 3 views, but you can experiment with this number."""


class PixelNeRFDataManager(VanillaDataManager):
    config: PixelNeRFDataManagerConfig

    def __init__(
        self,
        config: PixelNeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        **kwargs,
    ):
        # 1. Chame o super() para herdar todo o carregamento de dataset do Nerfstudio!
        super().__init__(config=config, device=device, test_mode=test_mode, **kwargs)

    def _sample_source_views(self, num_views: int) -> Dict[str, torch.Tensor]:
        """
        Sorteia N imagens do dataset para atuar como contexto (source views)
        e as formata para o pixelNeRF.
        """
        dataset = self.train_dataset
        # I should implement some smarter sampling strategy here, but for now I'll just randomly sample N views from the dataset.
        indices = random.sample(range(len(dataset)), num_views)

        src_rgbs = []
        src_poses = []
        focals = []
        cs = []

        for idx in indices:
            data = dataset[idx]
            src_rgbs.append(data["image"])
            camera = dataset.cameras[idx]
            c2w_3x4 = camera.camera_to_worlds
            c2w_4x4 = torch.cat(
                [c2w_3x4, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=c2w_3x4.device)],
                dim=0,
            )
            src_poses.append(c2w_4x4)

            focals.append(torch.tensor([camera.fx.item(), camera.fy.item()]))
            cs.append(torch.tensor([camera.cx.item(), camera.cy.item()]))

        return {
            "src_rgbs": torch.stack(src_rgbs).unsqueeze(0),
            "src_cameras": torch.stack(src_poses).unsqueeze(0),
            "focal": torch.stack(focals),
            "c": torch.stack(cs),
        }

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = self.train_pixel_sampler.sample(
            self.config.train_num_rays_per_batch
        )
        source_data = self._sample_source_views(self.config.num_source_views)
        ray_bundle.metadata.update(source_data)

        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """A mesma lógica de next_train, mas usando o eval_pixel_sampler."""
        ray_bundle, batch = self.eval_pixel_sampler.sample(
            self.config.eval_num_rays_per_batch
        )
        source_data = self._sample_source_views(self.config.num_source_views)
        ray_bundle.metadata.update(source_data)
        return ray_bundle, batch
