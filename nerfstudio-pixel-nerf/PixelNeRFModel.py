from dataclasses import dataclass, field
from typing import Type
from nerfstudio.models.base_model import Model, ModelConfig
from pixel_nerf import make_mlp


@dataclass 
class PixelNeRFModelConfig(ModelConfig): 
    """PixelNeRFModelConfig """
    _target: Type = field(default_factory=lambda: PixelNeRFModel, init=False)


class PixelNeRFModel(Model): 
    """PixelNeRFModel """
    config: PixelNeRFModelConfig

    def populate_modules(self):
        """Populates the modules used for the PixelNeRFModel."""
        self.mlp_file = make_mlp(config.mlp_file_config)
