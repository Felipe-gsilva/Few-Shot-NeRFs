from dataclasses import dataclass, field
from typing import Type
from nerfstudio.models.base_model import Model, ModelConfig


@dataclass 
class PixelNeRFModelConfig(ModelConfig): 
    """PixelNeRFModelConfig """
    _target: Type = field(default_factory=lambda: PixelNeRFModel, init=False)


class PixelNeRFModel(Model): 
    """PixelNeRFModel """
    config: PixelNeRFModelConfig

    def populate_modules(self):
        pass
