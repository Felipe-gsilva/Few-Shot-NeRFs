from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from nerfstudio.engine.callbacks import TrainingCallback
from nerfstudio.engine.optimizers import Optimizers

from GNTPipeline import GNTPipeline, GNTPipelineConfig


@dataclass
class GNTTrainerConfig:
    _target: Type = field(default_factory=lambda: GNTTrainer, init=False)
    """Configuration for the GNT trainer."""
    method_name: str = "gnt"
    """Method name. Required to set in python or via cli"""
    pipeline_config: GNTPipelineConfig = field(default_factory=GNTPipelineConfig)
    """Pipeline configuration for the GNT trainer."""
    optimizers_config: Dict[str, Any] = field(default_factory=dict)
    """Dictionary of optimizer groups and their schedulers"""


class GNTTrainer:
    pipeline: GNTPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: GNTTrainerConfig):
        self.pipeline = GNTPipeline(config.pipeline_config)
        self.optimizers = Optimizers(
            config.optimizers_config, self.pipeline.model.get_param_groups()
        )
        self.callbacks = []
