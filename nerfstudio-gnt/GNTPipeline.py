from dataclasses import dataclass, field
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils import profiler
from GNTDataManager import GNTDataManager, GNTDataManagerConfig
from GNTModel import GNTModel, GNTModelConfig
import torch

import torch.nn as nn


@dataclass
class GNTPipelineConfig:
    """Config for the GNTPipeline. This is where you can add any hyperparameters or
    settings that you want to be able to easily change when initializing the pipeline.
    """

    model_config: GNTModelConfig = field(default_factory=GNTModelConfig)
    datamanager_config: GNTDataManagerConfig = field(
        default_factory=GNTDataManagerConfig
    )
    scene_box: SceneBox = field(
        default_factory=lambda: SceneBox(
            aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)
        )
    )
    num_train_data: int = 0


class GNTPipeline(nn.Module):
    model: GNTModel
    datamanager: GNTDataManager

    def __init__(self, config: GNTPipelineConfig | None = None):
        super().__init__()
        if config is None:
            config = GNTPipelineConfig()
        self.model = GNTModel(
            config=config.model_config,
            scene_box=config.scene_box,
            num_train_data=config.num_train_data,
        )
        self.datamanager = GNTDataManager(config.datamanager_config)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict
