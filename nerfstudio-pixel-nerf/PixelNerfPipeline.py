from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler

from PixelNerfModel import PixelNerfModelConfig

@dataclass
class PixelNerfPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: PixelNerfPipeline)
    datamanager: VanillaDataManagerConfig = field(
        default_factory=lambda: VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
        )
    )
    model: PixelNerfModelConfig = field(default_factory=PixelNerfModelConfig)


class PixelNerfPipeline(VanillaPipeline):
    config: PixelNerfPipelineConfig
    datamanager: VanillaDataManager

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        self.eval()
        with torch.no_grad():
            ray_bundle, batch = self.datamanager.next_eval(step)
            model_outputs = self.model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict
