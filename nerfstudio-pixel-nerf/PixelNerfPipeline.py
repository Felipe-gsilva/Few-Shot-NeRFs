from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig

from PixelNeRFDataManager import PixelNeRFDataManagerConfig, PixelNeRFDataManager
from PixelNeRFModel import PixelNeRFModelConfig


@dataclass
class PixelNerfPipelineConfig(VanillaPipelineConfig):
    """Configuração da Pipeline do PixelNeRF."""
    
    # CORREÇÃO: Aponta para a classe da Pipeline real
    _target: Type = field(default_factory=lambda: PixelNerfPipeline, init=False)
    
    datamanager: PixelNeRFDataManagerConfig = field(
        default_factory=lambda: PixelNeRFDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
        )
    )
    
    model: PixelNeRFModelConfig = field(default_factory=PixelNeRFModelConfig)


class PixelNerfPipeline(VanillaPipeline):
    """Pipeline orquestradora do PixelNeRF."""
    
    config: PixelNerfPipelineConfig
    datamanager: PixelNeRFDataManager
