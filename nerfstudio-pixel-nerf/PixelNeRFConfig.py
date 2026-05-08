"""nerstudio-pixel-nerf/PixelNeRF.py"""

from PixelNeRFDataManager import PixelNeRFDataManagerConfig
from PixelNeRFModel import PixelNeRFModelConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from PixelNerfPipeline import PixelNerfPipelineConfig


PixelNeRF = MethodSpecification(
    config=TrainerConfig(
        method_name="pixel-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=300000,
        mixed_precision=True,
        pipeline=PixelNerfPipelineConfig(
            datamanager=PixelNeRFDataManagerConfig(),
            model=PixelNeRFModelConfig(),
        ),
        optimizers={
            "network": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=300000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 11),
        vis="tensorboard",
    ),
    description="Configuration for the PixelNeRF method",
)
