"""nerstudio-gnt/GNTConfig.py"""

from GNTDataManager import GNTDataManagerConfig
from GNTModel import GNTModelConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification


GNT = MethodSpecification(
    config=TrainerConfig(
        method_name="gnt",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=300000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=GNTDataManagerConfig(),
            model=GNTModelConfig(),
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
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description="Configuration for the GNT method.",
)
