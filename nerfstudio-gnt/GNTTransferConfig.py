from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from GNTPipeline import GNTPipelineConfig
from GNTModel import GNTModelConfig

GNT_TRANSFER = MethodSpecification(
    config=TrainerConfig(
        method_name="gnt-transfer",
        pipeline=GNTPipelineConfig(
            model=GNTModelConfig(
                transfer_learning=True,
                ckpt_path="/home/ubuntu/nerfstudio-gnt/pretrained_models/gnt_synthetic_nerf.ckpt",
            ),
        ),
        optimizers={
            "network": {
                "optimizer": AdamOptimizerConfig(lr=1e-3),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=100000,
                ),
            },
            "feature_net": {
                "optimizer": AdamOptimizerConfig(lr=1e-3),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=100000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 11),
        vis="tensorboard",
    ),
    description="Generalizable NeRF Transformer (GNT)",
)
