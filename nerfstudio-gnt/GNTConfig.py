"""nerstudio-gnt/GNTConfig.py"""

from GNTDataManager import GNTDataManagerConfig
from GNTModel import GNTModelConfig
from GNTTrainer import GNTTrainerConfig
from GNTPipeline import GNTPipelineConfig
from nerfstudio.plugins.types import MethodSpecification


GNT = MethodSpecification(
    config=GNTTrainerConfig(
        method_name="gnt",
        pipeline_config=GNTPipelineConfig(
            model_config=GNTModelConfig(),
            datamanager_config=GNTDataManagerConfig(),
        ),
    ),
    description="Configuration for the GNT method.",
)
