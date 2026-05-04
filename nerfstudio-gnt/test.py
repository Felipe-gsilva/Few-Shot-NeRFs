from GNTDataManager import GNTDataManager, GNTDataManagerConfig
from GNTModel import GNTModel, GNTModelConfig


datamanager = GNTDataManager(
    config=GNTDataManagerConfig(),
)

model = GNTModel(
    config=GNTModelConfig(),
    scene_box=datamanager.scene_box,
    num_train_data=len(datamanager.train_loader.dataset),
)

print(datamanager.config)
print(model.config)
