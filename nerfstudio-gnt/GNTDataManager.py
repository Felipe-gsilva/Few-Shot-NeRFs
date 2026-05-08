"""Compatibility aliases for migrating to Nerfstudio's VanillaDataManager."""

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

GNTDataManager = VanillaDataManager
GNTDataManagerConfig = VanillaDataManagerConfig
