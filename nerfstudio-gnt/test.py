from GNTConfig import GNT
from nerfstudio.plugins.types import MethodSpecification

if __name__ == "__main__":
    assert isinstance(GNT, MethodSpecification)
    assert GNT.config.method_name == "gnt"
    print("GNT method specification import OK")
