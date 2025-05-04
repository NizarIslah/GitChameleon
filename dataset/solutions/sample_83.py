# library: lightgbm
# version: 3.0.0
# extra_dependencies: []
import lightgbm.compat as compat


def decode_string(string: bytes) -> str:
    return compat.decode_string(string)
