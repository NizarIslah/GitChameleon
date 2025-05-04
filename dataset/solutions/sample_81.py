# library: lightgbm
# version: 3.0.0
# extra_dependencies: ['numpy==1.26.4']
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier


def predict_start(
    model: LGBMClassifier, data: np.ndarray, start_iter: int
) -> np.ndarray:
    return model.predict(data, start_iteration=start_iter)
