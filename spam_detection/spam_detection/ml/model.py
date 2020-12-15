from typing import Union

import numpy as np
import torch

# Hyperparameters that are relevant at inference time.
PARAMS = {
    'embedding_dim': 16,

}

class Model:

    def __init__(self):
        self.embedding_dim = PARAMS['embedding_dim']


    def predict(self, x: Union[torch.Tensor, np.array]):
        pass

    def _init_artifacts(self):
        pass