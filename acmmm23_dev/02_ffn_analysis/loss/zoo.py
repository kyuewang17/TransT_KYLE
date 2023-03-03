from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import numpy as np


# CUDA Device Configuration
__CUDA_DEVICE__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BASE_LOSS(nn.Module):
    def __init__(self):
        super(BASE_LOSS, self).__init__()

        self.task_type = "cls"

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class CLS_LOSS(BASE_LOSS):
    def __init__(self, loss_type, **kwargs):
        super(CLS_LOSS, self).__init__()

        assert loss_type in ["CE", "WCE"]
        self.loss_type = loss_type
        self.device = kwargs.get("device", __CUDA_DEVICE__)

        # Define Loss
        if self.loss_type == "CE":
            self.loss_func = nn.CrossEntropyLoss()

        elif self.loss_type == "WCE":
            weights = kwargs.get("weights")
            assert isinstance(weights, (torch.Tensor, np.ndarray))
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights)
            weights = weights.to(device=self.device, dtype=torch.float32)
            self.loss_func = nn.CrossEntropyLoss(weight=weights)

        else:
            raise NotImplementedError()

    def forward(self, pred, target):
        # Return Loss
        return self.loss_func(pred, target)


if __name__ == "__main__":
    pass
