from __future__ import absolute_import, print_function

import torch
import torch.nn as nn


def init_optim(optim_type, model_params, **kwargs):
    assert optim_type in ["SGD", "Adam", "Adagrad"]

    # Unpack KWARGS
    weight_decay = kwargs.get("weight_decay", 0)
    base_lr = kwargs.get("base_lr")

    # Case-wise Optimizer Return
    if optim_type == "SGD":
        momentum = kwargs.get("momentum")
        return torch.optim.SGD(
            model_params, lr=base_lr,
            momentum=momentum, weight_decay=weight_decay
        )

    elif optim_type == "Adam":
        beta1, beta2 = kwargs.get("beta1"), kwargs.get("beta2")
        return torch.optim.Adam(
            model_params, lr=base_lr, betas=(beta1, beta2),
            weight_decay=weight_decay
        )

    elif optim_type == "Adagrad":
        return torch.optim.Adagrad(
            model_params, lr=base_lr, weight_decay=weight_decay
        )

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    pass
