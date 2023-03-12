from __future__ import absolute_import, print_function

from .bones import *


def load_model(cfg):
    # Model Initialization
    if cfg.MODEL.INITIALIZATION.switch:
        model_init_dist = cfg.MODEL.INITIALIZATION.distribution
    else:
        model_init_dist = None

    # Get Model type
    model_type = cfg.TRAIN.MODEL.type
    model_cfgs = cfg.MODEL[model_type]
    if model_type == "NN_statistics":
        model = NN_statistics(
            cls_loss=cfg.TRAIN.LOSS.type,
            overlap_criterion=cfg.DATA.OPTS.TEST_DATASET.overlap_criterion,
            dimensions=model_cfgs.dimensions, layers=model_cfgs.layers,
            hidden_activations=model_cfgs.hidden_activations,
            batchnorm_layers=model_cfgs.batchnorm_layers, dropout_probs=model_cfgs.dropout_probs,
            final_activation=model_cfgs.final_activation,
            init_dist=model_init_dist
        )
    elif model_type == "NN_dense_encoding":
        model = NN_dense_encoding(
            cls_loss=cfg.TRAIN.LOSS.type,
            overlap_criterion=cfg.DATA.OPTS.TEST_DATASET.overlap_criterion,
            dimensions=model_cfgs.dimensions, layers=model_cfgs.layers,
            hidden_activations=model_cfgs.hidden_activations,
            batchnorm_layers=model_cfgs.batchnorm_layers, dropout_probs=model_cfgs.dropout_probs,
            final_activation=model_cfgs.final_activation,
            init_dist=model_init_dist
        )
    else:
        raise NotImplementedError()

    return model


if __name__ == "__main__":
    pass
