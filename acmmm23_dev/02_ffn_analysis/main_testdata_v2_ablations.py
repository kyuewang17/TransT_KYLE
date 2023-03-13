from __future__ import absolute_import, print_function

import os
import numpy as np
import yaml
import wandb
import socket
import time
from copy import deepcopy
import torch
import itertools
from yacs.config import CfgNode
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from config import cfg
from miscs import set_logger, fix_seed
from data.trk_data_obj import BENCHMARK_DATA_OBJ
from data.test_loader import TEST_DATASET
from model import load_model
from loss.zoo import CLS_LOSS
from optim.zoo import init_optim
from main_utils import train, validate, test


# Set Important Paths
__PROJECT_MASTER_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE"

# Set Benchmark Dataset
__BENCHMARK_DATASET__ = "OTB100"
# __BENCHMARK_DATASET__ = "UAV123"

# Is Multiple Configuration Loading
__IS_MULTI_CFG_LOADING__ = False

# CUDA Device Configuration
__CUDA_DEVICE__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Memory Cut-off Percentage
__MEMORY_CUTOFF_PERCENT__ = 95

# Debugging Mode
__IS_DEBUG_MODE__ = False


def cfg_loader(logger, cfg_filepath, **kwargs):
    # Unpack KWARGS
    is_multi_loader = kwargs.get("is_multi_loader", False)
    assert isinstance(is_multi_loader, bool)
    ablation_cfg_filepath = kwargs.get("ablation_cfg_filepath")
    if is_multi_loader:
        assert ablation_cfg_filepath is not None
    ablation_cfg_shuffle_seed = kwargs.get("ablation_cfg_shuffle_seed", 333)
    assert isinstance(ablation_cfg_shuffle_seed, int) and ablation_cfg_shuffle_seed > 0

    # Load Configurations
    assert os.path.isfile(cfg_filepath)
    logger.info("Loading Configurations from: {}".format(cfg_filepath))
    cfg.merge_from_file(cfg_filename=cfg_filepath)
    time.sleep(1)
    logger.info("\n=== Configuration File Logging ===\n{}\n==================================".format(cfg))
    time.sleep(1)
    if is_multi_loader is False:
        return [cfg]
    else:
        assert os.path.isfile(ablation_cfg_filepath)
        assert ablation_cfg_filepath.split("/")[-1] == cfg_filepath.split("/")[-1]
        with open(ablation_cfg_filepath) as ablation_f:
            ablation_cfg = yaml.safe_load(ablation_f)
        logger.info("\nAblation Configuration File Loaded from: {}".format(ablation_cfg_filepath))
        time.sleep(1)
        return merge_ablation_cfg(
            ablation_cfg=ablation_cfg, shuffle_seed=ablation_cfg_shuffle_seed
        )


def merge_ablation_cfg(ablation_cfg, shuffle_seed=None):
    # Assertion first
    assert cfg.DATA.benchmark == ablation_cfg["BASE_VARS"]["benchmark"]
    if shuffle_seed is not None:
        assert isinstance(shuffle_seed, int) and shuffle_seed > 0
        np.random.seed(shuffle_seed)

    # Get Multiple Configurations for ablation first
    ablations, models = ablation_cfg["ABLATIONS"], ablation_cfg["MODELS"]

    # Filter Models
    new_models = {}
    for model_type, model_dict in models.items():
        if model_type in ablations["model_types"]:
            new_models[model_type] = model_dict
    models = new_models

    # Check "models" dictionary
    model_cfgs = {}
    for model_type, model_dict in models.items():
        _k, _v = zip(*model_dict.items())
        if not all(len(_v[0]) == len(__v) for __v in _v[1:]):
            raise AssertionError("Model params do not support permutations...!")
        _v_T = [list(i) for i in zip(*_v)]
        for _v_T_elem in _v_T:
            if model_type not in model_cfgs:
                model_cfgs[model_type] = [{k: v for k, v in zip(_k, _v_T_elem)}]
            else:
                model_cfgs[model_type].append({k: v for k, v in zip(_k, _v_T_elem)})

    # Get Permutations for Multiple Configurations using "ablations"
    _k, _v = zip(*ablations.items())
    multiple_cfgs = [dict(zip(_k, v)) for v in itertools.product(*_v)]

    # Traverse "multiple_cfgs" and apply "model_cfgs"
    new_multiple_cfgs = []
    for _cfg in multiple_cfgs:
        model_type = _cfg["model_types"]
        matched_model_cfgs = model_cfgs[model_type]
        for matched_model_cfg in matched_model_cfgs:
            _cfg_copy = deepcopy(_cfg)
            _cfg_copy[model_type] = matched_model_cfg
            new_multiple_cfgs.append(_cfg_copy)

    # Shuffle Multiple Configurations
    shuffle_perm = np.random.permutation(len(new_multiple_cfgs))
    multiple_cfgs = [new_multiple_cfgs[jj] for jj in shuffle_perm]

    # Merge with Configuration
    cfgs = []
    for _cfg in multiple_cfgs:
        # Copy original configuration
        new_cfg = deepcopy(cfg)

        # ==== Substitute as follows ==== #

        # Training Seed
        new_cfg.MAIN_PARAMS.train_seed = _cfg["train_seeds"]
        new_cfg.TRAIN.random_seed = _cfg["train_seeds"]

        # Batch Size
        new_cfg.MAIN_PARAMS.batch_size = _cfg["batch_sizes"]
        new_cfg.TRAIN.batch_size = _cfg["batch_sizes"]

        # Overlap Thresholds
        new_cfg.MAIN_PARAMS.overlap_thresholds = _cfg["overlap_thresholds"]
        new_cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_thresholds = _cfg["overlap_thresholds"]
        new_cfg.DATA.OPTS.TEST_DATASET.overlap_thresholds = _cfg["overlap_thresholds"]

        # Number of Epochs
        new_cfg.MAIN_PARAMS.num_epochs = _cfg["num_epochs"]
        new_cfg.TRAIN.num_epochs = _cfg["num_epochs"]
        new_cfg.SCHEDULER.CosineAnnealingLr.T_max = _cfg["num_epochs"]

        # Learning Rates
        new_cfg.MAIN_PARAMS.base_lr = _cfg["base_lrs"]
        new_cfg.TRAIN.base_lr = _cfg["base_lrs"]
        new_cfg.MAIN_PARAMS.min_lr = _cfg["min_lrs"]
        new_cfg.SCHEDULER.MultiStepLR.min_lr = _cfg["min_lrs"]
        new_cfg.SCHEDULER.CosineAnnealingLr.eta_min = _cfg["min_lrs"]

        # Loss Types
        new_cfg.MAIN_PARAMS.loss_type = _cfg["loss_types"]
        new_cfg.TRAIN.LOSS.type = _cfg["loss_types"]

        # Model Types & Models
        model_type = _cfg["model_types"]
        new_cfg.MAIN_PARAMS.model_type = model_type
        new_cfg.TRAIN.MODEL.type = model_type
        for model_param_name, model_params in _cfg[model_type].items():
            if hasattr(new_cfg.MODEL[model_type], model_param_name) is False:
                raise AssertionError()
            new_cfg.MODEL[model_type][model_param_name] = model_params

        # Append New Configurations
        cfgs.append(new_cfg)

    return cfgs


def run_mlp_model(trk_dataset, logger, cfgtion, device, **kwargs):
    assert isinstance(trk_dataset, TEST_DATASET)

    # Unpack Configurations
    random_seed = cfgtion.TRAIN.random_seed
    assert isinstance(random_seed, int) and random_seed > 0
    batch_size = cfgtion.TRAIN.batch_size
    assert isinstance(batch_size, int) and batch_size > 0
    num_epochs = cfgtion.TRAIN.num_epochs
    assert isinstance(num_epochs, int) and num_epochs > 0
    base_lr, weight_decay = cfgtion.TRAIN.base_lr, cfgtion.TRAIN.weight_decay
    loss_type, optim_type = cfgtion.TRAIN.LOSS.type, cfgtion.TRAIN.OPTIM.type
    if hasattr(cfgtion.OPTIM[optim_type], "momentum"):
        momentum = cfgtion.OPTIM[optim_type].momentum
    else:
        momentum = None
    if cfgtion.TRAIN.SCHEDULER.switch:
        scheduler_type = cfgtion.TRAIN.SCHEDULER.type
    else:
        scheduler_type = None
    is_validation = cfgtion.TRAIN.VALIDATION.switch
    assert isinstance(is_validation, bool)
    val_epoch_interval = cfgtion.TRAIN.VALIDATION.epoch_interval
    assert isinstance(val_epoch_interval, int) and 0 < val_epoch_interval < num_epochs
    if device is None:
        device = __CUDA_DEVICE__

    # Fix Seed
    fix_seed(random_seed=random_seed, logger=logger)

    # Split Trk Dataset
    if is_validation:
        train_ratio, val_ratio, test_ratio = \
            cfgtion.DATA.SPLIT_RATIO.train, cfgtion.DATA.SPLIT_RATIO.validation, cfgtion.DATA.SPLIT_RATIO.test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        train_dataset, _dataset = trk_dataset.split_obj(ratio=train_ratio, split_seed=random_seed)
        val_dataset, test_dataset = _dataset.split_obj(ratio=val_test_ratio, split_seed=random_seed)
    else:
        train_ratio = cfgtion.DATA.SPLIT_RATIO.train
        train_dataset, test_dataset = trk_dataset.split_obj(ratio=train_ratio, split_seed=random_seed)
        val_dataset = None

    # Wrap Dataset with PyTorch DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logger.info("\nTraining DataLoader Loaded Completely...! - # of Samples: [{:,}]".format(len(train_dataset)))
    if is_validation:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        logger.info("Validation DataLoader Loaded Completely...! - # of Samples: [{:,}]".format(len(val_dataset)))
    else:
        val_dataloader = None
        logger.info("Validation DataLoader Skipped...!")
        time.sleep(0.5)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Testing DataLoader Loaded Completely...! - # of Samples: [{:,}]".format(len(test_dataset)))
    time.sleep(0.5)

    # Load Model
    model = load_model(cfg=cfgtion)
    model.to(device=device)
    logger.info("Model Loaded...!")
    time.sleep(0.5)

    # Define Loss
    if loss_type == "WCE":
        train_class_ratio = train_dataset.compute_label_sums(cvt_to_ratio=True)
        ce_weights = 1 / train_class_ratio
        ce_weights /= ce_weights.sum()
    else:
        ce_weights = None
    criterion = CLS_LOSS(loss_type=loss_type, weights=ce_weights, device=device)
    logger.info("Criterion Loaded...!")
    time.sleep(0.5)

    # Define Optimizer
    optim = init_optim(
        optim_type=optim_type, model_params=model.parameters(),
        base_lr=base_lr, weight_decay=weight_decay, momentum=momentum,
    )
    logger.info("Optimizer Loaded...!")
    time.sleep(0.5)

    # Define Scheduler
    if scheduler_type is None:
        scheduler = None
        logger.info("Scheduler not Defined...!")
    else:
        if scheduler_type == "CosineAnnealingLr":
            cos_min_lr = cfgtion.SCHEDULER[scheduler_type].eta_min
            assert isinstance(cos_min_lr, float) and cos_min_lr < base_lr
            T_max = cfgtion.SCHEDULER[scheduler_type].T_max
            assert isinstance(T_max, int) and T_max <= num_epochs
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=optim, T_max=T_max, eta_min=cos_min_lr
            )
        else:
            raise NotImplementedError()
        logger.info("Scheduler Defined : [{}]".format(scheduler_type))
    time.sleep(0.5)

    # Iterative Training for each epoch
    for epoch in range(num_epochs):
        # Train Model
        model, train_loss, train_cls_metric_obj = train(
            train_dataloader=train_dataloader, model=model,
            criterion=criterion, optimizer=optim, epoch=epoch
        )

        # Compute Current Epoch's Classification Evaluations
        train_recalls = 100 * train_cls_metric_obj.compute_recalls()
        train_precisions = 100 * train_cls_metric_obj.compute_precisions()
        train_f1scores = 100 * train_cls_metric_obj.compute_f1scores()
        train_accuracy = 100 * train_cls_metric_obj.compute_accuracy()
        train_bal_accuracy = 100 * train_cls_metric_obj.compute_balanced_accuracy()

        # todo: Save Model (for each saving epoch) Later
        pass

        # Validate Model & Compute CLS Evaluations for Validation Set
        if is_validation:
            if epoch == 0 or (epoch + 1) % val_epoch_interval == 0:
                # Validate Model
                val_loss, val_cls_metric_obj = validate(
                    val_dataloader=val_dataloader, model=model,
                    criterion=criterion, epoch=epoch
                )

                # Compute Current Epoch's Classification Evaluations
                val_recalls = 100 * val_cls_metric_obj.compute_recalls()
                val_precisions = 100 * val_cls_metric_obj.compute_precisions()
                val_f1scores = 100 * val_cls_metric_obj.compute_f1scores()
                val_accuracy = 100 * val_cls_metric_obj.compute_accuracy()
                val_bal_accuracy = 100 * val_cls_metric_obj.compute_balanced_accuracy()
                val_TPR = 100 * val_cls_metric_obj.TPR()
                val_FNR = 100 * val_cls_metric_obj.FNR()
                val_FPR = 100 * val_cls_metric_obj.FPR()
                val_TNR = 100 * val_cls_metric_obj.TNR()

        # Scheduler
        if scheduler is not None:
            scheduler.step()

    # Test Model
    test_cls_metric_obj = \
        test(test_dataloader=test_dataloader, model=model)

    # Compute Current Epoch's Classification Evaluations
    test_recalls = 100 * test_cls_metric_obj.compute_recalls()
    test_precisions = 100 * test_cls_metric_obj.compute_precisions()
    test_f1scores = 100 * test_cls_metric_obj.compute_f1scores()
    test_accuracy = 100 * test_cls_metric_obj.compute_accuracy()
    test_bal_accuracy = 100 * test_cls_metric_obj.compute_balanced_accuracy()
    test_TPR = 100 * test_cls_metric_obj.TPR()
    test_FNR = 100 * test_cls_metric_obj.FNR()
    test_FPR = 100 * test_cls_metric_obj.FPR()
    test_TNR = 100 * test_cls_metric_obj.TNR()

    # todo: Save Final Model
    print(12345)


if __name__ == "__main__":
    # Logger
    _logger = set_logger()

    # Set Configuration File Paths
    if __BENCHMARK_DATASET__ == "OTB100":
        cfg_filepath = os.path.join(os.path.dirname(__file__), "cfgs", "base", "cfg_otb.yaml")
        ablation_cfg_filepath = os.path.join(
            os.path.dirname(__file__), "cfgs", "ablations", "cfg_otb.yaml"
        )
    elif __BENCHMARK_DATASET__ == "UAV123":
        cfg_filepath = os.path.join(os.path.dirname(__file__), "cfgs", "base", "cfg_uav.yaml")
        ablation_cfg_filepath = os.path.join(
            os.path.dirname(__file__), "cfgs", "ablations", "cfg_uav.yaml"
        )
    else:
        raise NotImplementedError()

    # Load Configurations via YAML file
    cfgs = cfg_loader(
        logger=_logger, cfg_filepath=cfg_filepath,
        is_multi_loader=__IS_MULTI_CFG_LOADING__, ablation_cfg_filepath=ablation_cfg_filepath
    )

    # Iterate for Configurations
    for _cfg_idx, _cfg in enumerate(cfgs):

        # Load Benchmark Data Object
        trk_data_obj = BENCHMARK_DATA_OBJ(
            logger=_logger,
            root_path=os.path.join(_cfg.DATA.root_path, _cfg.DATA.benchmark),
            benchmark=_cfg.DATA.benchmark,
            overlap_criterion=_cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_criterion,
            overlap_thresholds=_cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_thresholds,
            labeling_type=_cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.labeling_type,
            random_seed=_cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.random_seed,
            is_debug_mode=__IS_DEBUG_MODE__,
        )

        # Wrap with Test Loader
        trk_dataset = TEST_DATASET(data_obj=trk_data_obj, logger=_logger, init_mode="data_obj")

        # Run MLP Model Code
        run_mlp_model(
            trk_dataset=trk_dataset, logger=_logger, cfgtion=_cfg, device=__CUDA_DEVICE__,
        )

        pass
