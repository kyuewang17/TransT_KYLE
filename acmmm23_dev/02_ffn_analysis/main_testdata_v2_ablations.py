from __future__ import absolute_import, print_function

import os
import yaml
import wandb
import socket
import time
import torch
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

# CUDA Device Configuration
__CUDA_DEVICE__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Memory Cut-off Percentage
__MEMORY_CUTOFF_PERCENT__ = 95

# Debugging Mode
__IS_DEBUG_MODE__ = False


def cfg_loader(logger, cfg_filepath, is_multi_loader=False, ablation_cfg_filepath=None):
    assert os.path.isfile(cfg_filepath)
    logger.info("Loading Configurations from: {}".format(cfg_filepath))
    cfg.merge_from_file(cfg_filename=cfg_filepath)
    time.sleep(1)
    logger.info("\n=== Configuration File Logging ===\n{}\n==================================".format(cfg))
    time.sleep(1)
    if is_multi_loader is False:
        return cfg
    else:
        assert os.path.isfile(ablation_cfg_filepath)
        assert ablation_cfg_filepath.split("/")[-1] == cfg_filepath.split("/")[-1]
        with open(ablation_cfg_filepath) as ablation_f:
            ablation_cfg = yaml.safe_load(ablation_f)
        logger.info("\nAblation Configuration File Loaded from: {}".format(ablation_cfg_filepath))
        time.sleep(1)
        return merge_ablation_cfg(ablation_cfg=ablation_cfg)


def merge_ablation_cfg(ablation_cfg):

    print(12345)
    return cfg


def run_mlp_model(trk_dataset, logger, cfg, device, **kwargs):
    assert isinstance(trk_dataset, TEST_DATASET)

    # Unpack Configurations
    random_seed = cfg.TRAIN.random_seed
    assert isinstance(random_seed, int) and random_seed > 0
    batch_size = cfg.TRAIN.batch_size
    assert isinstance(batch_size, int) and batch_size > 0
    num_epochs = cfg.TRAIN.num_epochs
    assert isinstance(num_epochs, int) and num_epochs > 0
    base_lr, weight_decay = cfg.TRAIN.base_lr, cfg.TRAIN.weight_decay
    loss_type, optim_type = cfg.TRAIN.LOSS.type, cfg.TRAIN.OPTIM.type
    if hasattr(cfg.OPTIM[optim_type], "momentum"):
        momentum = cfg.OPTIM[optim_type].momentum
    else:
        momentum = None
    if cfg.TRAIN.SCHEDULER.switch:
        scheduler_type = cfg.TRAIN.SCHEDULER.type
    else:
        scheduler_type = None
    is_validation = cfg.TRAIN.VALIDATION.switch
    assert isinstance(is_validation, bool)
    val_epoch_interval = cfg.TRAIN.VALIDATION.epoch_interval
    assert isinstance(val_epoch_interval, int) and 0 < val_epoch_interval < num_epochs
    if device is None:
        device = __CUDA_DEVICE__

    # Fix Seed
    fix_seed(random_seed=random_seed, logger=logger)

    # Split Trk Dataset
    if is_validation:
        train_ratio, val_ratio, test_ratio = \
            cfg.DATA.SPLIT_RATIO.train, cfg.DATA.SPLIT_RATIO.validation, cfg.DATA.SPLIT_RATIO.test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        train_dataset, _dataset = trk_dataset.split_obj(ratio=train_ratio, split_seed=random_seed)
        val_dataset, test_dataset = _dataset.split_obj(ratio=val_test_ratio, split_seed=random_seed)
    else:
        train_ratio = cfg.DATA.SPLIT_RATIO.train
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
    model = load_model(cfg=cfg)
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
            cos_min_lr = cfg.SCHEDULER[scheduler_type].eta_min
            assert isinstance(cos_min_lr, float) and cos_min_lr < base_lr
            T_max = cfg.SCHEDULER[scheduler_type].T_max
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

    # Load Configurations via YAML file
    cfg = cfg_loader(
        logger=_logger,
        cfg_filepath=os.path.join(os.path.dirname(__file__), "cfgs", "base", "cfg_otb.yaml"),
        is_multi_loader=True,
        ablation_cfg_filepath=os.path.join(os.path.dirname(__file__), "cfgs", "ablations", "cfg_otb.yaml")
    )

    # Load Benchmark Data Object
    trk_data_obj = BENCHMARK_DATA_OBJ(
        logger=_logger,
        root_path=os.path.join(cfg.DATA.root_path, cfg.DATA.benchmark),
        benchmark=cfg.DATA.benchmark,
        overlap_criterion=cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_criterion,
        overlap_thresholds=cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_thresholds,
        labeling_type=cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.labeling_type,
        random_seed=cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.random_seed,
        is_debug_mode=__IS_DEBUG_MODE__,
    )

    # Wrap with Test Loader
    trk_dataset = TEST_DATASET(data_obj=trk_data_obj, logger=_logger, init_mode="data_obj")

    # Run MLP Model Code
    run_mlp_model(
        trk_dataset=trk_dataset, logger=_logger, cfg=cfg, device=__CUDA_DEVICE__,
    )

    pass
