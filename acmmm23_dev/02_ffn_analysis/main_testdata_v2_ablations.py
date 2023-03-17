from __future__ import absolute_import, print_function

import os
import numpy as np
import yaml
import wandb
import socket
import time
import datetime
from copy import deepcopy
import torch
import itertools
from yacs.config import CfgNode
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from config import cfg
from miscs import set_logger, fix_seed, get_current_datetime_str
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

# Get Experiment Machine List
# --> Assign "Hostname" for each experiment machine
#     "hostname" can be obtained using "socket.gethostname()"
__EXP_MACHINE_LIST__ = [
    # "PIL-kyle",
    "carpenters1",
    "carpenters2",
]

# CUDA Device Configuration
__CUDA_DEVICE__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Memory Cut-off Percentage
__MEMORY_CUTOFF_PERCENT__ = 95

# Debugging Mode
__IS_DEBUG_MODE__ = False


def cfg_loader(logger, cfg_filepath, **kwargs):
    # Unpack KWARGS
    exp_machine_list = kwargs.get("exp_machine_list")
    assert isinstance(exp_machine_list, list) and len(exp_machine_list) > 0
    curr_hostname = socket.gethostname()
    assert curr_hostname in exp_machine_list
    ablation_cfg_filepath = kwargs.get("ablation_cfg_filepath")
    if len(exp_machine_list) > 1:
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
    if len(exp_machine_list) == 1:
        return [cfg]
    else:
        assert os.path.isfile(ablation_cfg_filepath)
        assert ablation_cfg_filepath.split("/")[-1] == cfg_filepath.split("/")[-1]
        with open(ablation_cfg_filepath) as ablation_f:
            ablation_cfg = yaml.safe_load(ablation_f)
        logger.info("\nAblation Configuration File Loaded from: {}".format(ablation_cfg_filepath))
        time.sleep(1)
        cfgs = merge_ablation_cfg(
            ablation_cfg=ablation_cfg, shuffle_seed=ablation_cfg_shuffle_seed
        )
        return split_cfgs(cfgs=cfgs, exp_machine_list=exp_machine_list)[curr_hostname]


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


def split_cfgs(cfgs, exp_machine_list):
    chunks = len(cfgs) // len(exp_machine_list)
    cfgs_split = {}
    for jj, exp_machine_name in enumerate(exp_machine_list):
        if jj < len(exp_machine_list) - 1:
            cfgs_split[exp_machine_name] = cfgs[jj * chunks:(jj + 1) * chunks]
        else:
            cfgs_split[exp_machine_name] = cfgs[jj * chunks:]
    return cfgs_split


def generate_exp_name(cfgtion):
    # Generate Overlap Threshold String
    overlap_thresh_str = "["
    overlap_thresholds = cfgtion.MAIN_PARAMS.overlap_thresholds
    for i_idx, overlap_thresh in enumerate(overlap_thresholds):
        if i_idx < len(overlap_thresholds) - 1:
            overlap_thresh_str += "{}_".format(overlap_thresh)
        else:
            overlap_thresh_str += "{}]".format(overlap_thresh)

    # Experiment Machine and Date String
    exp_machine_date_str = "[HOST_{}]__[DATE_{}]".format(
        socket.gethostname(), get_current_datetime_str("(%y-%m-%d)-(%H-%M-%S)")
    )

    # Experiment Key Information String
    exp_key_info_str = "[DATA_{}]__[SEED_{}]__[OVTHR_{}]__[MODEL_{}]__[LOSS_{}]__[BSZ_{}]__[Lr_{}-{}]".format(
        cfg.MAIN_PARAMS.benchmark, cfg.MAIN_PARAMS.train_seed, overlap_thresh_str, cfg.MAIN_PARAMS.model_type,
        cfg.MAIN_PARAMS.loss_type, cfg.MAIN_PARAMS.batch_size, cfg.MAIN_PARAMS.base_lr, cfg.MAIN_PARAMS.min_lr
    )

    # Concatenate Experiment Strings
    exp_str = "{}__{}".format(exp_machine_date_str, exp_key_info_str)
    return exp_str


def convert_cfgtion_to_wandb_cfg(cfgtion):
    # Set WandB Configuration
    wandb_cfg = {
        "seed": cfgtion.MAIN_PARAMS.train_seed,
        "overlap_thresholds": cfgtion.MAIN_PARAMS.overlap_thresholds,
        "batch_size": cfgtion.MAIN_PARAMS.batch_size, "num_epochs": cfgtion.MAIN_PARAMS.num_epochs,
        "loss_type": cfgtion.MAIN_PARAMS.loss_type, "model_type": cfgtion.MAIN_PARAMS.model_type,

        # Data Information
        "benchmark": cfgtion.MAIN_PARAMS.benchmark,
        "benchmark_shuffle_seed": cfgtion.DATA.OPTS.BENCHMARK_DATA_OBJ.random_seed,
        "split_ratio": {
            "train": cfgtion.DATA.SPLIT_RATIO.train,
            "validation": int(cfgtion.TRAIN.VALIDATION.switch) * cfgtion.DATA.SPLIT_RATIO.validation,
            "test": (1 - int(cfgtion.TRAIN.VALIDATION.switch)) * cfgtion.DATA.SPLIT_RATIO.validation + cfgtion.DATA.SPLIT_RATIO.test
        },

        # Model Information
        "model": cfgtion.MODEL[cfgtion.TRAIN.MODEL.type],

        # Optimizer Information
        "optim": cfgtion.OPTIM[cfgtion.TRAIN.OPTIM.type],

        # Scheduler Information
        "scheduler": cfgtion.SCHEDULER[cfgtion.TRAIN.SCHEDULER.type] if cfgtion.TRAIN.SCHEDULER.switch is True else None
    }
    return wandb_cfg


def run_mlp_model(trk_dataset, logger, cfgtion, **kwargs):
    assert isinstance(trk_dataset, TEST_DATASET)

    # Unpack KWARGS
    device = kwargs.get("device", __CUDA_DEVICE__)
    is_debug_mode = kwargs.get("is_debug_mode", False)
    assert isinstance(is_debug_mode, bool)
    exp_machine_list = kwargs.get("exp_machine_list")
    assert isinstance(exp_machine_list, list) and len(exp_machine_list) > 0

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

    # Fix Seed
    fix_seed(random_seed=random_seed, logger=logger)

    # Generate Experiment Name
    exp_name = generate_exp_name(cfgtion=cfgtion)

    # Convert "cfgtion" to "wandb_cfg"
    wandb_cfg = convert_cfgtion_to_wandb_cfg(cfgtion=cfgtion)
    wandb_entity, wandb_project = cfgtion.WANDB.entity, cfgtion.WANDB.project

    # === Import WandB if non-debug mode === #
    if is_debug_mode is False:
        # Declare WandB Api
        wandb_api = wandb.Api()

        # Get Runs, to exclude same experiments
        try:
            wandb_runs = wandb_api.runs(
                wandb_entity + "/" + wandb_project, per_page=5000, filters={}
            )

            # Iterate for Runs and Gather Names
            wandb_run_names = [wandb_run.name for wandb_run in wandb_runs]
            raise AssertionError()

        except ValueError:
            pass

        # Initialize WandB
        wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_cfg, reinit=True)
        wandb.run.name = exp_name
        logger.info("WandB Initialized as [{}]...!".format(exp_name))
        time.sleep(1)

    else:
        logger.warn("WandB Not Initialized since Debugging Mode is True...!")
        time.sleep(2)

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

    # # Watch Model via WandB
    # if is_debug_mode is False:
    #     wandb.watch(model)
    #     logger.info("Watching Model via WandB...!")
    #     time.sleep(1)

    # Iterative Training for each epoch
    for epoch in range(num_epochs):
        # if epoch == 26:
        #     print(1234)

        # Train Model
        model, train_loss, train_cls_metric_obj = train(
            train_dataloader=train_dataloader, model=model,
            criterion=criterion, optimizer=optim, epoch=epoch
        )

        # Initialize WandB Logging Dictionary
        wandb_log_dict = {}

        # Compute Current Epoch's Classification Evaluations
        train_recalls = 100 * train_cls_metric_obj.compute_recalls()
        train_precisions = 100 * train_cls_metric_obj.compute_precisions()
        train_f1scores = 100 * train_cls_metric_obj.compute_f1scores()
        train_accuracy = 100 * train_cls_metric_obj.compute_accuracy()
        train_bal_accuracy = 100 * train_cls_metric_obj.compute_balanced_accuracy()

        # Prepare WandB Logging
        wandb_log_dict["Train/Lr"] = optim.param_groups[0]["lr"]
        wandb_log_dict["Train/Loss"] = train_loss
        wandb_log_dict["Train/Acc"] = train_accuracy
        for label_idx in range(train_cls_metric_obj.class_num):
            wandb_log_dict["Train/Recall(#{:02d})".format(label_idx)] = train_recalls[label_idx]
            wandb_log_dict["Train/Precision(#{:02d})".format(label_idx)] = train_precisions[label_idx]
            wandb_log_dict["Train/F1-Score(#{:02d})".format(label_idx)] = train_f1scores[label_idx]

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

                # Prepare WandB Logging
                wandb_log_dict["Val/Loss"] = val_loss
                wandb_log_dict["Val/Acc"] = val_accuracy
                wandb_log_dict["Val/TPR"] = val_TPR
                wandb_log_dict["Val/FPR"] = val_FPR
                for label_idx in range(val_cls_metric_obj.class_num):
                    wandb_log_dict["Val/Recall(#{:02d})".format(label_idx)] = val_recalls[label_idx]
                    wandb_log_dict["Val/Precision(#{:02d})".format(label_idx)] = val_precisions[label_idx]
                    wandb_log_dict["Val/F1-Score(#{:02d})".format(label_idx)] = val_f1scores[label_idx]

        # WandB Train/Val Logging
        wandb_log_commit = False if epoch == num_epochs - 1 else True
        wandb.log(wandb_log_dict, step=epoch, commit=wandb_log_commit)

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

    # Prepare WandB Logging
    wandb_test_log_dict = {
        "Test/Acc": test_accuracy, "Test/TPR": test_TPR, "Test/FPR": test_FPR,
    }
    for label_idx in range(test_cls_metric_obj.class_num):
        wandb_test_log_dict["Test/Recall(#{:02d})".format(label_idx)] = test_recalls[label_idx]
        wandb_test_log_dict["Test/Precision(#{:02d})".format(label_idx)] = test_precisions[label_idx]
        wandb_test_log_dict["Test/F1-Score(#{:02d})".format(label_idx)] = test_f1scores[label_idx]

    # WandB Test Logging
    wandb.log(wandb_test_log_dict)

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
        exp_machine_list=__EXP_MACHINE_LIST__,
        ablation_cfg_filepath=ablation_cfg_filepath
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
            trk_dataset=trk_dataset, logger=_logger, cfgtion=_cfg,
            device=__CUDA_DEVICE__,
            is_debug_mode=__IS_DEBUG_MODE__, exp_machine_list=__EXP_MACHINE_LIST__,
        )

        pass
