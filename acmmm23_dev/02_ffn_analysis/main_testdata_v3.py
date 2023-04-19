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
import torch.utils.data.sampler as t_sampler
import torch.multiprocessing as t_mp
from torch.multiprocessing import set_start_method
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

try:
    set_start_method("spawn")
except RuntimeError:
    pass


# Set Important Paths
__PROJECT_MASTER_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE"

# Set Benchmark Dataset
__BENCHMARK_DATASET__ = "OTB100"
# __BENCHMARK_DATASET__ = "UAV123"

# Experiment Machine Dictionary
# "do not" put <under-bar> ("_") to the machine host name (key)
__EXP_MACHINE_DICT__ = {
    "PIL-kyle": ["cpu", "cuda:0"],
    "carpenters1": ["cpu", "cuda:0", "cuda:1"],
    "carpenters2": ["cpu", "cuda:0", "cuda:1"],
}

# Memory Cut-off Percentage
__MEMORY_CUTOFF_PERCENT__ = 95

# Debugging Mode
__IS_DEBUG_MODE__ = False


def cfg_loader(logger, cfg_filepath, **kwargs):
    # Get Current Machine Information (Hostname, GPUs, etc.)
    curr_hostname = socket.gethostname()
    gpu_devices = [*["cpu"], *["cuda:{}".format(j) for j in range(torch.cuda.device_count())]]

    # Unpack KWARGS and Do Assertion Test
    exp_machine_dict = kwargs.get("exp_machine_dict")
    assert isinstance(exp_machine_dict, dict) and len(exp_machine_dict) > 0
    assert curr_hostname in list(exp_machine_dict.keys())
    for host, chips in exp_machine_dict.items():
        assert isinstance(chips, list) and len(chips) > 0
        if host == curr_hostname:
            assert set(chips).issubset(set(gpu_devices))

    # Permute Possible Experiments
    counter = 0
    exp_instances, curr_host_exp_indices = [], []
    for idx, (host, chips) in enumerate(exp_machine_dict.items()):
        for chip in chips:
            if host == curr_hostname:
                curr_host_exp_indices.append(counter)
            exp_name = "{}_{}".format(host, chip)
            exp_instances.append(exp_name)
            counter += 1

    ablation_cfg_filepath = kwargs.get("ablation_cfg_filepath")
    if len(exp_instances) > 1:
        assert ablation_cfg_filepath is not None
    else:
        assert ablation_cfg_filepath is None
    ablation_cfg_shuffle_seed = kwargs.get("ablation_cfg_shuffle_seed", 333)
    assert isinstance(ablation_cfg_shuffle_seed, int) and ablation_cfg_shuffle_seed > 0

    # Load Configurations
    assert os.path.isfile(cfg_filepath)
    logger.info("Loading Configurations from: {}".format(cfg_filepath))
    cfg.merge_from_file(cfg_filename=cfg_filepath)
    time.sleep(1)
    logger.info("\n=== Configuration File Logging ===\n{}\n==================================".format(cfg))
    time.sleep(1)
    if len(exp_instances) == 1:
        if exp_instances[0].__contains__("cpu"):
            cfg.device = "cpu"
        elif exp_instances[0].__contains__("cuda:"):
            cfg.device = exp_instances[0].split("_")[-1]
        else:
            raise AssertionError()
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
        cfgs_split = split_cfgs(cfgs=cfgs, experiments=exp_instances)

        # Select Experiment Instances of Current
        curr_host_cfgs_split = {}
        for jj, (exp_inst_name, _cfgs) in enumerate(cfgs_split.items()):
            if jj in curr_host_exp_indices:
                curr_host_cfgs_split[exp_inst_name] = _cfgs

        return curr_host_cfgs_split


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


def split_cfgs(cfgs, experiments):
    chunks = len(cfgs) // len(experiments)
    cfgs_split = {}
    for jj, experiment in enumerate(experiments):
        curr_cfgs = cfgs[jj * chunks:(jj + 1) * chunks]
        if experiment.__contains__("cpu"):
            for _cfg in curr_cfgs:
                _cfg.device = "cpu"
        elif experiment.__contains__("cuda:"):
            gpu_device = experiment.split("_")[-1]
            for _cfg in curr_cfgs:
                _cfg.device = gpu_device
        else:
            raise AssertionError()
        if jj < len(experiments) - 1:
            cfgs_split[experiment] = curr_cfgs
        else:
            cfgs_split[experiment] = cfgs[jj * chunks:]
    return cfgs_split


def generate_exp_name(cfgtion, exp_instance):
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
        exp_instance, get_current_datetime_str("(%y-%m-%d)-(%H-%M-%S)")
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


def load_trk_data_obj(cfgtion):
    return BENCHMARK_DATA_OBJ(
        root_path=os.path.join(cfgtion.DATA.root_path, cfgtion.DATA.benchmark),
        benchmark=cfgtion.DATA.benchmark,
        overlap_criterion=cfgtion.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_criterion,
        overlap_thresholds=cfgtion.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_thresholds,
        labeling_type=cfgtion.DATA.OPTS.BENCHMARK_DATA_OBJ.labeling_type,
        random_seed=cfgtion.DATA.OPTS.BENCHMARK_DATA_OBJ.random_seed,
        device=cfgtion.device
    )


def mp_run_mlp_model(return_dict, trk_data_obj, **kwargs):
    assert isinstance(trk_data_obj, BENCHMARK_DATA_OBJ)

    # Unpack KWARGS
    cfgtions = kwargs.get("cfgtions")
    assert isinstance(cfgtions, list) and len(cfgtions) > 0
    exp_instance = kwargs.get("exp_instance")
    logger = kwargs.get("logger")
    proc_idx = kwargs.get("proc_idx")
    assert isinstance(proc_idx, int) and proc_idx >= 0
    Nprocs = kwargs.get("Nprocs")
    assert isinstance(Nprocs, int) and Nprocs > 0

    # Log Process Start
    if logger is not None:
        logger.info("Starting Process ({}/{})".format(proc_idx+1, Nprocs))

    # Iterate for Configurations
    for _cfg_idx, _cfg in enumerate(cfgtions):
        trk_data_obj.reload( # noqa
            overlap_criterion=_cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_criterion,
            overlap_thresholds=_cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_thresholds,
            labeling_type=_cfg.DATA.OPTS.BENCHMARK_DATA_OBJ.labeling_type,
        )

        # Wrap with Test Loader
        trk_dataset = TEST_DATASET(data_obj=trk_data_obj, init_mode="data_obj")

        # Run MLP Model Code
        run_mlp_model(
            trk_dataset=trk_dataset, cfgtion=_cfg,
            exp_instance=exp_instance, device=_cfg.device,
        )

        # Flush CUDA Memory
        torch.cuda.empty_cache()

        # Sleep for 20 seconds (wait enough for wandb process to finish)
        time.sleep(20)

    # Log Process End
    if logger is not None:
        logger.info("Ending Process ({}/{})".format(proc_idx + 1, Nprocs))
    time.sleep(3)


def run_mlp_model(trk_dataset, cfgtion, **kwargs):
    assert isinstance(trk_dataset, TEST_DATASET)

    # Unpack KWARGS
    logger = kwargs.get("logger")
    device = kwargs.get("device")
    assert device is not None
    is_debug_mode = kwargs.get("is_debug_mode", False)
    assert isinstance(is_debug_mode, bool)
    exp_instance = kwargs.get("exp_instance")

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
    assert isinstance(val_epoch_interval, int) and 0 < val_epoch_interval

    # Fix Seed
    fix_seed(random_seed=random_seed, logger=logger)

    # Generate Experiment Name
    exp_name = generate_exp_name(cfgtion=cfgtion, exp_instance=exp_instance)

    # Convert "cfgtion" to "wandb_cfg"
    wandb_cfg = convert_cfgtion_to_wandb_cfg(cfgtion=cfgtion)
    wandb_entity, wandb_project = cfgtion.WANDB.entity, cfgtion.WANDB.project

    # === Import WandB if non-debug mode === #
    if is_debug_mode is False:
        # # Declare WandB Api
        # wandb_api = wandb.Api()

        # todo: sophisticate this code...!
        # # Get Runs, to exclude same experiments
        # try:
        #     wandb_runs = wandb_api.runs(
        #         wandb_entity + "/" + wandb_project, per_page=5000, filters={}
        #     )
        #
        #     # Iterate for Runs and Gather Names
        #     wandb_run_names = [wandb_run.name for wandb_run in wandb_runs]
        #     wandb_exp_names = [wandb_run_name.split("__", 2)[-1] for wandb_run_name in wandb_run_names]
        #     wandb_exp_names = np.array(wandb_exp_names)
        #
        #     # Find if Current Experiment Exist in the Project
        #     wandb_indices = np.where(wandb_exp_names == exp_name.split("__", 2)[-1])[0]
        #     if len(wandb_indices) > 0:
        #         logger.warn("Skipping Experiment: {}".format(exp_name))
        #         time.sleep(2)
        #         return
        #
        # except ValueError:
        #     pass

        # Initialize WandB
        wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_cfg, reinit=True)
        wandb.run.name = exp_name
        if logger is not None:
            logger.info("WandB Initialized as [{}]...!".format(exp_name))
        time.sleep(1)

    else:
        if logger is not None:
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
    _weights = 1. / train_dataset.compute_label_sums()
    s_weights = np.array(
        [_weights[_data["label"].argmax()] for _data in train_dataset.data]
    )
    s_weights = torch.from_numpy(s_weights)
    train_sampler = t_sampler.WeightedRandomSampler(
        s_weights.type("torch.DoubleTensor"), len(s_weights)
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    if logger is not None:
        logger.info("\nTraining DataLoader Loaded Completely...! - # of Samples: [{:,}]".format(len(train_dataset)))
    if is_validation:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        if logger is not None:
            logger.info("Validation DataLoader Loaded Completely...! - # of Samples: [{:,}]".format(len(val_dataset)))
    else:
        val_dataloader = None
        if logger is not None:
            logger.info("Validation DataLoader Skipped...!")
        time.sleep(0.5)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if logger is not None:
        logger.info("Testing DataLoader Loaded Completely...! - # of Samples: [{:,}]".format(len(test_dataset)))
    time.sleep(0.5)

    # Load Model
    model = load_model(cfg=cfgtion)
    model.to(device=device)
    if logger is not None:
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
    if logger is not None:
        logger.info("Criterion Loaded...!")
    time.sleep(0.5)

    # Define Optimizer
    optim = init_optim(
        optim_type=optim_type, model_params=model.parameters(),
        base_lr=base_lr, weight_decay=weight_decay, momentum=momentum,
    )
    if logger is not None:
        logger.info("Optimizer Loaded...!")
    time.sleep(0.5)

    # Define Scheduler
    if scheduler_type is None:
        scheduler = None
        if logger is not None:
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
        if logger is not None:
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
        if is_debug_mode is False:
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
    if is_debug_mode is False:
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
    cfgs_dict = cfg_loader(
        logger=_logger, cfg_filepath=cfg_filepath,
        exp_machine_dict=__EXP_MACHINE_DICT__, ablation_cfg_filepath=ablation_cfg_filepath
    )

    # Load "trk_data_obj" for the very first configurations
    trk_data_obj = load_trk_data_obj(cfgtion=list(cfgs_dict.values())[0][0])

    # Multiprocess
    procs = []
    t_mp_manager = t_mp.Manager()
    return_dict = t_mp_manager.dict()
    for proc_idx, (exp_inst, cfgs) in enumerate(cfgs_dict.items()):

        # Gather Process KWARGS
        proc_kwargs = {
            "exp_instance": exp_inst, "cfgtions": cfgs,
            "logger": _logger, "proc_idx": proc_idx, "Nprocs": len(cfgs_dict),
        }

        # Get Process
        # fixme: too long (not function "mp_run_mlp_model")
        #       --> "trk_data_obj" initialization inside function "mp_run_mlp_model" ??
        proc = t_mp.Process(
            target=mp_run_mlp_model,
            args=(return_dict, trk_data_obj), kwargs=proc_kwargs
        )

        # Append Process and Start
        procs.append(proc)
        proc.start()

    # Join Processes
    for proc in procs:
        proc.join()
