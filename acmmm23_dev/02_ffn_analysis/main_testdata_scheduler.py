from __future__ import absolute_import, print_function

import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from miscs import set_logger, fix_seed
from data.trk_data_obj import BENCHMARK_DATA_OBJ
from data.test_loader import TEST_DATASET
from model.bones import OVERLAP_CLASSIFIER
from loss.zoo import CLS_LOSS
from optim.zoo import init_optim
from main_utils import train, validate, test


# Set Important Paths
__PROJECT_MASTER_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE"
__FFN_DATA_ROOT_PATH__ = \
    os.path.join(os.path.join(__PROJECT_MASTER_PATH__, "acmmm23_dev"), "ffn_data")

__BENCHMARK_DATASET__ = "OTB100"
# __BENCHMARK_DATASET__ = "UAV123"

# CUDA Device Configuration
__CUDA_DEVICE__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Memory Cut-off Percentage
__MEMORY_CUTOFF_PERCENT__ = 95


def run_mlp_model(trk_dataset, logger, **kwargs):
    assert isinstance(trk_dataset, TEST_DATASET)

    # Unpack KWARGS
    batch_size = kwargs.get("batch_size", 128)
    assert isinstance(batch_size, int) and batch_size > 0
    loss_type, optim_type = kwargs.get("loss_type"), kwargs.get("optim_type")
    weight_decay = kwargs.get("weight_decay", 0)
    base_lr, momentum = kwargs.get("base_lr"), kwargs.get("momentum")
    random_seed = kwargs.get("random_seed")
    assert isinstance(random_seed, int) and random_seed > 0
    scheduler_type = kwargs.get("scheduler_type")
    num_epochs = kwargs.get("num_epochs")
    if num_epochs is None:
        raise AssertionError()
    else:
        assert isinstance(num_epochs, int) and num_epochs > 0
    val_epoch_interval = kwargs.get("val_epoch_interval")
    assert isinstance(val_epoch_interval, int) and 0 < val_epoch_interval < num_epochs
    device = kwargs.get("device", __CUDA_DEVICE__)

    # Fix Seed
    fix_seed(random_seed=random_seed, logger=logger)

    # Split Trk Dataset
    train_dataset, _dataset = trk_dataset.split_obj(ratio=0.8, split_seed=random_seed)
    val_dataset, test_dataset = _dataset.split_obj(ratio=0.5, split_seed=random_seed)

    # Wrap Dataset with PyTorch DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logger.info("\nTraining DataLoader Loaded Completely...! - # of Samples: [{:,}]".format(len(train_dataset)))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    logger.info("Validation DataLoader Loaded Completely...! - # of Samples: [{:,}]".format(len(val_dataset)))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Testing DataLoader Loaded Completely...! - # of Samples: [{:,}]".format(len(test_dataset)))
    time.sleep(0.5)

    # Define Model
    model = OVERLAP_CLASSIFIER(
        cls_loss=loss_type, overlap_criterion=train_dataset.overlap_criterion,

        # dimensions=[256, 32, 10, 2],
        dimensions=[768, 100, 32, 2],
        layers=["fc", "fc", "fc"],
        hidden_activations=["ReLU", "ReLU"],
        batchnorm_layers=[True, True],
        dropout_probs=[0.2, 0.2],
        final_activation="softmax",

        init_dist="normal"
    )
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

        base_lr=base_lr,
        weight_decay=weight_decay, momentum=momentum,
    )
    logger.info("Optimizer Loaded...!")
    time.sleep(0.5)

    # Define Scheduler
    if scheduler_type is None:
        scheduler = None
        logger.info("Scheduler not Defined...!")
    else:
        if scheduler_type == "CosineAnnealingLr":
            cos_min_lr = kwargs.get("cos_min_lr", 0)
            assert isinstance(cos_min_lr, float) and cos_min_lr < base_lr
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=optim, T_max=num_epochs, eta_min=cos_min_lr
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
    # Set FFN Data Path
    ffn_data_path = os.path.join(__FFN_DATA_ROOT_PATH__, __BENCHMARK_DATASET__)

    # Logger
    _logger = set_logger()

    # Load Benchmark Data Object
    trk_data_obj = BENCHMARK_DATA_OBJ(
        root_path=ffn_data_path, benchmark=__BENCHMARK_DATASET__, logger=_logger,

        overlap_criterion="iou", overlap_thresholds=[0.5],
        labeling_type="one_hot",

        random_seed=1234,
    )

    # Wrap with Test Loader
    trk_dataset = TEST_DATASET(data_obj=trk_data_obj, logger=_logger, init_mode="data_obj")

    # Run MLP Model Code
    run_mlp_model(
        trk_dataset=trk_dataset,
        logger=_logger,

        random_seed=1111,

        batch_size=512, num_epochs=50,
        val_epoch_interval=5,

        loss_type="WCE", optim_type="SGD", weight_decay=0,
        base_lr=0.25, momentum=0.9,

        scheduler_type="CosineAnnealingLr", cos_min_lr=0.05,

        device=__CUDA_DEVICE__,
    )


    pass
