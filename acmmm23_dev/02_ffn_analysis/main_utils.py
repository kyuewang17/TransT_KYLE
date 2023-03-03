from __future__ import absolute_import, print_function

import os
import time
import datetime
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from miscs import Timer
from miscs.cls_metric import CLS_METRIC


# ===== Train ===== #
def train(train_dataloader, model, criterion, optimizer, epoch):
    # Assertion
    assert train_dataloader.dataset.labeling_type == "one_hot"

    # Change Model to Train Mode
    model.train()

    # Overlap Criterion
    overlap_criterion = model.overlap_criterion

    # Get Model Output Dimensions (class labels)
    out_dim = model.get_feat_dims("output")

    # Set Various Parameters
    # ------ < loss_sum > : loss sum
    loss_sum = 0

    # Declare tqdm iteration object
    tqdm_iter = tqdm(
        train_dataloader,
        desc="Train (Epoch: {}), Overlap: [{}]".format(epoch, overlap_criterion),
        leave=True, total=len(train_dataloader)
    )

    # Initialize Classification Metric Object
    cls_metric_obj = CLS_METRIC(class_num=out_dim)

    # Mini-Batch Training for Current Epoch
    for batch_idx, (ffn_outputs, gt_bboxes, trk_bboxes, overlaps, labels) in enumerate(tqdm_iter):
        # Change View of "ffn_outputs"
        B, C, Sf = ffn_outputs.shape[0], ffn_outputs.shape[1], ffn_outputs.shape[2]
        S = int(np.sqrt(Sf))

        # Forward Pass Model
        label_preds = model(ffn_outputs.view(B, C, S, S))

        # Compute Loss
        loss = criterion(label_preds, labels.argmax(dim=1))

        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Sum Loss
        loss_sum += loss.item()

        # Get Current Loss Average
        curr_loss_avg = loss_sum / (batch_idx + 1)

        # Get Current Correct, Total, and Prediction
        curr_total = labels.sum(dim=0).cpu().numpy().astype(int)
        if isinstance(criterion.loss_func, torch.nn.CrossEntropyLoss):
            label_preds = F.softmax(label_preds, dim=1)
        _, predicted = label_preds.max(1)
        _, label_indices = labels.max(1)
        predicted = predicted.cpu().numpy()
        label_indices = label_indices.cpu().numpy()
        curr_prediction = np.zeros(out_dim).astype(int)
        for pred in predicted:
            curr_prediction[pred] += 1
        curr_correct = np.zeros(out_dim).astype(int)
        for j, label_idx in enumerate(label_indices):
            if predicted[j] == label_idx:
                curr_correct[label_idx] += 1

        # Add Classification Results (Accumulate)
        cls_metric_obj.add_cls_results(
            totals=curr_total, corrects=curr_correct, predictions=curr_prediction
        )

        # Compute Current Mini-batch F1-Scores and Accuracy
        curr_f1scores = 100 * cls_metric_obj.compute_f1scores()
        curr_accuracy = 100 * cls_metric_obj.compute_accuracy()
        curr_bal_accuracy = 100 * cls_metric_obj.compute_balanced_accuracy()

        # Append to tqdm postfix ordered dictionary
        tqdm_od = OrderedDict({
            "Lr": "{:.6f}".format(optimizer.param_groups[0]["lr"]),
            "LossAvg": "{:.7f}".format(curr_loss_avg),
        })
        for label_idx in range(cls_metric_obj.class_num):
            tqdm_od["F1-Score[{:02d}]".format(label_idx+1)] = \
                "{:.3f}% (# {:,})".format(curr_f1scores[label_idx], cls_metric_obj.totals[label_idx])
        tqdm_od["Accuracy"] = "{:.3f}% (Bal {:.3f}%) (# {:,})".format(
            curr_accuracy, curr_bal_accuracy, sum(cls_metric_obj.totals)
        )
        tqdm_iter.set_postfix(tqdm_od)

    # Compute Average Loss
    loss_avg = loss_sum / len(train_dataloader)

    # Close tqdm iter object
    tqdm_iter.close()

    return model, loss_avg, cls_metric_obj


# ===== Validation ===== #
def validate(val_dataloader, model, criterion, epoch):
    # Assertion
    assert val_dataloader.dataset.labeling_type == "one_hot"

    # Change Model to Evaluation Mode
    model.eval()

    # Overlap Criterion
    overlap_criterion = model.overlap_criterion

    # Get Model Output Dimensions (class labels)
    out_dim = model.get_feat_dims("output")

    # Set Various Parameters
    # ------ < loss_sum > : loss sum
    loss_sum = 0

    # Declare tqdm iteration object
    tqdm_iter = tqdm(
        val_dataloader,
        desc="Validation (Epoch: {}), Overlap: [{}]".format(epoch, overlap_criterion),
        leave=True, total=len(val_dataloader)
    )

    # Initialize Classification Metric Object
    cls_metric_obj = CLS_METRIC(class_num=out_dim)

    # Mini-Batch Training for Current Epoch
    with torch.no_grad():
        for batch_idx, (ffn_outputs, gt_bboxes, trk_bboxes, overlaps, labels) in enumerate(tqdm_iter):
            # Change View of "ffn_outputs"
            B, C, Sf = ffn_outputs.shape[0], ffn_outputs.shape[1], ffn_outputs.shape[2]
            S = int(np.sqrt(Sf))

            # Forward Pass Model
            label_preds = model(ffn_outputs.view(B, C, S, S))

            # Compute Loss
            loss = criterion(label_preds, labels.argmax(dim=1))

            # Sum Loss
            loss_sum += loss.item()

            # Get Current Loss Average
            curr_loss_avg = loss_sum / (batch_idx + 1)

            # Get Current Correct, Total, and Prediction
            curr_total = labels.sum(dim=0).cpu().numpy().astype(int)
            _, predicted = label_preds.max(1)
            _, label_indices = labels.max(1)
            predicted = predicted.cpu().numpy()
            label_indices = label_indices.cpu().numpy()
            curr_prediction = np.zeros(out_dim).astype(int)
            for pred in predicted:
                curr_prediction[pred] += 1
            curr_correct = np.zeros(out_dim).astype(int)
            for j, label_idx in enumerate(label_indices):
                if predicted[j] == label_idx:
                    curr_correct[label_idx] += 1

            # Add Classification Results (Accumulate)
            cls_metric_obj.add_cls_results(
                totals=curr_total, corrects=curr_correct, predictions=curr_prediction
            )

            # Compute Current Mini-batch F1-Scores and Accuracy
            curr_f1scores = 100 * cls_metric_obj.compute_f1scores()
            curr_accuracy = 100 * cls_metric_obj.compute_accuracy()
            curr_bal_accuracy = 100 * cls_metric_obj.compute_balanced_accuracy()

            # Append to tqdm postfix ordered dictionary
            tqdm_od = OrderedDict({
                "LossAvg": "{:.7f}".format(curr_loss_avg),
            })
            for label_idx in range(cls_metric_obj.class_num):
                tqdm_od["F1-Score[{:02d}]".format(label_idx + 1)] = \
                    "{:.3f}% (# {:,})".format(curr_f1scores[label_idx], cls_metric_obj.totals[label_idx])
            tqdm_od["Accuracy"] = "{:.3f}% (Bal {:.3f}%) (# {:,})".format(
                curr_accuracy, curr_bal_accuracy, sum(cls_metric_obj.totals)
            )
            tqdm_iter.set_postfix(tqdm_od)

    # Compute Average Loss
    loss_avg = loss_sum / len(val_dataloader)

    # Close tqdm iter object
    tqdm_iter.close()

    return loss_avg, cls_metric_obj


# ===== Test ===== #
def test(test_dataloader, model):
    # Assertion
    assert test_dataloader.dataset.labeling_type == "one_hot"

    # Change Model to Evaluation Mode
    model.eval()

    # Overlap Criterion
    overlap_criterion = model.overlap_criterion

    # Get Model Output Dimensions (class labels)
    out_dim = model.get_feat_dims("output")

    # Declare tqdm iteration object
    tqdm_iter = tqdm(
        test_dataloader,
        desc="Test, Overlap: [{}]".format(overlap_criterion),
        leave=True, total=len(test_dataloader)
    )

    # Initialize Classification Metric Object
    cls_metric_obj = CLS_METRIC(class_num=out_dim)

    # Mini-Batch Training for Current Epoch
    with torch.no_grad():
        for batch_idx, (ffn_outputs, gt_bboxes, trk_bboxes, overlaps, labels) in enumerate(tqdm_iter):
            # Change View of "ffn_outputs"
            B, C, Sf = ffn_outputs.shape[0], ffn_outputs.shape[1], ffn_outputs.shape[2]
            S = int(np.sqrt(Sf))

            # Forward Pass Model
            label_preds = model(ffn_outputs.view(B, C, S, S))

            # Get Current Correct, Total, and Prediction
            curr_total = labels.sum(dim=0).cpu().numpy().astype(int)
            _, predicted = label_preds.max(1)
            _, label_indices = labels.max(1)
            predicted = predicted.cpu().numpy()
            label_indices = label_indices.cpu().numpy()
            curr_prediction = np.zeros(out_dim).astype(int)
            for pred in predicted:
                curr_prediction[pred] += 1
            curr_correct = np.zeros(out_dim).astype(int)
            for j, label_idx in enumerate(label_indices):
                if predicted[j] == label_idx:
                    curr_correct[label_idx] += 1

            # Add Classification Results (Accumulate)
            cls_metric_obj.add_cls_results(
                totals=curr_total, corrects=curr_correct, predictions=curr_prediction
            )

            # Compute Current Mini-batch F1-Scores and Accuracy
            curr_f1scores = 100 * cls_metric_obj.compute_f1scores()
            curr_accuracy = 100 * cls_metric_obj.compute_accuracy()
            curr_bal_accuracy = 100 * cls_metric_obj.compute_balanced_accuracy()

            # Append to tqdm postfix ordered dictionary
            tqdm_od = OrderedDict({})
            for label_idx in range(cls_metric_obj.class_num):
                tqdm_od["F1-Score[{:02d}]".format(label_idx + 1)] = \
                    "{:.3f}% (# {:,})".format(curr_f1scores[label_idx], cls_metric_obj.totals[label_idx])
            tqdm_od["Accuracy"] = "{:.3f}% (Bal {:.3f}%) (# {:,})".format(
                curr_accuracy, curr_bal_accuracy, sum(cls_metric_obj.totals)
            )
            tqdm_iter.set_postfix(tqdm_od)

    # Close tqdm iter object
    tqdm_iter.close()

    return cls_metric_obj


if __name__ == "__main__":
    pass
