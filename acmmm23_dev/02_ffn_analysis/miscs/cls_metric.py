from __future__ import absolute_import, print_function

import numpy as np


class CLS_METRIC(object):
    def __init__(self, class_num):
        assert isinstance(class_num, int) and class_num > 0
        self.class_num = class_num

        # Totals
        self.totals = np.zeros(class_num).astype(int)

        # Corrects
        self.corrects = np.zeros(class_num).astype(int)

        # Predictions
        self.predictions = np.zeros(class_num).astype(int)

    def __len__(self):
        return sum(self.totals)

    # (Label-wise OvR) Actual Positives, predicted as Positives
    def TP(self):
        return self.corrects

    # (Label-wise OvR) Actual Positives, Predicted as Negatives
    def FN(self):
        return self.totals - self.corrects

    # (Label-wise OvR) Actual Negatives, Predicted as Negatives
    def TN(self):
        return sum(self.corrects) - self.corrects

    # (Label-wise OvR) Actual Negatives, Predicted as Positives
    def FP(self):
        return self.predictions - self.corrects

    # (Label-wise OvR) Compute Actual Positives
    def P(self):
        return self.totals

    # (Label-wise OvR) Compute Actual Negatives
    def N(self):
        return sum(self.totals) - self.totals

    # Compute True/False Positive/Negative Info
    def compute_tfpn_info(self):
        return {
            "TP": self.TP(), "FN": self.FN(),
            "FP": self.FP(), "TN": self.TN(),
        }

    # (Label-wise OvR) Compute Recalls
    def compute_recalls(self):
        return np.divide(self.corrects, self.totals)

    # (Label-wise OvR) Compute Precisions
    def compute_precisions(self):
        return np.divide(self.corrects, self.predictions)

    # (Label-wise OvR) Compute F1-Scores
    def compute_f1scores(self, **kwargs):
        recalls, precisions = kwargs.get("recalls"), kwargs.get("precisions")
        if recalls is None:
            recalls = self.compute_recalls()
        if precisions is None:
            precisions = self.compute_precisions()
        _sum = recalls + precisions
        _mul = np.multiply(recalls, precisions)
        return 2 * np.divide(_mul, _sum)

    # Compute Accuracy
    def compute_accuracy(self):
        return self.corrects.sum() / self.totals.sum()

    # Compute True Positive Rate (TPR)
    def TPR(self):
        return np.divide(self.corrects.sum(), self.totals.sum())

    # Compute False Negative Rate (FNR)
    def FNR(self):
        return 1 - self.TPR()

    # Compute True Negative Rate (TNR)
    def TNR(self):
        true_negatives = self.TN().sum()
        false_positives = self.FP().sum()
        return np.divide(true_negatives, (false_positives + true_negatives))

    # Compute False Positive Rate (FPR)
    def FPR(self):
        return 1 - self.TNR()

    # Compute Balanced Accuracy (BA)
    def compute_balanced_accuracy(self):
        return (self.TPR() + self.TNR()) / 2.0

    def add_cls_results(self, totals=None, corrects=None, predictions=None):
        if totals is not None:
            self.add_totals(totals=totals)
        if corrects is not None:
            self.add_corrects(corrects=corrects)
        if predictions is not None:
            self.add_predictions(predictions=predictions)

    def add_totals(self, totals):
        assert isinstance(totals, np.ndarray)
        self.totals += totals

    def add_corrects(self, corrects):
        assert isinstance(corrects, np.ndarray)
        self.corrects += corrects

    def add_predictions(self, predictions):
        assert isinstance(predictions, np.ndarray)
        self.predictions += predictions


if __name__ == "__main__":
    totals = [3000, 3000, 400]
    corrects = [2500, 2500, 100]
    predictions = [2700, 2600, 1100]

    # Init Object
    cls_metric_obj = CLS_METRIC(class_num=len(totals))
    cls_metric_obj.add_cls_results(
        totals=np.array(totals), corrects=np.array(corrects), predictions=np.array(predictions),
    )

    # Test
    pass
