from __future__ import absolute_import, print_function

import os
import psutil
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

try:
    from .utils import set_logger, load_npy_cvt_torch
    from .trk_data_obj import BENCHMARK_DATA_OBJ
except ImportError:
    from utils import set_logger, load_npy_cvt_torch
    from trk_data_obj import BENCHMARK_DATA_OBJ


# Set Important Paths
__PROJECT_MASTER_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE"
__FFN_DATA_ROOT_PATH__ = \
    os.path.join(os.path.join(__PROJECT_MASTER_PATH__, "acmmm23_dev"), "ffn_data")

__BENCHMARK_DATASET__ = "OTB100"

# CUDA Device Configuration
__CUDA_DEVICE__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Memory Cut-off Percentage
__MEMORY_CUTOFF_PERCENT__ = 95


class TEST_DATASET(Dataset):
    # def __init__(self, data_obj, logger, **kwargs):
    def __init__(self, logger, init_mode, **kwargs):

        # Set Basic Variables
        self.logger = logger

        # Assertion of Init Mode
        assert init_mode in ["data_obj", "data_list"]

        if init_mode == "data_obj":
            # Get Data Object
            data_obj = kwargs.get("data_obj")
            assert isinstance(data_obj, BENCHMARK_DATA_OBJ)

            # Unpack KWARGS
            device = kwargs.get("device", data_obj.device)
            self.device = device
            overlap_criterion = kwargs.get("overlap_criterion", data_obj.overlap_criterion)
            assert overlap_criterion in ["iou", "diou"]
            self.overlap_criterion = overlap_criterion
            overlap_thresholds = kwargs.get("overlap_thresholds", data_obj.overlap_thresholds)
            assert isinstance(overlap_thresholds, (list, tuple, np.ndarray))
            if isinstance(overlap_thresholds, tuple):
                overlap_thresholds = list(overlap_thresholds)
            elif isinstance(overlap_thresholds, np.ndarray):
                overlap_thresholds = overlap_thresholds.tolist()
            self.overlap_thresholds = sorted(overlap_thresholds)
            labeling_type = kwargs.get("labeling_type", data_obj.labeling_type)
            assert labeling_type in ["one_hot", "scalar"]
            self.labeling_type = labeling_type

            # Get Dataset Name
            self.dataset_name = data_obj.benchmark

            # Get Data for "data_obj"
            self.data = data_obj[:]

        else:
            # Unpack KWARGS
            device = kwargs.get("device")
            assert device is not None
            self.device = device
            overlap_criterion = kwargs.get("overlap_criterion")
            assert overlap_criterion in ["iou", "diou"]
            self.overlap_criterion = overlap_criterion
            overlap_thresholds = kwargs.get("overlap_thresholds")
            assert isinstance(overlap_thresholds, (list, tuple, np.ndarray))
            if isinstance(overlap_thresholds, tuple):
                overlap_thresholds = list(overlap_thresholds)
            elif isinstance(overlap_thresholds, np.ndarray):
                overlap_thresholds = overlap_thresholds.tolist()
            self.overlap_thresholds = sorted(overlap_thresholds)
            labeling_type = kwargs.get("labeling_type")
            assert labeling_type in ["one_hot", "scalar"]
            self.labeling_type = labeling_type

            # Dataset Name
            dataset_name = kwargs.get("dataset_name")
            assert dataset_name is not None
            self.dataset_name = dataset_name

            # Data
            data = kwargs.get("data")
            assert isinstance(data, list) and len(data) > 0
            self.data = data

        # Set Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        raise NotImplementedError()

    def __getitem__(self, item):
        # Get Current "data dictionary" of index
        curr_data_dict = self.data[item]

        # Load "ffn_outputs"
        # ffn_outputs = load_npy_cvt_torch(curr_data_dict["ffn_filepath"], device=self.device)
        ffn_arr = np.array(curr_data_dict["ffn_mmap"])
        ffn_outputs = torch.from_numpy(ffn_arr).to(device=self.device)
        ffn_outputs = ffn_outputs.permute(1, 0)

        # Wrap "bboxes" to Tensor
        gt_bbox = torch.from_numpy(curr_data_dict["gt_bbox"]).to(
            device=self.device, dtype=torch.float32
        )
        trk_bbox = torch.from_numpy(curr_data_dict["trk_bbox"]).to(
            device=self.device, dtype=torch.float32
        )

        # Wrap "overlap" to Tensor
        overlap = torch.from_numpy(np.array(curr_data_dict["overlap"], dtype=np.float32)).to(
            device=self.device
        )

        # Wrap "label" to Tensor
        label = torch.from_numpy(curr_data_dict["label"]).to(
            device=self.device, dtype=torch.int64
        )

        return ffn_outputs, gt_bbox, trk_bbox, overlap, label

    def split_obj(self, ratio, split_seed=None):
        assert isinstance(ratio, float) and 0 < ratio < 1

        # Set Split Samples
        split_samples = int(len(self) * ratio)
        whole_indices = list(range(len(self)))
        assert split_samples > 0

        # Choose Split Indices
        if split_seed is not None:
            np.random.seed(split_seed)
        split_choice_indices = \
            np.random.choice(np.array(whole_indices), split_samples, replace=False)
        post_split_choice_indices = \
            np.array(list(set(whole_indices) - set(split_choice_indices)))

        # Select Data
        split_data = [self.data[jj] for jj in split_choice_indices]
        post_split_data = [self.data[jj] for jj in post_split_choice_indices]

        # Return Split Objects
        return self._clone(split_data), self._clone(post_split_data)

    def _clone(self, new_data):
        assert new_data is not None and isinstance(new_data, list)
        assert len(new_data) > 0
        new_obj = TEST_DATASET(
            logger=self.logger, init_mode="data_list",
            device=self.device, data=new_data, dataset_name=self.dataset_name,
            overlap_criterion=self.overlap_criterion, overlap_thresholds=self.overlap_thresholds,
            labeling_type=self.labeling_type,
        )
        return new_obj

    def compute_label_sums(self, cvt_to_ratio=False):
        labels = self.get_labels()
        label_sums = np.array(labels).sum(axis=0)
        if cvt_to_ratio:
            return label_sums / label_sums.sum()
        else:
            return label_sums

    def get_labels(self):
        if self.labeling_type == "scalar":
            _c_self = self._clone(self.data)
            _c_self.convert_labeling_type(labeling_type="one_hot")
            data_dict_list = _c_self.data
        else:
            data_dict_list = self.data

        # Gather entire labels iteratively
        labels = []
        for data_dict in data_dict_list:
            labels.append(data_dict["label"])

        return labels

    def convert_labeling_type(self, labeling_type):
        assert labeling_type in ["one_hot", "scalar"]
        if self.labeling_type == labeling_type:
            self.logger.warning("Labeling Type is Same... Conversion Skipped!")
            return

        # Iterate for "self.data"
        for data in self.data:
            # Get Label
            label = deepcopy(data["label"])

            # One-Hot -> Scalar
            if labeling_type == "scalar":
                data["label"] = np.argmax(label)

            # Scalar -> One-Hot
            elif labeling_type == "one_hot":
                new_label = np.zeros(len(self.overlap_thresholds)+1).astype(int)
                new_label[label] += 1
                data["label"] = new_label

        # Complete Message
        self.logger.info(
            "Label Conversion from [{}] to [{}] completed...!".format(
                self.labeling_type, labeling_type
            )
        )

        # Set Labeling Type Parameter
        self.labeling_type = labeling_type

    def relabel(self, overlap_thresholds):
        assert isinstance(overlap_thresholds, (list, tuple, np.ndarray))
        if isinstance(overlap_thresholds, tuple):
            overlap_thresholds = list(overlap_thresholds)
        elif isinstance(overlap_thresholds, np.ndarray):
            overlap_thresholds = overlap_thresholds.tolist()
        overlap_thresholds = sorted(overlap_thresholds)

        # Iterate for "self.data"
        for data in self.data:
            # Get Overlap Value
            overlap = data["overlap"]

            # Label according to overlap threshold, w.r.t. labeling type
            label = np.zeros(len(overlap_thresholds)+1).astype(int)

            # Compute Label
            for i_idx, overlap_thresh in enumerate(overlap_thresholds):
                if overlap < overlap_thresh:
                    np.add.at(label, [i_idx], 1)
                detect_true_label = np.where(label == 1)[0]
                if len(detect_true_label) > 0:
                    break
            if label.max() == 0:
                label[-1] = 1
            if self.labeling_type == "scalar":
                label = label.argmax()

            # Re-label
            data["label"] = label

            # Change "overlap_thresholds"
            data["miscs"]["overlap_thresholds"] = overlap_thresholds

        # Convert overlap thresholds
        self.overlap_thresholds = overlap_thresholds


if __name__ == "__main__":
    # Logger
    _logger = set_logger()

    # Set FFN Data Path
    ffn_data_path = os.path.join(__FFN_DATA_ROOT_PATH__, __BENCHMARK_DATASET__)

    # FFN Object
    FFN_OBJ = BENCHMARK_DATA_OBJ(
        root_path=ffn_data_path, benchmark=__BENCHMARK_DATASET__, logger=_logger,

        overlap_criterion="iou", overlap_thresholds=[0.5],
        labeling_type="one_hot",

        random_seed=1234, is_debug_mode=True,
    )

    # Wrap with Loader
    test_dataset = TEST_DATASET(
        data_obj=FFN_OBJ, logger=_logger, init_mode="data_obj"
    )

    # aa, bb = test_dataset.split_obj(ratio=0.8)

    aaa = test_dataset[33]


    pass
