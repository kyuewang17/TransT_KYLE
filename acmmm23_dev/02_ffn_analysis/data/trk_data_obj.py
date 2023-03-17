from __future__ import absolute_import, print_function

import os
import sys
import logging
import psutil
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.multiprocessing as t_mp
from torch.multiprocessing import set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass

try:
    from .utils import set_logger
    from .utils.rect_metrics import rect_iou, rect_diou
except ImportError:
    from utils import set_logger
    from utils.rect_metrics import rect_iou, rect_diou


# Set Important Paths
__PROJECT_MASTER_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE"
__FFN_DATA_ROOT_PATH__ = \
    os.path.join(os.path.join(__PROJECT_MASTER_PATH__, "acmmm23_dev"), "ffn_data")

__BENCHMARK_DATASET__ = "OTB100"

# CUDA Device Configuration
__CUDA_DEVICE__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Memory Cut-off Percentage
__MEMORY_CUTOFF_PERCENT__ = 95


class BENCHMARK_DATA_OBJ(object):
    def __init__(self, root_path, benchmark, logger, **kwargs):
        assert os.path.isdir(root_path)

        # Set Basic Variables
        self.root_path, self.benchmark = root_path, benchmark
        self.logger = logger

        # Unpack "root_path" subdirectories to get total video names
        video_names = sorted(os.listdir(root_path))

        # Unpack KWARGS
        video_indices = kwargs.get("video_indices")
        if video_indices is not None:
            assert isinstance(video_indices, (list, tuple, np.ndarray))
            assert len(video_indices) <= len(video_names) and max(video_indices) < len(video_names)
            if isinstance(video_indices, tuple):
                video_indices = list(video_indices)
            elif isinstance(video_indices, np.ndarray):
                video_indices = video_indices.tolist()
            video_indices = sorted(video_indices)
        else:
            video_indices = list(range(len(video_names)))

        overlap_criterion = kwargs.get("overlap_criterion")
        assert overlap_criterion in ["iou", "giou", "diou", "ciou"]
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
        random_seed = kwargs.get("random_seed")
        if random_seed is not None:
            assert isinstance(random_seed, int) and random_seed >= 0
        self.random_seed = random_seed

        device = kwargs.get("device", __CUDA_DEVICE__)
        self.device = device

        # load_with_mp = kwargs.get("load_with_mp", False)
        # assert isinstance(load_with_mp, bool)
        # self.load_with_mp = load_with_mp
        # num_cpu_workers = kwargs.get("num_cpu_workers")
        # total_cpu_workers = t_mp.cpu_count()
        # if num_cpu_workers is None:
        #     self.num_cpu_workers = int(total_cpu_workers * 0.75)
        # else:
        #     assert isinstance(num_cpu_workers, int) and 0 < num_cpu_workers < total_cpu_workers
        #     self.num_cpu_workers = num_cpu_workers
        # if self.num_cpu_workers == 1:
        #     self.load_with_mp = False

        # Get Selected Video Names and Paths
        video_names = [video_name for v_idx, video_name in enumerate(video_names) if v_idx in video_indices]
        video_paths = [os.path.join(root_path, vn) for vn in video_names]

        # Set Video Information Variables
        self.video_info = {
            "indices": video_indices, "names": video_names,
        }

        # Initialize tqdm iteration object
        benchmark_tqdm_iter_obj = tqdm(
            video_names, leave=True,
            desc="Loading FFN Data for Benchmark [{}]".format(benchmark)
        )

        # Initialize List of "VIDEO_DATA_OBJ"
        self.video_data_objs = []
        accum_sample_numbers = 0
        for video_idx, video_name in enumerate(benchmark_tqdm_iter_obj):
            video_path = video_paths[video_idx]
            video_data_obj = VIDEO_DATA_OBJ(
                root_path=video_path, benchmark=benchmark, video_name=video_name, logger=logger,
                overlap_criterion=overlap_criterion, overlap_thresholds=overlap_thresholds,
                labeling_type=labeling_type, device=device
            )
            video_data_obj.set_labels()
            self.video_data_objs.append(video_data_obj)

            # Accumulate Sample Numbers
            accum_sample_numbers += (len(self.video_data_objs[-1]) - 1)

            # Compute Memory Percent
            curr_memory_percent = psutil.virtual_memory().percent
            if curr_memory_percent > __MEMORY_CUTOFF_PERCENT__:
                raise OverflowError("Memory Critical...!")

            # Set tqdm postfix
            benchmark_tqdm_iter_obj.set_postfix({
                "Video": video_name,
                "Accum. Sample Number": "{:,}".format(accum_sample_numbers),
                "RAM Memory": "{:.2f}%".format(curr_memory_percent),
            })

        # Gather below variables and concatenate
        # (1) ffn_outputs_path || (2) overlaps || (3) labels
        self.ffn_filepaths, self.overlaps, self.labels = [], [], []
        self.gt_bboxes, self.trk_bboxes = [], []
        for video_data_obj in self.video_data_objs:
            self.ffn_filepaths = [*self.ffn_filepaths, *video_data_obj.ffn_filepaths[1:]]
            self.overlaps.append(video_data_obj.overlaps[1:])
            self.labels.append(video_data_obj.labels[1:])
            self.gt_bboxes.append(video_data_obj.gt_bboxes[1:])
            self.trk_bboxes.append(video_data_obj.trk_bboxes[1:])
        self.overlaps = np.concatenate(self.overlaps)
        self.labels = np.concatenate(self.labels, axis=0)
        self.gt_bboxes = np.concatenate(self.gt_bboxes, axis=0)
        self.trk_bboxes = np.concatenate(self.trk_bboxes, axis=0)
        assert self.overlaps.shape[0] == self.labels.shape[0] == len(self.ffn_filepaths)

        # Debug Mode
        is_debug_mode = kwargs.get("is_debug_mode", False)
        assert isinstance(is_debug_mode, bool)
        if is_debug_mode:
            debug_samples = kwargs.get("debug_samples", int(len(self.ffn_filepaths) * 0.1))
            assert isinstance(debug_samples, int) and 0 < debug_samples <= len(self.ffn_filepaths)
            self.ffn_filepaths = self.ffn_filepaths[:debug_samples]
            self.overlaps, self.labels = self.overlaps[:debug_samples], self.labels[:debug_samples]
            self.gt_bboxes, self.trk_bboxes = self.gt_bboxes[:debug_samples], self.trk_bboxes[:debug_samples]

        # Set Random Seed if not None
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Random Permutation for Shuffling
        rand_perm = np.random.permutation(len(self))
        self.rand_perm = rand_perm

        # Permute Important Variables w.r.t. "rand_perm"
        self.ffn_filepaths = [self.ffn_filepaths[j] for j in rand_perm]
        self.overlaps, self.labels = self.overlaps[rand_perm], self.labels[rand_perm]
        self.gt_bboxes, self.trk_bboxes = self.gt_bboxes[rand_perm], self.trk_bboxes[rand_perm]

        # Load Numpy Arrays via loading mode of "mmap", initialize list of mmap arrays
        self.ffn_mmaps = [None] * len(self.ffn_filepaths)
        self.__load_ffn()

        # Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return len(self.ffn_filepaths)

    def __repr__(self):
        return self.benchmark

    def __getitem__(self, item):
        assert isinstance(item, (int, slice))
        if isinstance(item, int):
            assert 0 <= item < len(self)
            return {
                "ffn_filepath": self.ffn_filepaths[item], "ffn_mmap": self.ffn_mmaps[item],
                "overlap": self.overlaps[item], "label": self.labels[item],
                "gt_bbox": self.gt_bboxes[item], "trk_bbox": self.trk_bboxes[item],
                "miscs": {
                    "overlap_criterion": self.overlap_criterion,
                    "overlap_thresholds": self.overlap_thresholds,
                }
            }
        else:
            item_start = 0 if item.start is None else item.start
            item_step = 1 if item.step is None else item.step
            if item.stop is None:
                item_stop = len(self)
            else:
                if item.stop < 0:
                    item_stop = len(self) + item.stop + 1
                else:
                    item_stop = item.stop
            ret_list = []
            for j_idx in range(item_start, item_stop, item_step):
                j_data_dict = self[j_idx]
                ret_list.append(j_data_dict)
            return ret_list

    def __iter__(self):
        return self

    def __next__(self):
        try:
            ret_val = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return ret_val

    def __load_ffn(self):
        # Define tqdm iteration object
        mmap_load_tqdm_iter = tqdm(
            self.ffn_filepaths, desc="Loading Fusion Vectors via MMAP Mode...!", leave=True
        )

        # Open First npy file for probing information
        for jj, ffn_filepath in enumerate(mmap_load_tqdm_iter):
            # Load via MMAP Mode
            ffn_mmap = np.load(ffn_filepath, mmap_mode="c")
            self.ffn_mmaps[jj] = ffn_mmap

            # Compute Memory Percent
            curr_memory_percent = psutil.virtual_memory().percent
            if curr_memory_percent > __MEMORY_CUTOFF_PERCENT__:
                raise OverflowError("Memory Critical...!")

            # Set tqdm postfix
            if jj % 100 == 0:
                mmap_load_tqdm_iter.set_postfix({
                    "RAM Memory": "{:.2f}%".format(curr_memory_percent),
                })

        # Close tqdm iteration object
        mmap_load_tqdm_iter.close()

    def _revert_shuffling(self):
        def invert_permutation(p):
            """Return an array s with which np.array_equal(arr[p][s], arr) is True.
            The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
            """
            p = np.asanyarray(p)  # in case p is a tuple, etc.
            s = np.empty_like(p)
            s[p] = np.arange(p.size)
            return s

        # Get Inverse Random Permutation
        inverse_rand_perm = invert_permutation(self.rand_perm)

        # Revert by Applying Inverse Permutation
        self.ffn_filepaths = [self.ffn_filepaths[jj] for jj in inverse_rand_perm]
        if hasattr(self, "ffn_mmaps"):
            self.ffn_mmaps = [self.ffn_mmaps[jj] for jj in inverse_rand_perm]
        self.overlaps, self.labels = self.overlaps[inverse_rand_perm], self.labels[inverse_rand_perm]
        self.gt_bboxes, self.trk_bboxes = self.gt_bboxes[inverse_rand_perm], self.trk_bboxes[inverse_rand_perm]

    def shuffle(self, random_seed=None):
        # Revert Shuffling
        self._revert_shuffling()

        # Set Random Seed if not None
        if random_seed is not None:
            assert isinstance(random_seed, int) and random_seed > 0
            if random_seed == self.random_seed:
                raise AssertionError("Same Random Seed...!")
            self.random_seed = random_seed
            np.random.seed(self.random_seed)

        # Random Permutation for Shuffling
        rand_perm = np.random.permutation(len(self))
        self.rand_perm = rand_perm

        # Permute Important Variables w.r.t. "rand_perm"
        self.ffn_filepaths = [self.ffn_filepaths[j] for j in rand_perm]
        if hasattr(self, "ffn_mmaps"):
            self.ffn_mmaps = [self.ffn_mmaps[j] for j in rand_perm]
        self.overlaps, self.labels = self.overlaps[rand_perm], self.labels[rand_perm]
        self.gt_bboxes, self.trk_bboxes = self.gt_bboxes[rand_perm], self.trk_bboxes[rand_perm]

    def set_labels(self, **kwargs):
        # Unpack KWARGS
        labeling_type = kwargs.get("labeling_type")
        if labeling_type is None:
            labeling_type = self.labeling_type
        else:
            assert labeling_type in ["one_hot", "scalar"]
            self.labeling_type = labeling_type

        overlap_thresholds = kwargs.get("overlap_thresholds")
        if overlap_thresholds is None:
            overlap_thresholds = self.overlap_thresholds
        else:
            assert isinstance(overlap_thresholds, (list, tuple, np.ndarray))
            assert len(overlap_thresholds) > 0
            if isinstance(overlap_thresholds, tuple):
                overlap_thresholds = list(overlap_thresholds)
            elif isinstance(overlap_thresholds, np.ndarray):
                overlap_thresholds = overlap_thresholds.tolist()
            overlap_thresholds = sorted(overlap_thresholds)
            self.overlap_thresholds = overlap_thresholds

        video_indices = kwargs.get("video_indices", self.video_info["indices"])
        assert set(video_indices).issubset(self.video_info["indices"])

        # Iteratively Set Labels for "video_data_objs", gather labels simultaneously
        labels = []
        for v_idx, video_data_obj in enumerate(self.video_data_objs):
            if v_idx in video_indices:
                video_data_obj.set_labels(
                    labeling_type=labeling_type, overlap_thresholds=overlap_thresholds
                )
            labels.append(video_data_obj.labels)

        # Concatenate Labels
        self.labels = np.concatenate(labels, axis=0)

    def reload(self, **kwargs):
        # Unpack KWARGS
        overlap_criterion = kwargs.get("overlap_criterion", self.overlap_criterion)
        assert overlap_criterion in ["iou", "giou", "diou", "ciou"]
        # self.overlap_criterion = overlap_criterion
        overlap_thresholds = kwargs.get("overlap_thresholds", self.overlap_thresholds)
        assert isinstance(overlap_thresholds, (list, tuple, np.ndarray))
        if isinstance(overlap_thresholds, tuple):
            overlap_thresholds = list(overlap_thresholds)
        elif isinstance(overlap_thresholds, np.ndarray):
            overlap_thresholds = overlap_thresholds.tolist()
        # self.overlap_thresholds = sorted(overlap_thresholds)
        labeling_type = kwargs.get("labeling_type", self.labeling_type)
        assert labeling_type in ["one_hot", "scalar"]
        # self.labeling_type = labeling_type

        # If "overlap_criterion" changed,
        if overlap_criterion != self.overlap_criterion:
            if overlap_criterion == "diou":
                self.overlaps = rect_diou(self.gt_bboxes, self.trk_bboxes)
            elif overlap_criterion == "iou":
                self.overlaps = rect_iou(self.gt_bboxes, self.trk_bboxes)
            else:
                raise NotImplementedError()
            self.overlap_criterion = overlap_criterion

        # If "overlap_thresholds" or "labeling_type" changed,
        if overlap_thresholds != self.overlap_thresholds or labeling_type != self.labeling_type:
            self.set_labels(
                overlap_thresholds=overlap_thresholds, labeling_type=labeling_type
            )


class VIDEO_DATA_OBJ(object):
    def __init__(self, root_path, benchmark, video_name, logger, **kwargs):
        assert os.path.isdir(root_path)

        # Set Basic Variables
        self.root_path = root_path
        self.benchmark, self.video_name = benchmark, video_name
        self.logger = logger

        # Unpack KWARGS
        overlap_criterion = kwargs.get("overlap_criterion")
        assert overlap_criterion in ["iou", "giou", "diou", "ciou"]
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
        device = kwargs.get("device", __CUDA_DEVICE__)
        self.device = device

        # Initialize "self.labels"
        self.labels = None

        # List Video Data
        video_dir_names = os.listdir(root_path)
        for video_dir_name in video_dir_names:
            if video_dir_name.endswith("npy"):
                # check if it is file,
                video_filepath = os.path.join(root_path, video_dir_name)
                # assert os.path.isfile(video_filepath)
                setattr(self, "{}_path".format(video_dir_name.split(".")[0]), video_filepath)
            else:
                # check if it is a ffn directory,
                video_ffn_dir = os.path.join(root_path, video_dir_name)
                assert os.path.isdir(video_ffn_dir)
                frame_ffn_filenames = sorted(os.listdir(video_ffn_dir))
                self.ffn_filepaths = [os.path.join(video_ffn_dir, _fn) for _fn in frame_ffn_filenames]
                # self.ffn_mmaps = [np.load(fp, mmap_mode="c") for fp in self.ffn_filepaths]

        # # Smart Load "ffn_feats" -> Open First FFN Feature as an example, find its shape.
        # #                           initialize Tensor (float32) of the found shape.
        # #                           Directly initialize on GPU
        # init_sample_ffn_feat = np.load(self.ffn_filepaths[0])
        # N, S, C = len(self.ffn_filepaths), init_sample_ffn_feat.shape[0], init_sample_ffn_feat.shape[1]
        # with torch.cuda.device(device):
        #     self.ffn_feats = torch.cuda.FloatTensor(N+1, S, C).fill_(0) # noqa
        #     self.ffn_feats[0] = float("nan")
        # for f_idx in range(N):
        #     self.ffn_feats[f_idx+1] = torch.from_numpy(np.load(self.ffn_filepaths[f_idx])).cuda(device)

        # Insert "None" to "self.ffn_filepaths"
        self.ffn_filepaths.insert(0, None)
        # self.ffn_mmaps.insert(0, None)

        # Load Bounding Boxes
        self.gt_bboxes, self.trk_bboxes = \
            np.load(self.gt_bboxes_path), np.load(self.trk_bboxes_path)

        # Convert BBOX Format from [L T R B] to [L T W H]
        self.gt_bboxes[:, 2:4] -= self.gt_bboxes[:, 0:2]
        self.trk_bboxes[:, 2:4] -= self.trk_bboxes[:, 0:2]

        # Set Overlap
        if overlap_criterion == "iou":
            self.overlaps = rect_iou(self.gt_bboxes, self.trk_bboxes)
        elif overlap_criterion == "diou":
            self.overlaps = rect_diou(self.gt_bboxes, self.trk_bboxes)
        else:
            raise NotImplementedError()

        # Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return self.gt_bboxes.shape[0]

    def __repr__(self):
        return self.video_name

    def __getitem__(self, item):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            ret_val = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return ret_val

    def set_labels(self, **kwargs):
        # Unpack KWARGS
        labeling_type = kwargs.get("labeling_type")
        if labeling_type is None:
            labeling_type = self.labeling_type
        else:
            assert labeling_type in ["one_hot", "scalar"]
            self.labeling_type = labeling_type

        overlap_thresholds = kwargs.get("overlap_thresholds")
        if overlap_thresholds is None:
            overlap_thresholds = self.overlap_thresholds
        else:
            assert isinstance(overlap_thresholds, (list, tuple, np.ndarray))
            assert len(overlap_thresholds) > 0
            if isinstance(overlap_thresholds, tuple):
                overlap_thresholds = list(overlap_thresholds)
            elif isinstance(overlap_thresholds, np.ndarray):
                overlap_thresholds = overlap_thresholds.tolist()
            overlap_thresholds = sorted(overlap_thresholds)
            self.overlap_thresholds = overlap_thresholds

        # Set Labels
        labels = []
        for overlap in self.overlaps:
            # Initialize "label" array for every iteration
            label = np.zeros(len(overlap_thresholds)+1).astype(int)

            # Compute Label for Current Sample
            for i_idx, overlap_thresh in enumerate(overlap_thresholds):
                if overlap < overlap_thresh:
                    np.add.at(label, [i_idx], 1)
                detect_true_label = np.where(label == 1)[0]
                if len(detect_true_label) > 0:
                    break
            if label.max() == 0:
                label[-1] = 1
            if labeling_type == "scalar":
                label = label.argmax()

            # Append to List
            labels.append(label)

        # Convert to Numpy Array
        self.labels = np.array(labels)


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

        # load_with_mp=True, num_cpu_workers=1,
    )

    pass
