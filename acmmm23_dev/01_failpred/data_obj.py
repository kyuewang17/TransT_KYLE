from __future__ import absolute_import, print_function

import os
import psutil
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from utils.bbox import IoU
from utils.rect_metrics import rect_iou, rect_diou


__FFN_DATA_ROOT_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE/acmmm23_dev/ffn_data"
__BENCHMARK_DATASET__ = "OTB100"


class BENCHMARK_FFN_OBJECT(object):
    def __init__(self, root_path, benchmark, overlap_criterion, **kwargs):
        assert os.path.isdir(root_path)
        assert overlap_criterion in ["iou", "giou", "diou"]

        # Unpack KWARGS
        video_indices = kwargs.get("load_video_indices")
        assert isinstance(video_indices, (int, list, tuple, np.ndarray))
        if isinstance(video_indices, int):
            video_indices = [video_indices]
        elif isinstance(video_indices, tuple):
            video_indices = list(video_indices)
        elif isinstance(video_indices, np.ndarray):
            video_indices = video_indices.tolist()
        video_indices = sorted(video_indices)
        overlap_thresholds = kwargs.get("overlap_thresholds")
        labeling_type = kwargs.get("labeling_type")
        is_auto_labeling = kwargs.get("is_auto_labeling", False)
        assert isinstance(is_auto_labeling, bool)
        is_drop_hier_objs = kwargs.get("is_drop_hier_objs", False)
        assert isinstance(is_drop_hier_objs, bool)

        # Set Root Path
        self.root_path = root_path

        # Set Benchmark
        self.benchmark = benchmark

        # Check "root_path" subdirectories, get Video Names and the paths
        video_names = sorted(os.listdir(root_path))
        if video_indices is not None:
            video_names = [video_name for v_idx, video_name in enumerate(video_names) if v_idx in video_indices]
        video_paths = [os.path.join(root_path, vn) for vn in video_names]
        self.video_indices, self.video_names = video_indices, video_names

        # Initialize tqdm iteration object
        bench_tqdm_iter_obj = tqdm(
            video_names, desc="Loading FFN Data for Benchmark [{}]".format(benchmark), leave=True
        )

        # Initialize list of "VIDEO_FFN_OBJ"
        self.video_ffn_objs = []
        accum_sample_numbers = 0
        for video_idx, video_name in enumerate(bench_tqdm_iter_obj):
            video_path = video_paths[video_idx]
            self.video_ffn_objs.append(
                VIDEO_FFN_OBJECT(
                    video_name=video_name, video_path=video_path, benchmark=benchmark,
                    overlap_criterion=overlap_criterion,
                )
            )

            # Accumulate Sample Numbers
            accum_sample_numbers += len(self.video_ffn_objs[-1])

            # Compute Memory Percent
            curr_memory_percent = psutil.virtual_memory().percent

            # Set tqdm postfix
            bench_tqdm_iter_obj.set_postfix({
                "Video": video_name,
                "Accum. Sample Number": "{:,}".format(accum_sample_numbers),
                "RAM Memory": "{:.2f}%".format(curr_memory_percent),
            })

        # Concatenate all IoUs and FFN outputs
        self.video_frame_lens = []
        gathered_overlap, gathered_ffn_outputs = [], []
        for video_ffn_obj in self.video_ffn_objs:
            self.video_frame_lens.append(len(video_ffn_obj))

            # Gathered IoU
            video_overlaps = video_ffn_obj.overlaps
            gathered_overlap.append(video_overlaps)

            # Get FFN outputs of video
            video_ffn_outputs = video_ffn_obj.get_ffn_outputs(include_init=True)
            gathered_ffn_outputs.append(video_ffn_outputs)
        self.gathered_overlap = np.concatenate(gathered_overlap)
        self.gathered_ffn_outputs = np.concatenate(gathered_ffn_outputs, axis=0)
        assert self.gathered_overlap.shape[0] == self.gathered_ffn_outputs.shape[0]

        # Overlap Thresholds and Criterion
        self.overlap_thresholds, self.overlap_criterion = overlap_thresholds, overlap_criterion

        # Labeling Type
        self.labeling_type = labeling_type

        # Label of Samples w.r.t. IoU thresholds
        self.gathered_samples_label = None
        if is_auto_labeling:
            if overlap_thresholds is None or overlap_criterion is None or labeling_type is None:
                raise AssertionError()
            self.set_overlap(
                overlap_thresholds=overlap_thresholds, is_recursive_setting=True
            )
            self.set_label(
                labeling_type=labeling_type, is_recursive_setting=True
            )

        # Drop hierarchical objects
        if is_drop_hier_objs:
            self.video_ffn_objs = []

    def __repr__(self):
        return self.benchmark

    def __len__(self):
        return len(self.gathered_overlap)

    def set_overlap(self, overlap_thresholds, is_recursive_setting=False):
        # Check Threshold Validity and Modify appropriately if not so wrong.
        assert isinstance(overlap_thresholds, (float, list, tuple, np.ndarray))
        if isinstance(overlap_thresholds, float):
            overlap_thresholds = [overlap_thresholds]
        elif isinstance(overlap_thresholds, tuple):
            overlap_thresholds = list(overlap_thresholds)
        elif isinstance(overlap_thresholds, np.ndarray):
            overlap_thresholds = overlap_thresholds.tolist()

        # Traverse "iou_thresholds" list and check the range
        for overlap_threshold in overlap_thresholds:
            assert isinstance(overlap_threshold, float)
            if self.overlap_criterion == "iou":
                assert 0 <= overlap_threshold <= 1
            elif self.overlap_criterion == "giou":
                raise NotImplementedError()
            elif self.overlap_criterion == "diou":
                assert -1 <= overlap_threshold <= 1
            else:
                raise AssertionError()
        self.overlap_thresholds = sorted(overlap_thresholds)

        # If gathered samples label exist, change label w.r.t. "overlap_thresholds"
        if self.gathered_samples_label is not None:
            self.set_label(labeling_type=self.labeling_type)

        # For Recursive Setting
        if is_recursive_setting:
            if len(self.video_ffn_objs) == 0:
                raise AssertionError()

            # Iterate for "self.video_ffn_objs"
            for video_ffn_obj in self.video_ffn_objs:
                # Set Overlap
                video_ffn_obj.set_overlap(
                    overlap_thresholds=overlap_thresholds, is_recursive_setting=True
                )

    def set_label(self, labeling_type, is_recursive_setting=False):
        assert labeling_type in ["one_hot", "scalar"]
        self.labeling_type = labeling_type

        # Raise Assertion if "self.overlap_thresholds" is None
        if self.overlap_thresholds is None:
            raise AssertionError()

        # Initialize "self.gathered_samples_label" Array
        gathered_samples_label = []
        for overlap in self.gathered_overlap:

            # Initialize "label" array every iteration
            label = np.zeros(len(self.overlap_thresholds)+1).astype(int)

            # Compute Label for Current Sample
            for i_idx, overlap_thresh in enumerate(self.overlap_thresholds):
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
            gathered_samples_label.append(label)

        # Convert to Numpy Array
        self.gathered_samples_label = np.array(gathered_samples_label)

        # For Recursive Setting
        if is_recursive_setting:
            if len(self.video_ffn_objs) == 0:
                raise AssertionError()

            # Iterate for "self.video_ffn_objs"
            for video_ffn_obj in self.video_ffn_objs:
                # Set Label
                video_ffn_obj.set_label(labeling_type=labeling_type, is_recursive_setting=True)


class VIDEO_FFN_OBJECT(object):
    def __init__(self, video_name, video_path, overlap_criterion, **kwargs):
        assert os.path.isdir(video_path)
        assert overlap_criterion in ["iou", "giou", "diou"]

        # Unpack KWARGS
        self.benchmark = kwargs.get("benchmark")
        overlap_thresholds = kwargs.get("overlap_thresholds")
        labeling_type = kwargs.get("labeling_type")

        # Set Video Name and Path
        self.video_name, self.video_path = video_name, video_path

        # Load "ffn_outputs", "gt_bboxes", and "trk_bboxes"
        ffn_outputs = np.load(os.path.join(video_path, "ffn_outputs.npy"))
        gt_bboxes = np.load(os.path.join(video_path, "gt_bboxes.npy"))
        trk_bboxes = np.load(os.path.join(video_path, "trk_bboxes.npy"))

        # Initialize list of "FRAME_FFN_OBJECT"s
        self.frame_ffn_objs = []
        for fidx in range(gt_bboxes.shape[0]):
            ffn_output = ffn_outputs[fidx-1] if fidx > 0 else None
            gt_bbox, trk_bbox = gt_bboxes[fidx], trk_bboxes[fidx]
            self.frame_ffn_objs.append(
                FRAME_FFN_OBJECT(
                    ffn_output=ffn_output, gt_bbox=gt_bbox, trk_bbox=trk_bbox,
                    overlap_criterion=overlap_criterion,
                )
            )

        # Extract Overlaps
        overlaps = []
        for frame_ffn_obj in self.frame_ffn_objs:
            overlaps.append(frame_ffn_obj.overlap)
        self.overlaps = np.array(overlaps)

        # Overlap Thresholds & Criterion
        self.overlap_thresholds, self.overlap_criterion = overlap_thresholds, overlap_criterion

        # Labeling Type
        self.labeling_type = labeling_type

        # Labels w.r.t. Overlap Thresholds
        self.labels = None

        # Iteration Counter
        self.__iter_counter = 0

    def __repr__(self):
        return self.video_name

    def __len__(self):
        return len(self.frame_ffn_objs)

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

    def drop_hier_objs(self):
        self.frame_ffn_objs = []

    def set_overlap(self, overlap_thresholds, is_recursive_setting=False):
        # Check Threshold Validity and Modify appropriately if not so wrong.
        assert isinstance(overlap_thresholds, (float, list, tuple, np.ndarray))
        if isinstance(overlap_thresholds, float):
            overlap_thresholds = [overlap_thresholds]
        elif isinstance(overlap_thresholds, tuple):
            overlap_thresholds = list(overlap_thresholds)
        elif isinstance(overlap_thresholds, np.ndarray):
            overlap_thresholds = overlap_thresholds.tolist()

        # Traverse "iou_thresholds" list and check the range
        for overlap_threshold in overlap_thresholds:
            assert isinstance(overlap_threshold, float)
            if self.overlap_criterion == "iou":
                assert 0 <= overlap_threshold <= 1
            elif self.overlap_criterion == "giou":
                raise NotImplementedError()
            elif self.overlap_criterion == "diou":
                assert -1 <= overlap_threshold <= 1
            else:
                raise AssertionError()
        self.overlap_thresholds = sorted(overlap_thresholds)

        # If labels exist, change label w.r.t. "overlap_thresholds"
        if self.labels is not None:
            self.set_label(labeling_type=self.labeling_type)

        # For Recursive Setting
        if is_recursive_setting:
            # Iterate for "self.frame_ffn_objs"
            for frame_ffn_obj in self.frame_ffn_objs:
                frame_ffn_obj.set_overlap(overlap_thresholds=overlap_thresholds)

    def set_label(self, labeling_type, is_recursive_setting=False):
        assert labeling_type in ["one_hot", "scalar"]
        self.labeling_type = labeling_type

        # Raise Assertion if "self.overlap_thresholds" is None
        if self.overlap_thresholds is None:
            raise AssertionError()

        # Initialize "self.classes" Array
        labels = []
        for overlap in self.overlaps:
            # Initialize "label" array every iteration
            label = np.zeros(len(self.overlap_thresholds) + 1).astype(int)

            # Compute Label for Current Sample
            for i_idx, overlap_thresh in enumerate(self.overlap_thresholds):
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

        # For Recursive Setting
        if is_recursive_setting:
            # Iterate for "self.frame_ffn_objs"
            for frame_ffn_obj in self.frame_ffn_objs:
                # Set Label
                frame_ffn_obj.set_label(labeling_type=labeling_type)

    def get_ffn_outputs(self, include_init=False):
        ffn_outputs = np.load(os.path.join(self.video_path, "ffn_outputs.npy"))
        if include_init:
            init_ffn_sub = np.empty(shape=ffn_outputs.shape[1:])
            init_ffn_sub.fill(np.nan)
            return np.concatenate((np.expand_dims(init_ffn_sub, axis=0), ffn_outputs), axis=0)
        else:
            return ffn_outputs


class FRAME_FFN_OBJECT(object):
    def __init__(self, ffn_output, gt_bbox, trk_bbox, overlap_criterion, **kwargs):
        self.is_init_frame = True if ffn_output is None else False
        assert overlap_criterion in ["iou", "giou", "diou"]

        # Unpack KWARGS
        overlap_thresholds = kwargs.get("overlap_thresholds")
        labeling_type = kwargs.get("labeling_type")

        # Set "ffn_output", "gt_bbox", and "trk_bbox"
        self.ffn_output = ffn_output
        gt_bbox[2], gt_bbox[3] = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
        trk_bbox[2], trk_bbox[3] = trk_bbox[2] - trk_bbox[0], trk_bbox[3] - trk_bbox[1]
        self.gt_bbox, self.trk_bbox = deepcopy(gt_bbox), deepcopy(trk_bbox)

        # Overlap Thresholds & Criterion
        self.overlap_thresholds, self.overlap_criterion = overlap_thresholds, overlap_criterion

        # Compute Overlap btw GT and TRK BBOXES
        if overlap_criterion == "iou":
            self.overlap = rect_iou(gt_bbox, trk_bbox)
        elif overlap_criterion == "giou":
            raise NotImplementedError()
        elif overlap_criterion == "diou":
            self.overlap = rect_diou(gt_bbox, trk_bbox)
        else:
            raise AssertionError()

        # Labeling Type
        self.labeling_type = labeling_type

        # Label
        self.label = None

        # Iteration Counter
        self.__iter_counter = 0

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

    def set_overlap(self, overlap_thresholds):
        # Check Threshold Validity and Modify appropriately if not so wrong.
        assert isinstance(overlap_thresholds, (float, list, tuple, np.ndarray))
        if isinstance(overlap_thresholds, float):
            overlap_thresholds = [overlap_thresholds]
        elif isinstance(overlap_thresholds, tuple):
            overlap_thresholds = list(overlap_thresholds)
        elif isinstance(overlap_thresholds, np.ndarray):
            overlap_thresholds = overlap_thresholds.tolist()

        # Traverse "iou_thresholds" list and check the range
        for overlap_threshold in overlap_thresholds:
            assert isinstance(overlap_threshold, float)
            if self.overlap_criterion == "iou":
                assert 0 <= overlap_threshold <= 1
            elif self.overlap_criterion == "giou":
                raise NotImplementedError()
            elif self.overlap_criterion == "diou":
                assert -1 <= overlap_threshold <= 1
            else:
                raise AssertionError()
        self.overlap_thresholds = sorted(overlap_thresholds)

        # If label exists, change label w.r.t. "overlap_thresholds"
        if self.label is not None:
            self.set_label(labeling_type=self.labeling_type)

    def set_label(self, labeling_type):
        assert labeling_type in ["one_hot", "scalar"]
        self.labeling_type = labeling_type

        # Raise Assertion if "self.overlap_thresholds" is None
        if self.overlap_thresholds is None:
            raise AssertionError()

        # Initialize "label" array
        label = np.zeros(len(self.overlap_thresholds) + 1).astype(int)

        # Compute Label for Current Sample
        for i_idx, overlap_thresh in enumerate(self.overlap_thresholds):
            if self.overlap < overlap_thresh:
                np.add.at(label, [i_idx], 1)
            detect_true_label = np.where(label == 1)[0]
            if len(detect_true_label) > 0:
                break
        if label.max() == 0:
            label[-1] = 1
        if labeling_type == "scalar":
            label = label.argmax()

        # Assign to "self.label"
        self.label = label


if __name__ == "__main__":

    ffn_data_path = os.path.join(__FFN_DATA_ROOT_PATH__, __BENCHMARK_DATASET__)

    # FFN Object
    FFN_OBJ = BENCHMARK_FFN_OBJECT(
        root_path=ffn_data_path, benchmark=__BENCHMARK_DATASET__,
        load_video_indices=[0, 1, 2, 3],

        # Overlap-related Arguments
        overlap_criterion="iou",
        overlap_thresholds=[0.5],
        # overlap_thresholds=[0.0],

        # Labeling-related Arguments
        labeling_type="one_hot", is_auto_labeling=True,
    )


    print(123)
