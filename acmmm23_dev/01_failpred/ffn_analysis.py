from __future__ import absolute_import, print_function

import os
import psutil
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TF
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.bbox import IoU
from data_obj import BENCHMARK_FFN_OBJECT, VIDEO_FFN_OBJECT


__FFN_DATA_ROOT_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE/acmmm23_dev/ffn_data"
# __BENCHMARK_DATASET__ = "OTB100"
__BENCHMARK_DATASET__ = "UAV123"

# FFN Analysis Save Path
__FFN_ANALYSIS_SAVE_PATH__ = "/home/kyle/Dropbox/SNU/Research/ACM_MM23/230214/ffn_analysis"


class FFN_ANALYSIS_BENCHMARK_OBJ(BENCHMARK_FFN_OBJECT):
    def __init__(self, root_path, benchmark, overlap_criterion, **kwargs):
        super().__init__(root_path, benchmark, overlap_criterion, **kwargs)

    def reinit(self):
        assert len(self.video_ffn_objs) > 0
        new_video_ffn_objs = []
        for video_ffn_obj in self.video_ffn_objs:
            _OBJ = FFN_ANALYSIS_VIDEO_OBJ(
                    video_name=video_ffn_obj.video_name, video_path=video_ffn_obj.video_path,
                    benchmark=video_ffn_obj.benchmark,
                    overlap_criterion=video_ffn_obj.overlap_criterion,
                    overlap_thresholds=video_ffn_obj.overlap_thresholds,
                    labeling_type=video_ffn_obj.labeling_type,
            )
            _OBJ.set_label(labeling_type=video_ffn_obj.labeling_type)
            new_video_ffn_objs.append(_OBJ)
        self.video_ffn_objs = new_video_ffn_objs

    def plot_data(self, **kwargs):


        # Assertion

        # Unpack KWARGS
        overlap_thresholds = kwargs.get("overlap_thresholds")
        if overlap_thresholds is not None:
            assert isinstance(overlap_thresholds, (list, tuple, np.ndarray))
        else:
            overlap_thresholds = self.overlap_thresholds

        sampling_ratio = kwargs.get("sampling_ratio", 0.05)
        assert 0 < sampling_ratio <= 1
        sampling_seed = kwargs.get("sampling_seed", 1234)

        cmap_name = kwargs.get("cmap_name", "bwr_r")
        overlap_colormap_center_val = kwargs.get("overlap_colormap_center_val")
        elev_3d, azim_3d = kwargs.get("elev_3d", 15), kwargs.get("azim_3d", -45)

        # is_save_figs = kwargs.get("is_save_figs", False)
        # assert isinstance(is_save_figs, bool)

        # Sample Data
        if sampling_ratio < 1:
            sampling_number = int(len(self.gathered_overlap) * sampling_ratio)
            np.random.seed(sampling_seed)
            sampling_indices = \
                np.random.choice(np.arange(len(self.gathered_overlap)), sampling_number, replace=False)
            sampling_indices = np.sort(sampling_indices)

            # Initialize Empty Numpy Array of Sampling Number Length
            sampled_overlap = self.gathered_overlap[sampling_indices]
            sampled_label = self.gathered_samples_label[sampling_indices]
            sampled_ffn_outputs = self.gathered_ffn_outputs[sampling_indices]

            # DEBUG DEBUG
            # plt.imshow(sampled_ffn_outputs.max(axis=1), cmap="jet", interpolation="nearest")
            # plt.imshow(sampled_ffn_outputs.mean(axis=1), cmap="jet", interpolation="nearest")
            plt.imshow(sampled_ffn_outputs.std(axis=1), cmap="jet", interpolation="nearest")
            plt.show()

            print(123)

            # # for statistical analysis,
            # if mode == "statistics":
            #     plot_samples =

            # TODO: Plot with the following methods...
            #       (1) - Statistical Terms (mean, max, std)
            #       (2) - High-dimensional Visualization Tools (t-SNE)



class FFN_ANALYSIS_VIDEO_OBJ(VIDEO_FFN_OBJECT):
    def __init__(self, video_name, video_path, overlap_criterion, **kwargs):
        super().__init__(video_name, video_path, overlap_criterion, **kwargs)

        # Performance Dictionary
        self.performance_dict = {}

    def compute_precision(self, **kwargs):
        raise NotImplementedError()

    def compute_success(self):
        return np.nanmean(self.overlaps)

    def analyze_video_ffn(self, **kwargs):
        # Unpack KWARGS
        overlap_thresholds = kwargs.get("overlap_thresholds")
        if overlap_thresholds is not None:
            assert isinstance(overlap_thresholds, (list, tuple, np.ndarray))
            labels = []
            for overlap in self.overlaps:
                label = np.zeros(len(overlap_thresholds) + 1).astype(int)
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
                labels.append(label)
        else:
            labels = self.labels

        cmap_name = kwargs.get("cmap_name", "jet")
        # elev_3d, azim_3d = kwargs.get("elev_3d", 15), kwargs.get("azim_3d", -45)

        is_save_mode = kwargs.get("is_save_mode", False)
        assert isinstance(is_save_mode, bool)
        save_base_path = kwargs.get("save_base_path")
        if save_base_path is not None:
            if is_save_mode:
                assert os.path.isdir(save_base_path)
        save_format = kwargs.get("save_format", "png")

        # Get FFN outputs
        ffn_outputs = self.get_ffn_outputs(include_init=False)

        # Channel-wise Spatial Statistics
        ch_ffn_outputs_stats = {
            "max": ffn_outputs.max(axis=1),
            "mean": ffn_outputs.mean(axis=1),
            "std": ffn_outputs.std(axis=1),
        }

        # Get Overlaps and Labels for non-initial frames
        overlaps, labels = self.overlaps[1:], labels[1:]

        # Draw Plt Figure
        fig_stats, ax_stats = \
            plt.subplots(1, len(ch_ffn_outputs_stats), dpi=150, figsize=(15, 15))

        # === Draw Channel-wise Spatial Statistics === #
        for idx, (stat_key, stat_val) in enumerate(ch_ffn_outputs_stats.items()):

            # Draw plt imshow
            ax_stats[idx].imshow(stat_val, cmap=cmap_name, interpolation="nearest")
            ax_stats[idx].set(aspect=0.5)
            ax_stats[idx].set_title("[{}]".format(stat_key))

        # Tight Layout
        plt.tight_layout()

        # Set Super-Title
        fig_stats.suptitle(
            "[{}]_succ-[{:.3f}]".format(self.video_name, self.compute_success())
        )

        # Save Plot
        if is_save_mode:
            plt.savefig(
                os.path.join(save_base_path, "{}.{}".format(self.video_name, save_format))
            )

        # Close Figure
        plt.close(fig=fig_stats)





if __name__ == "__main__":
    # FFN Data Path
    ffn_data_path = os.path.join(__FFN_DATA_ROOT_PATH__, __BENCHMARK_DATASET__)

    # Set Iteration Dictionary
    iter_dict = {
        "1": np.arange(0, 10), "2": np.arange(10, 20), "3": np.arange(20, 30),
        "4": np.arange(30, 40), "5": np.arange(40, 50), "6": np.arange(50, 60),
        "7": np.arange(60, 70), "8": np.arange(70, 80), "9": np.arange(80, 90),
        "10": np.arange(90, 100)
    }

    N_bench = len(os.listdir(ffn_data_path))
    iter_numbers = 20
    iter_interval = N_bench // iter_numbers

    iter_indices_arr = []
    for idx in range(iter_numbers):
        start_idx, end_idx = idx * iter_interval, min((idx + 1) * iter_interval, N_bench)
        curr_iter_indices = np.arange(start_idx, end_idx)
        iter_indices_arr.append(curr_iter_indices)

    # For every iteration, initialize FFN Analysis Objects
    for idx, iter_indices in enumerate(iter_indices_arr):

        # Initialize FFN Analysis Object
        FFN_ANALYSIS_OBJ = FFN_ANALYSIS_BENCHMARK_OBJ(
            root_path=ffn_data_path, benchmark=__BENCHMARK_DATASET__,
            load_video_indices=iter_indices,

            # Overlap-related Arguments
            overlap_criterion="iou", overlap_thresholds=[0.5],

            # Labeling-related Arguments
            labeling_type="one_hot", is_auto_labeling=True,
        )
        FFN_ANALYSIS_OBJ.reinit()

        # Iterate for FFN Analysis Object of Current "iter_dict"
        for idx2 in range(len(FFN_ANALYSIS_OBJ.video_ffn_objs)):
            FFN_ANALYSIS_OBJ.video_ffn_objs[idx2].analyze_video_ffn(
                is_save_mode=True,
                # save_base_path=os.path.join(__FFN_ANALYSIS_SAVE_PATH__, "002__otb_video_spatial_statistics"),
                save_base_path=os.path.join(__FFN_ANALYSIS_SAVE_PATH__, "002__uav_video_spatial_statistics"),
            )

        # Delete Current Iteration's FFN Analysis Objects (RAM Memory Issue)
        del FFN_ANALYSIS_OBJ

