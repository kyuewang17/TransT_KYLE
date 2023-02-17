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

    def analyze_ffn_pca(self, **kwargs):
        """
        Codes referred: https://builtin.com/data-science/tsne-python

        """
        import pandas as pd
        import seaborn as sns
        from sklearn.decomposition import PCA
        from mpl_toolkits.mplot3d import Axes3D

        # Unpack KWARGS
        overlap_thresholds = kwargs.get("overlap_thresholds")
        if overlap_thresholds is not None:
            assert isinstance(overlap_thresholds, (list, tuple, np.ndarray))
            labels = []
            for overlap in self.gathered_overlap:
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
            labels = self.gathered_samples_label

        pca_components = kwargs.get("pca_components", 3)
        random_seed = kwargs.get("random_seed")
        if random_seed is not None:
            assert isinstance(random_seed, int) and random_seed >= 0
            np.random.seed(random_seed)

        # Plotting KWARGS
        scatter_size = kwargs.get("scatter_size")
        if scatter_size is not None:
            assert isinstance(scatter_size, float)
        cmap_name = kwargs.get("cmap_name", "bwr_r")
        # trunc_flen = kwargs.get("trunc_flen", 10)
        elev_3d, azim_3d = kwargs.get("elev_3d", 15), kwargs.get("azim_3d", -45)

        save_base_path = kwargs.get("save_base_path")
        is_save_mode = kwargs.get("is_save_mode", False)
        if save_base_path is not None:
            if is_save_mode:
                assert os.path.isdir(save_base_path)
        save_format = kwargs.get("save_format", "png")

        # Convert Label format to "scalar"
        if self.labeling_type == "one_hot":
            labels = labels.argmax(axis=1)

        # Find initial frame indices according to "np.nan" in "self.gathered_ffn_outputs"
        init_fidx_indices = \
            np.argwhere(np.isnan(self.gathered_ffn_outputs.reshape(len(labels), -1)).all(axis=1))
        init_fidx_indices = init_fidx_indices.reshape(-1).tolist()

        # Prepare Column and Row(row_indices) for DataFrame
        col_elems = []
        for spat_idx in range(self.gathered_ffn_outputs.shape[1]):
            for ch_idx in range(self.gathered_ffn_outputs.shape[2]):
                col_elems.append("s_{}__c_{}".format(spat_idx, ch_idx))

        row_indices = []
        for video_idx, video_frame_len in enumerate(self.video_frame_lens):
            for video_fidx in range(video_frame_len):
                row_idx_comp = "[{}]-[{:4d}]".format(self.video_names[video_idx], video_fidx)
                row_indices.append(row_idx_comp)

        # Select non-initial frame indices
        self.gathered_ffn_outputs = \
            np.delete(self.gathered_ffn_outputs, init_fidx_indices, axis=0)
        self.gathered_overlap = np.delete(self.gathered_overlap, init_fidx_indices)
        labels = np.delete(labels, init_fidx_indices)
        row_indices = \
            [row_idx for jj, row_idx in enumerate(row_indices) if jj not in init_fidx_indices]

        # === Prepare Data via pd.DataFrame === #
        # self.gathered_ffn_outputs = self.gathered_ffn_outputs[:trunc_flen]
        # row_indices = row_indices[:trunc_flen]
        # labels = labels[:trunc_flen]
        df = pd.DataFrame(
            self.gathered_ffn_outputs.reshape(self.gathered_ffn_outputs.shape[0], -1),
            columns=col_elems, index=row_indices
        )
        df["y"] = labels
        df["labels"] = df["y"].apply(lambda j: str(j))

        # Random Permutation
        rand_perm = np.random.permutation(df.shape[0])

        # === Analyze via PCA === #
        pca = PCA(n_components=pca_components)
        pca_result = pca.fit_transform(df[col_elems].values)

        for pca_comp_idx in range(pca_components):
            curr_pca_result = pca_result[:, pca_comp_idx]
            df["pca_{}".format(pca_comp_idx+1)] = curr_pca_result

        # Set Analyze Targets String
        analyze_targets = "{}_[{}-{}]".format(self.benchmark, self.video_indices[0], self.video_indices[-1])

        # === Draw Analyzed Result === #
        if pca_components == 2:
            raise NotImplementedError()

        elif pca_components >= 3:
            # Init plt figure
            fig_pca = plt.figure(dpi=200)
            ax_pca = fig_pca.add_subplot(111, projection="3d")

            # Choose df["pca_X"]
            if pca_components >= 4:
                raise NotImplementedError()
            else:
                scatter_dict = {
                    "xs": df["pca_1"].values[rand_perm],
                    "ys": df["pca_2"].values[rand_perm],
                    "zs": df["pca_3"].values[rand_perm],
                    "c": df["y"].values[rand_perm],
                }

            # Plot Scatter to Ax
            ax_pca.scatter(
                xs=scatter_dict["xs"], ys=scatter_dict["ys"], zs=scatter_dict["zs"],
                c=scatter_dict["c"], cmap=cmap_name, s=scatter_size,
            )

            # Set Labels
            ax_pca.set_xlabel("pca_1")
            ax_pca.set_ylabel("pca_2")
            ax_pca.set_zlabel("pca_3")

            # Set Figure Title
            ax_pca.set_title(analyze_targets)

        else:
            raise NotImplementedError()

        # Save Plot
        if is_save_mode:
            plt.savefig(
                os.path.join(save_base_path, "{}.{}".format(analyze_targets, save_format))
            )
        else:
            plt.show()

        plt.close(fig_pca)

    def analyze_ffn_tsne(self, **kwargs):
        """
        Codes referred: https://builtin.com/data-science/tsne-python

        """
        import pandas as pd
        import seaborn as sns







    def old_plot_data(self, **kwargs):

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

    def analyze_ffn_pca(self, **kwargs):
        """
        Codes referred: https://builtin.com/data-science/tsne-python

        """
        import pandas as pd
        import seaborn as sns
        from sklearn.decomposition import PCA
        from mpl_toolkits.mplot3d import Axes3D

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

        pca_components = kwargs.get("pca_components", 3)
        random_seed = kwargs.get("random_seed")
        if random_seed is not None:
            assert isinstance(random_seed, int) and random_seed >= 0
            np.random.seed(random_seed)
        cmap_name = kwargs.get("cmap_name", "jet")
        # trunc_flen = kwargs.get("trunc_flen", 10)

        save_base_path = kwargs.get("save_base_path")
        is_save_mode = kwargs.get("is_save_mode", False)
        if save_base_path is not None:
            if is_save_mode:
                assert os.path.isdir(save_base_path)
        save_format = kwargs.get("save_format", "png")

        # === Prepare Dataset via pd.DataFrame === #
        # Remove First Frame
        labels = labels[1:]

        # Change Label Type to "Scalar" if "One_hot"
        if self.labeling_type == "one_hot":
            labels = labels.argmax(axis=1)

        # === Truncated Length
        # labels = labels[:trunc_flen]

        # Get FFN outputs and Flatten
        ffn_outputs = self.get_ffn_outputs(include_init=False)
        # ffn_outputs = ffn_outputs[:trunc_flen]
        col_elems = []
        for spat_idx in range(ffn_outputs.shape[1]):
            for ch_idx in range(ffn_outputs.shape[2]):
                col_elems.append("spat_{}__ch_{}".format(spat_idx, ch_idx))
        row_indices = ["fidx_{}".format(j + 1) for j in range(ffn_outputs.shape[0])]
        ffn_outputs = ffn_outputs.reshape(ffn_outputs.shape[0], -1)

        # Initialize DataFrame
        df = pd.DataFrame(ffn_outputs, columns=col_elems, index=row_indices)
        df["y"] = labels
        df["label"] = df["y"].apply(lambda i: str(i))

        # Random Permutation
        rndperm = np.random.permutation(df.shape[0])

        # === Analyze via PCA === #
        pca = PCA(n_components=pca_components)
        pca_result = pca.fit_transform(df[col_elems].values)

        for pca_comp_idx in range(pca_components):
            curr_pca_result = pca_result[:, pca_comp_idx]
            df["pca_{}".format(pca_comp_idx+1)] = curr_pca_result

        # === Draw Analyzed Result === #
        if pca_components == 2:
            raise NotImplementedError()

        elif pca_components >= 3:
            # Init plt figure
            fig_pca = plt.figure(dpi=200)
            ax_pca = fig_pca.add_subplot(111, projection="3d")

            # Choose df["pca_X"]
            if pca_components >= 4:
                raise NotImplementedError()
            else:
                scatter_dict = {
                    "xs": df["pca_1"].values[rndperm],
                    "ys": df["pca_2"].values[rndperm],
                    "zs": df["pca_3"].values[rndperm],
                    "c": df["y"].values[rndperm],
                }

            # Plot Scatter to Ax
            ax_pca.scatter(
                xs=scatter_dict["xs"], ys=scatter_dict["ys"], zs=scatter_dict["zs"],
                c=scatter_dict["c"], cmap=cmap_name,
            )

            # Set Labels
            ax_pca.set_xlabel("pca_1")
            ax_pca.set_ylabel("pca_2")
            ax_pca.set_zlabel("pca_3")

            # Set Figure Title
            ax_pca.set_title("{}".format(self.video_name))

        # # Plot Show
        # plt.show()

        # Save Plot
        if is_save_mode:
            plt.savefig(
                os.path.join(save_base_path, "{}.{}".format(self.video_name, save_format))
            )
        else:
            plt.show()

        plt.close(fig_pca)

    def analyze_ffn_tsne(self, **kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    # FFN Data Path
    ffn_data_path = os.path.join(__FFN_DATA_ROOT_PATH__, __BENCHMARK_DATASET__)

    N_bench = len(os.listdir(ffn_data_path))
    iter_numbers = 25
    iter_interval = N_bench // iter_numbers

    iter_indices_arr = []
    for idx in range(iter_numbers):
        start_idx, end_idx = idx * iter_interval, min((idx + 1) * iter_interval, N_bench)
        curr_iter_indices = np.arange(start_idx, end_idx)
        iter_indices_arr.append(curr_iter_indices)

    # For every iteration, initialize FFN Analysis Objects
    for idx, iter_indices in enumerate(iter_indices_arr):
        # if min(iter_indices) != 36:
        #     continue

        # Initialize FFN Analysis Object
        FFN_ANALYSIS_OBJ = FFN_ANALYSIS_BENCHMARK_OBJ(
            root_path=ffn_data_path, benchmark=__BENCHMARK_DATASET__,
            load_video_indices=iter_indices,

            # Overlap-related Arguments
            overlap_criterion="iou", overlap_thresholds=[0.5],

            # Labeling-related Arguments
            labeling_type="one_hot", is_auto_labeling=True,

            # Drop(delete) hierarchical objects
            is_drop_hier_objs=True,
        )
        # FFN_ANALYSIS_OBJ.reinit()

        # Adjoin Save Base Path
        if __BENCHMARK_DATASET__ == "OTB100":
            save_base_path = os.path.join(
                __FFN_ANALYSIS_SAVE_PATH__, "003__otb_video_pca_analysis"
            )

        elif __BENCHMARK_DATASET__ == "UAV123":
            save_base_path = os.path.join(
                __FFN_ANALYSIS_SAVE_PATH__, "003__uav_video_pca_analysis"
            )

        else:
            raise NotImplementedError()

        # Analyze FFN PCA
        FFN_ANALYSIS_OBJ.analyze_ffn_pca(
            scatter_size=0.5,

            save_base_path=save_base_path, is_save_mode=True,
        )

        # Delete Current Iteration's FFN Analysis Objects (RAM Memory Issue)
        del FFN_ANALYSIS_OBJ

