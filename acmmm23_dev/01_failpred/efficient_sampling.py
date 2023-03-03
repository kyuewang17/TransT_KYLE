from __future__ import absolute_import, print_function

import os
import psutil
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from data_obj import BENCHMARK_FFN_OBJECT, VIDEO_FFN_OBJECT


__FFN_DATA_ROOT_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE/acmmm23_dev/ffn_data"
__BENCHMARK_DATASET__ = "OTB100"
# __BENCHMARK_DATASET__ = "UAV123"

# FFN Analysis Save Path
__FFN_ANALYSIS_SAVE_PATH__ = "/home/kyle/Dropbox/SNU/Research/ACM_MM23/230214/ffn_analysis"


class EFFICIENT_BENCHMARK_FFN_OBJECT(BENCHMARK_FFN_OBJECT):
    def __init__(self, root_path, benchmark, overlap_criterion, **kwargs):
        super().__init__(root_path, benchmark, overlap_criterion, **kwargs)

    def reinit(self):
        assert len(self.video_ffn_objs) > 0
        new_video_ffn_objs = []
        for video_ffn_obj in self.video_ffn_objs:
            _OBJ = EFFICIENT_VIDEO_FFN_OBJECT(
                video_name=video_ffn_obj.video_name, video_path=video_ffn_obj.video_path,
                benchmark=video_ffn_obj.benchmark,
                overlap_criterion=video_ffn_obj.overlap_criterion,
                overlap_thresholds=video_ffn_obj.overlap_thresholds,
                labeling_type=video_ffn_obj.labeling_type,
            )
            _OBJ.set_label(labeling_type=video_ffn_obj.labeling_type)
            new_video_ffn_objs.append(_OBJ)
        self.video_ffn_objs = new_video_ffn_objs

    def analyze_nearby_frame_fusion_vectors(self, video_idx, nearby_frame_window):
        assert video_idx in self.video_indices
        assert isinstance(nearby_frame_window, int) and nearby_frame_window > 0

        # Analyze for Selected Video Index
        self.video_ffn_objs[video_idx].analyze_nearby_frame_fusion_vectors(
            nearby_frame_window=nearby_frame_window
        )


class EFFICIENT_VIDEO_FFN_OBJECT(VIDEO_FFN_OBJECT):
    def __init__(self, video_name, video_path, overlap_criterion, **kwargs):
        super().__init__(video_name, video_path, overlap_criterion, **kwargs)

    def analyze_nearby_frame_fusion_vectors(self, nearby_frame_window):
        assert isinstance(nearby_frame_window, int) and nearby_frame_window > 0

        # Get FFN outputs
        ffn_outputs = self.get_ffn_outputs(include_init=True)

        # Draw Figure
        fig_nearby = plt.figure(dpi=200)
        ax_nearby = fig_nearby.add_subplot(111)

        # Iterate for Range of "nearby_frame_window"
        cnt = 0
        start_fidx = 0
        window_data_dict = {}
        while True:
            # Get Overlaps and Labels
            overlaps = self.overlaps[start_fidx:start_fidx+nearby_frame_window]
            labels = self.labels[start_fidx:start_fidx+nearby_frame_window]

            # Current Window FFN Outputs
            c_w_ffn_outputs = ffn_outputs[start_fidx:start_fidx+nearby_frame_window]
            if start_fidx == 0:
                c_w_ffn_outputs = c_w_ffn_outputs[1:]
                overlaps, labels = overlaps[1:], labels[1:]

            # FFN Cosine Similarity Matrix
            flat_c_w_ffn_outputs = c_w_ffn_outputs.reshape(c_w_ffn_outputs.shape[0], -1)
            ffn_norm_mat = np.linalg.norm(flat_c_w_ffn_outputs, ord=2, axis=1)
            ffn_norm_mat_square = \
                np.dot(ffn_norm_mat.reshape(-1, 1), ffn_norm_mat.reshape(1, -1))
            cos_sim_mat = np.matmul(flat_c_w_ffn_outputs, flat_c_w_ffn_outputs.T)
            cos_sim_mat = np.divide(cos_sim_mat, ffn_norm_mat_square)
            c_w_cos_sim_vec = cos_sim_mat[0]

            # Plot
            ax_nearby.plot(
                c_w_cos_sim_vec, overlaps, "bo"
            )

            # Set Window Data
            window_data_dict[cnt] = {
                "start_fidx": start_fidx if start_fidx > 0 else 1,
                "end_fidx": min(start_fidx + nearby_frame_window - 1, ffn_outputs.shape[0]-1),
                "overlaps": overlaps, "c_w_cos_sim_vec": c_w_cos_sim_vec,
            }

            # Add Window Number to Starting Frame Index
            start_fidx += nearby_frame_window

            # Increase Counter
            cnt += 1

            # Break Condition
            if start_fidx >= ffn_outputs.shape[0]:
                break

        # Show Plot
        plt.show()

        # Accumulative Plot for "overlaps"(y) and "c_w_cos_sim_vec"(x)


        print(123)


if __name__ == "__main__":
    # FFN Data Path
    ffn_data_path = os.path.join(__FFN_DATA_ROOT_PATH__, __BENCHMARK_DATASET__)

    # Initialize Efficient FFN Analysis Object
    EFF_FFN_ANALYSIS_OBJ = EFFICIENT_BENCHMARK_FFN_OBJECT(
        root_path=ffn_data_path, benchmark=__BENCHMARK_DATASET__,
        load_video_indices=np.arange(0, 10),

        # Overlap-related Arguments
        overlap_criterion="iou", overlap_thresholds=[0.5],

        # Labeling-related Arguments
        labeling_type="one_hot", is_auto_labeling=True,

        # Drop(delete) hierarchical objects
        # is_drop_hier_objs=True,
    )
    EFF_FFN_ANALYSIS_OBJ.reinit()

    # Analyze Nearby Frame Fusion Vectors
    EFF_FFN_ANALYSIS_OBJ.analyze_nearby_frame_fusion_vectors(
        video_idx=0, nearby_frame_window=100
    )

    print(123)
