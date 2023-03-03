from __future__ import absolute_import, print_function

import os
import numpy as np
from tqdm import tqdm


__FFN_DATA_ROOT_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE/acmmm23_dev/ffn_data"
# __BENCHMARK_DATASET__ = "OTB100"
__BENCHMARK_DATASET__ = "UAV123"


if __name__ == "__main__":

    ffn_data_base_path = os.path.join(__FFN_DATA_ROOT_PATH__, __BENCHMARK_DATASET__)

    # Get Video Names
    video_names = sorted(os.listdir(ffn_data_base_path))
    ffn_data_paths = [os.path.join(ffn_data_base_path, vn) for vn in video_names]

    # Initialize tqdm iteration object
    tqdm_iter_obj = tqdm(
        ffn_data_paths, leave=True,
        desc="[{}] FFN numpy array splitting...".format(__BENCHMARK_DATASET__)
    )

    # Iterate for Video Paths
    for v_idx, ffn_data_path in enumerate(tqdm_iter_obj):
        # Get FFN Outputs Numpy File
        ffn_data_filepath = os.path.join(ffn_data_path, "ffn_outputs.npy")
        if os.path.isfile(ffn_data_filepath) is False:
            continue

        # Load Numpy Array
        ffn_outputs = np.load(ffn_data_filepath)

        # Make FFN outputs directory
        ffn_outputs_dir = os.path.join(ffn_data_path, "ffn_outputs")
        if os.path.isdir(ffn_outputs_dir) is False:
            os.makedirs(ffn_outputs_dir)

        # Iterate for Frame Indices
        for idx in range(ffn_outputs.shape[0]):
            # Compute Actual Frame Index
            fidx = idx + 1

            # Get "ffn_output" vector of index
            ffn_output = ffn_outputs[idx]

            # Save each ffn_output vector
            ffn_output_fname = "{:06d}.npy".format(fidx)
            np.save(os.path.join(ffn_outputs_dir, ffn_output_fname), ffn_output)

        # Delete FFN Outputs File
        os.remove(ffn_data_filepath)

        # Set PostFix
        tqdm_iter_obj.set_postfix({
            "Video": video_names[v_idx],
        })
