# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import argparse
import os
from tqdm import tqdm
import psutil

import cv2
import torch
import numpy as np

from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.bbox import IoU, center2corner, rect_2_cxy_wh
from pysot_toolkit.toolkit.datasets import DatasetFactory
from pysot_toolkit.toolkit.utils.region import vot_overlap, vot_float2str
from pysot_toolkit.trackers.tracker import Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone

parser = argparse.ArgumentParser(description='transt tracking')
parser.add_argument('--dataset', type=str, help='datasets')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--vis', action='store_true', help='whether visualzie result')
parser.add_argument('--save', action='store_true', help='whether to save tracking result')
parser.add_argument('--save_ffn_feats', action='store_true', help='whether to save ffn features and bboxes')

parser.add_argument('--name', default='', type=str, help='name of results')
args = parser.parse_args()

torch.set_num_threads(1)

# Set Project Root Path
__PROJECT_ROOT_PATH__ = "/home/kyle/PycharmProjects/TransT_KYLE"


def main():
    # load config

    # Set Dataset Root Path and Network Path, conditionally
    dataset_base_path = os.path.join(__PROJECT_ROOT_PATH__, "datasets")
    net_base_path = "/home/kyle/PycharmProjects/TransT_KYLE/pytracking/networks"
    if args.dataset in ["CVPR13", "OTB50", "OTB100"]:
        dataset_root = os.path.join(dataset_base_path, "OTB100")
        net_path = os.path.join(net_base_path, "transt.pth")
        ffn_data_path = os.path.join(__PROJECT_ROOT_PATH__, "acmmm23_dev", "ffn_data", "OTB100")
    elif args.dataset == "UAV123":
        dataset_root = os.path.join(dataset_base_path, "UAV123", "data_seq", "UAV123")
        net_path = os.path.join(net_base_path, "transt.pth")
        ffn_data_path = os.path.join(__PROJECT_ROOT_PATH__, "acmmm23_dev", "ffn_data", "UAV123")
    elif args.dataset == "GOT-10k":
        dataset_root = os.path.join(dataset_base_path, "GOT-10k", "test")
        net_path = os.path.join(net_base_path, "TransT_GOT.pth")
        ffn_data_path = os.path.join(__PROJECT_ROOT_PATH__, "acmmm23_dev", "ffn_data", "GOT-10k")
    elif args.dataset == "LaSOT":
        dataset_root = os.path.join(dataset_base_path, "LaSOT")
        net_path = os.path.join(net_base_path, "transt.pth")
        ffn_data_path = os.path.join(__PROJECT_ROOT_PATH__, "acmmm23_dev", "ffn_data", "LaSOT")
    else:
        raise NotImplementedError()

    # Check if "ffn_data_path" exists, and make if not
    if os.path.isdir(ffn_data_path) is False:
        os.makedirs(ffn_data_path)

    # create model
    net = NetWithBackbone(net_path=net_path, use_gpu=True)
    tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root, load_img=False)

    model_name = args.name
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # noqa

                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount() # noqa
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-w/2, cy-h/2, w, h]
                    init_info = {'init_bbox':gt_bbox_}
                    tracker.initialize(img, init_info)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    info = {}
                    outputs = tracker.track(img, info)
                    pred_bbox = outputs['target_bbox']
                    # if cfg.MASK.MASK:
                    #     pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic # noqa
                if idx == 0:
                    cv2.destroyAllWindows() # noqa
                if args.vis and idx > frame_counter:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # noqa
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3) # noqa
                    bbox = list(map(int, pred_bbox)) # noqa
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3) # noqa
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # noqa
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # noqa
                    cv2.imshow(video.name, img) # noqa
                    if cv2.waitKey() & 0xFF == ord('q'): # noqa
                        break
            toc /= cv2.getTickFrequency() # noqa
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    # === Non-VOTs === #
    # (OTB, UAV123, GOT-10k, LaSOT, TrackingNet, etc.)
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            video_ffn_analysis = {"gt_bboxes": [], "trk_bboxes": [], "ffn_outputs": []}

            # Declare tqdm iteration object
            video_tqdm_iter = tqdm(
                video, desc="({}/{}) Running Video [{}]".format(v_idx+1, len(dataset), video.name),
                leave=True, total=len(video)-1
            )

            for idx, (img, gt_bbox) in enumerate(video_tqdm_iter):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # noqa
                tic = cv2.getTickCount() # noqa
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-w/2, cy-h/2, w, h]
                    init_info = {'init_bbox':gt_bbox_}
                    tracker.initialize(img, init_info)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                    # Append "None" for first frame FFN output
                    if args.save_ffn_feats:
                        video_ffn_analysis["ffn_outputs"].append(None)
                        video_ffn_analysis["gt_bboxes"].append(np.array(center2corner([cx, cy, w, h])))
                        video_ffn_analysis["trk_bboxes"].append(np.array(center2corner([cx, cy, w, h])))
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['target_bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                    # Convert "gt_bbox"
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    # Append FFN output for this frame
                    if args.save_ffn_feats:
                        video_ffn_analysis["ffn_outputs"].append(outputs["FFN_output"].squeeze().cpu().numpy())
                        video_ffn_analysis["gt_bboxes"].append(np.array(center2corner([cx, cy, w, h])))
                        video_ffn_analysis["trk_bboxes"].append(np.array(center2corner(np.concatenate(rect_2_cxy_wh(pred_bbox)))))
                toc += cv2.getTickCount() - tic # noqa
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency()) # noqa
                if idx == 0:
                    cv2.destroyAllWindows() # noqa
                if args.vis and idx > 0:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # noqa
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3) # noqa
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3) # noqa
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # noqa
                    cv2.imshow(video.name, img) # noqa
                    cv2.waitKey(1) # noqa

                # Compute Memory Percent
                curr_memory_percent = psutil.virtual_memory().percent

                # Set tqdm postfix
                video_tqdm_iter.set_postfix({
                    "FIDX": "({}/{})".format(idx+1, len(video)),
                    "RAM Memory": "{:.2f}%".format(curr_memory_percent),
                })

            # Toc
            toc /= cv2.getTickFrequency() # noqa

            if args.save_ffn_feats:
                # Stack(concat) Saving Variables
                gt_bboxes = np.vstack(video_ffn_analysis["gt_bboxes"])
                trk_bboxes = np.vstack(video_ffn_analysis["trk_bboxes"])
                ffn_outputs = np.dstack(video_ffn_analysis["ffn_outputs"][1:]).transpose(2, 0, 1)

                # Make Directory for Current Video
                curr_video_ffn_data_path = os.path.join(ffn_data_path, video.name)
                if os.path.isdir(curr_video_ffn_data_path) is False:
                    os.makedirs(curr_video_ffn_data_path)

                # Save *.npy files
                np.save(os.path.join(curr_video_ffn_data_path, "gt_bboxes.npy"), gt_bboxes)
                np.save(os.path.join(curr_video_ffn_data_path, "trk_bboxes.npy"), trk_bboxes)
                np.save(os.path.join(curr_video_ffn_data_path, "ffn_outputs.npy"), ffn_outputs)

            # Close tqdm iteration object
            video_tqdm_iter.close()

            # === Save Results === #
            if args.save:
                if 'VOT2018-LT' == args.dataset:
                    video_path = os.path.join('results', args.dataset, model_name, 'longterm', video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x])+'\n')
                    result_path = os.path.join(video_path, '{}_001_confidence.value'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in scores:
                            f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                    result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                elif 'GOT-10k' == args.dataset:
                    video_path = os.path.join('results', args.dataset, model_name, video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x])+'\n')
                    result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                else:
                    model_path = os.path.join('results', args.dataset, model_name)
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)
                    result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x])+'\n')
                print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
