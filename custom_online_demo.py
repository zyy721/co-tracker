# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
# import imageio.v3 as iio
import numpy as np
import pickle as pkl
from PIL import Image
from tqdm import tqdm

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import gc

class NuscenesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# DEFAULT_DEVICE = (
#     "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--single-gpu", action='store_true')

    args = parser.parse_args()


    # a = list(range(20))


    torch.cuda.set_device(args.local_rank)
    if args.single_gpu:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=1, rank=0) 
    else:
        dist.init_process_group(backend='nccl') 
    device = torch.device("cuda", args.local_rank)

    # info_train_sparsepad_path = 'data/nuscenes_cam_w_plan/nuscenes_infos_train_sweeps_occ.pkl'
    # with open(info_train_sparsepad_path, "rb") as f:
    #     loaded_data = pkl.load(f)
    # infos = loaded_data['infos']
    # scene_tokens = infos.keys()
    # first_scene_token = list(scene_tokens)[0]
    # img_path_list, img_array_list = [], []

    # for cur_info in infos[first_scene_token][10:26]:
    #     # cur_img_path = os.path.join('data/nuscenes', cur_info['data']['CAM_FRONT']['filename'])
    #     cur_img_path = os.path.join('data/nuscenes', cur_info['data']['CAM_FRONT_LEFT']['filename'])

    #     cur_img = Image.open(cur_img_path)
    #     cur_img_array = np.array(cur_img)
    #     img_array_list.append(cur_img_array)

    with open('data/nuscenes_cam_w_plan/nuscenes_infos_train_sweeps_occ.pkl', "rb") as f:
        train_nusc_data = pkl.load(f)['infos']

    with open('data/nuscenes_cam_w_plan/nuscenes_infos_val_sweeps_occ.pkl', "rb") as f:
        val_nusc_data = pkl.load(f)['infos']

    nusc_data = {**train_nusc_data, **val_nusc_data}

    with open('data/nuscenes/nuscenes_unified_infos_train_v4.pkl', "rb") as f:
        train_loaded_list = pkl.load(f)
    train_infos = train_loaded_list['infos']

    with open('data/nuscenes/nuscenes_unified_infos_val_v4.pkl', "rb") as f:
        val_loaded_list = pkl.load(f)
    val_infos = val_loaded_list['infos']

    train_val_infos = {}
    for cur_info in train_infos + val_infos:
        cur_token = cur_info['token']
        train_val_infos[cur_token] = cur_info


    nusc_dataset = NuscenesDataset(list(range(len(nusc_data))))
    train_sampler = DistributedSampler(nusc_dataset, shuffle=False)
    trainloader = DataLoader(nusc_dataset, batch_size=1, num_workers=4, sampler=train_sampler)


    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    # model = model.to(DEFAULT_DEVICE)
    model = model.to(device)

    # window_frames = []

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                # np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
                np.stack(window_frames[-model.step * 2 :]), device=device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    save_path = 'data/co-tracker'
    camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

    scene_name_list = list(nusc_data.keys())

    for index_data in tqdm(trainloader):

        scene = scene_name_list[index_data]
        for cam in camera_names:
            img_array_list, sample_token_list, sample_idx_list = [], [], []
            cur_save_tracks_path = os.path.join(save_path, 'tracks', cam)
            cur_save_visibility_path = os.path.join(save_path, 'visibility', cam)

            if not os.path.exists(os.path.join(cur_save_tracks_path, scene)):
                os.makedirs(os.path.join(cur_save_tracks_path, scene), exist_ok=True) 

            if not os.path.exists(os.path.join(cur_save_visibility_path, scene)):
                os.makedirs(os.path.join(cur_save_visibility_path, scene), exist_ok=True) 

            for sample_idx, sample in enumerate(nusc_data[scene]):
                cur_img_path = sample['data'][cam]['filename']
                cur_img = Image.open(os.path.join('data/nuscenes', cur_img_path))
                cur_img_array = np.array(cur_img)
                img_array_list.append(cur_img_array)

                if sample['is_key_frame']:
                    token = sample['token']
                    sample_token_list.append(token)
                    sample_idx_list.append(sample_idx)

            for token, sample_idx in zip(sample_token_list, sample_idx_list):
                if sample_idx > 0:
                    start_idx = sample_idx - 9 if sample_idx>=9 else 0
                    end_idx = sample_idx + 7
                    cur_img_array_list = img_array_list[start_idx:end_idx]
                    # cur_img_path_list = img_path_list[start_idx:end_idx]

                    grid_query_frame = 8 if sample_idx>=9 else sample_idx-1

                    window_frames = []

                    # Iterating over video frames, processing one window at a time:
                    is_first_step = True
                    for i, frame in enumerate(
                        # iio.imiter(
                        #     args.video_path,
                        #     plugin="FFMPEG",
                        # )
                        cur_img_array_list
                        # cur_img_path_list
                    ):

                        # frame = Image.open(os.path.join('data/nuscenes', frame_path))
                        # frame = np.array(frame)

                        if i % model.step == 0 and i != 0:
                            pred_tracks, pred_visibility = _process_step(
                                window_frames,
                                is_first_step,
                                grid_size=args.grid_size,
                                # grid_query_frame=args.grid_query_frame,
                                grid_query_frame=grid_query_frame,
                            )
                            is_first_step = False
                        window_frames.append(frame)
                    # Processing the final video frames in case video length is not a multiple of model.step
                    pred_tracks, pred_visibility = _process_step(
                        window_frames[-(i % model.step) - model.step - 1 :],
                        is_first_step,
                        grid_size=args.grid_size,
                        # grid_query_frame=args.grid_query_frame,
                        grid_query_frame=grid_query_frame,
                    )

                    # print("Tracks are computed")

                    # Store the tracks
                    save_pred_tracks = pred_tracks[:, grid_query_frame-1:grid_query_frame+2]
                    save_pred_visibility = pred_visibility[:, grid_query_frame-1:grid_query_frame+2]
                    # save_pred_visibility = torch.all(save_pred_visibility, dim=1)
                    # save_pred_tracks = save_pred_tracks[:, :, save_pred_visibility[0]]

                    np.save(os.path.join(cur_save_tracks_path, scene, f'{token}.npy'), save_pred_tracks.cpu().numpy().astype(np.float16))
                    np.save(os.path.join(cur_save_visibility_path, scene, f'{token}.npy'), save_pred_visibility.cpu().numpy())

                    # # save a video with predicted tracks
                    # seq_name = args.video_path.split("/")[-1]
                    # # video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
                    # #     0, 3, 1, 2
                    # # )[None]
                    # video = torch.tensor(np.stack(window_frames), device=device).permute(
                    #     0, 3, 1, 2
                    # )[None]
                    # vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
                    # # vis.visualize(
                    # #     video, pred_tracks, pred_visibility, filename="nusc_front", query_frame=args.grid_query_frame
                    # # )
                    # vis.visualize(
                    #     video, pred_tracks, pred_visibility, filename="nusc_front", query_frame=grid_query_frame
                    # )

                    # print()

        del img_array_list
        del window_frames
        gc.collect()
        torch.cuda.empty_cache()


        # for sample in nusc_data[scene]:
        #     if sample['is_key_frame']:
        #         token = sample['token']

        #         if not os.path.exists(os.path.join(save_path, scene)):
        #             os.makedirs(os.path.join(save_path, scene), exist_ok=True) 


        #         cur_cam_sweeps_info = train_val_infos[token]['cam_sweeps_info']

        #         if len(cur_cam_sweeps_info['CAM_FRONT']) > 0:

        #             if os.path.exists(os.path.join(save_path, scene, f'{token}.npy')):
        #                 if not args.overwrite:
        #                     continue

        #             # for i_sweep in range(1,3):

        #             for cam in camera_names:

        #                 img_path_list, img_array_list = [], []

        #                 for cur_img_path_info in cur_cam_sweeps_info[cam]:

        #                     cur_img = Image.open(cur_img_path_info['data_path'])
        #                     cur_img_array = np.array(cur_img)
        #                     img_array_list.append(cur_img_array)

        #                 img_array_list.reverse()
        #                 grid_query_frame = len(img_array_list) - 1

        #                 # Iterating over video frames, processing one window at a time:
        #                 is_first_step = True
        #                 # frame 720,1296,3
        #                 for i, frame in enumerate(
        #                     # iio.imiter(
        #                     #     args.video_path,
        #                     #     plugin="FFMPEG",
        #                     # )
        #                     img_array_list
        #                 ):
        #                     if i % model.step == 0 and i != 0:
        #                         pred_tracks, pred_visibility = _process_step(
        #                             window_frames,
        #                             is_first_step,
        #                             grid_size=args.grid_size,
        #                             # grid_query_frame=args.grid_query_frame,
        #                             grid_query_frame=grid_query_frame,
        #                         )
        #                         is_first_step = False
        #                     window_frames.append(frame)
        #                 # Processing the final video frames in case video length is not a multiple of model.step
        #                 pred_tracks, pred_visibility = _process_step(
        #                     window_frames[-(i % model.step) - model.step - 1 :],
        #                     is_first_step,
        #                     grid_size=args.grid_size,
        #                     # grid_query_frame=args.grid_query_frame,
        #                     grid_query_frame=grid_query_frame,
        #                 )

        #                 print("Tracks are computed")

        #                 # save a video with predicted tracks
        #                 seq_name = args.video_path.split("/")[-1]
        #                 # video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        #                 #     0, 3, 1, 2
        #                 # )[None]
        #                 video = torch.tensor(np.stack(window_frames), device=device).permute(
        #                     0, 3, 1, 2
        #                 )[None]
        #                 vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
        #                 # vis.visualize(
        #                 #     video, pred_tracks, pred_visibility, filename="nusc_front", query_frame=args.grid_query_frame
        #                 # )
        #                 vis.visualize(
        #                     video, pred_tracks, pred_visibility, filename="nusc_front", query_frame=grid_query_frame
        #                 )