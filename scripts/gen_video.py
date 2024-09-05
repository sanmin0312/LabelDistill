import os
from argparse import ArgumentParser

import cv2
import mmcv


def parse_args():
    parser = ArgumentParser(add_help=False)
    # parser.add_argument('idx',
    #                     type=int,
    #                     help='Index of the dataset to be visualized.')
    parser.add_argument('--result_path',
                        help='Path of the result json file.',
                        default='concat3_05_hres')
    parser.add_argument('--target_path',
                        help='Target path to save the visualization result.',
                        default='/home/user/data/SanminKim/BEVDepth/viz')

    args = parser.parse_args()
    return args


def gen_vid(
    nusc_results_file,
    dump_file,
):
    # Set cameras
    IMG_KEYS = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    # Get data from dataset
    infos = mmcv.load('data/nuScenes/nuscenes_infos_val.pkl')
    
    # Call all samples and Sort samples by scene tokens
    video_dict = dict()
    for info in infos:
        if info['scene_token'] not in video_dict:
            video_dict[info['scene_token']] = list()
        video_dict[info['scene_token']].append(info)

    # Sort samples in scenes by timestamps
    for scene_token, frame_ls in video_dict.items():
        sorted_frame_ls = sorted(frame_ls, key=lambda frame: frame['timestamp'])
        video_dict[scene_token] = sorted_frame_ls
    
    viz_dir = nusc_results_file
    
    valid_scene = [
        "01e4fcbe6e49483293ce45727152b36e",
        "54f56f80350b4c07af598ee87cf3886a",
        "16be583c31a2403caa6c158bb55ae616",
        "3363f396bb43405fbdd17d65dc123d4e",
        "5af9c7f124d84e7e9ac729fafa40ea01",
        "57dc3221a3d845b5ab17ff0f98ce336f",
        "7061c08f7eec4495979a0cf68ab6bb79",
        "c525507ee2ef4c6d8bb64b0e0cf0dd32",
        "c65c4acf86954f8cbd53a3541a3bfa3a",
        "b07358651c604e2d83da7c4d4755de73",
        "a178a1b5415f45c08d179bd2cacdf284",
        "93608f0d57794ba6b014314c488e2b4a",
        "0ac05652a4c44374998be876ba5cd6fd",
        "a7d073bc435b4356a0a9a5ebfb61f229",
        "85651af9c04945c3a394cf845cb480a6",
        "e8834785d9ff4783a5950281a4579943",
        "d29527ec841045d18d04a933e7a0afd2",
        "85889db3628342a482c5c0255e5347e9"
    ]
        
    # Check for image existence    
    for si, (scene_token, frame_ls) in enumerate(video_dict.items()):
        # images = []
        # if si == 50: break
        if scene_token not in valid_scene: continue
        video_path = os.path.join(dump_file, viz_dir, scene_token)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fps = 5
        # video = cv2.VideoWriter(video_path + f"/v_{fps}fps_{scene_token}.mp4", fourcc, fps, (2400, 800))
        print(f"{si:03d}/{len(video_dict)} {scene_token}:")
        # if not os.path.isdir(video_path):
        #     os.makedirs(video_path)
        for fi, info in enumerate(frame_ls):
            # img_path = dump_file + scene_token + f"/{fi:02d}_" + info['sample_token'] + ".png"
            # if (fi % 2) != 0: continue # downsample
            img_path = video_path + f"/{fi:02d}_{info['sample_token']}.png"
            assert os.path.isfile(img_path), f"{img_path} not exists"
            # if True:
            # break
            # video.write(cv2.imread(img_path))
        # break
    print('All Exist')
        
    # Generate video    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 5
    for si, (scene_token, frame_ls) in enumerate(video_dict.items()):
        if scene_token not in valid_scene: continue
        video_path = os.path.join(dump_file, viz_dir, scene_token)
        video = cv2.VideoWriter(video_path + f"/v_{fps}fps_{scene_token}x2.mp4", fourcc, fps, (2400, 2400))
        # video = cv2.VideoWriter(video_path + f"/tmp.avi", fourcc, fps, (1200, 1200))
        print(f"{si:03d}/{len(video_dict)} {scene_token}:")
        for fi, info in enumerate(frame_ls):
            img_path = video_path + f"/{fi:02d}_{info['sample_token']}.png"
            video.write(cv2.imread(img_path))
        video.release()
        # break
            

if __name__ == '__main__':
    args = parse_args()
    gen_vid(
        args.result_path,
        args.target_path
    )
    
