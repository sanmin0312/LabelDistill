from nuscenes import NuScenes
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Moving Object Evaluation")
    parser.add_argument("--root", type=str, default="./data/nuScenes")
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--exp_config", type=str, default='LiDARandLabelDistill_r50_128x128_e24/lightning_logs/version_29')

    args = parser.parse_args()

    return args

def eval_tracking():
    args = parse_args()
    work_dir = os.getcwd()
    eval(os.path.join(work_dir, 'outputs',args.exp_config, 'results_nusc.json'),
         "val",
         work_dir,
         args.root
         )
    # eval(os.path.join(work_dir, 'outputs', 'bev_stereo', args.exp_config, 'lightning_logs','version_0', 'results_nusc.json'),
    #      "val",
    #      work_dir,
    #      args.root
    #      )

def eval(res_path, eval_set="val", output_dir=None, root_path=None):
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.common.config import config_factory as track_configs

    cfg = track_configs("detection_cvpr_2019")

    nusc_ = NuScenes(version="v1.0-trainval", dataroot=root_path)

    nusc_eval = DetectionEval(nusc_,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )

    metrics_summary = nusc_eval.main(render_curves=True)


if __name__ == '__main__':
    eval_tracking()
