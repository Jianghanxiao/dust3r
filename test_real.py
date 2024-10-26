from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import json
import numpy as np
import torch
import glob
import os
import pickle
from tqdm import tqdm

if not os.path.exists("results"):
    os.mkdir("results")

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect"
case_name = "rope_double_hand"

if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsics"])
    WH = data["WH"]
    frame_num = data["frame_num"]
    num_cam = len(intrinsics)
    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))

    for index in tqdm(range(frame_num)):
        image_paths = []
        camera_poses = []
        focals = []
        for i in range(num_cam):
            image_paths.append(f"{base_path}/{case_name}/color/{i}/{index}.png")
            camera_poses.append(c2ws[i])
            focals.append(intrinsics[i][0, 0])

        device = "cuda"
        batch_size = 1
        schedule = "cosine"
        lr = 0.01
        niter = 300

        model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

        images = load_images(image_paths, size=512)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        scene = global_aligner(
            output,
            device=device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            min_conf_thr=3,
        )
        scene.preset_pose(torch.tensor(camera_poses, device=device))
        scene.preset_focal(np.array(focals))

        loss = scene.compute_global_alignment(
            init="known_poses", niter=niter, schedule=schedule, lr=lr
        )
        # retrieve useful values from scene:
        imgs = scene.imgs
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()
        pp = scene.get_principal_points()

        np.savez(
            f"results/points_colors_{index}.npz",
            # f"test.npz",
            pts3d=torch.stack(pts3d).detach().cpu().numpy(),
            imgs=np.stack(imgs),
            confidence_masks=torch.stack(confidence_masks).detach().cpu().numpy(),
            focals=focals.detach().cpu().numpy(),
            poses=poses.detach().cpu().numpy(),
            pp=pp.detach().cpu().numpy(),
        )
