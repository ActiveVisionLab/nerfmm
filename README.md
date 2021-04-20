# NeRF--: Neural Radiance Fields Without Known Camera Parameters

**[Project Page](https://nerfmm.active.vision/) | [Arxiv](https://arxiv.org/abs/2102.07064) | [Colab Notebook](https://colab.research.google.com/drive/1pRljG5lYj_dgNG_sMRyH2EKbpq3OezvK?usp=sharing) | [Data](https://www.robots.ox.ac.uk/~ryan/nerfmm2021/nerfmm_release_data.tar.gz)**

Zirui Wang¹, 
[Shangzhe Wu²](http://elliottwu.com), 
[Weidi Xie²](https://weidixie.github.io/weidi-personal-webpage/), 
[Min Chen³](https://sites.google.com/site/drminchen/home), 
[Victor Adrian Prisacariu¹](http://www.robots.ox.ac.uk/~victor/). 

¹Active Vision Lab + ²Visual Geometry Group + ³e-Research Centre, University of Oxford.

## Overview
We provide 3 training targets in this repository, under the `tasks` directory:
1. `task/nerfmm/train.py`: This is our main training script for the [NeRF-LLFF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), which estimates camera poses, focal lenghts and a NeRF jointly and monitors the absolute trajectory error (ATE) between our estimation of camera parameters and COLMAP estimation during training. This target can also start training from a COLMAP initialisation and refine the COLMAP camera parameters.
2. `task/refine_nerfmm/train.py`: This is the training script that refines a pretrained nerfmm system.
3. `task/any_folder/train.py`: This is a training script that takes a folder that contains forward-facing images and trains with our nerfmm system without making any comparison with COLMAP. It is similar to what we offer in our [CoLab notebook](TODO) and we treat this `any_folder` target as a **playgraound**, where users can try novel view synthesis by just providing an image folder and do not care how the camera parameter estimation compares with COLMAP. 

For each target, we provide relevant utilities to evaluate our system. Specifically, 
- for the `nerfmm` target, we provide three utility files:
    - `eval.py` to evaluate image rendering quality on validation splits with PSNR, SSIM and LPIPS, i.e, results in Table 1.
    - `spiral.py` to render novel views using a spiral camera trajectory, i.e. results in Figure 1.
    - `vis_learned_poses.py` to visualise our camera parameter estimation with COLMAP estimation in 3D. It also computes ATE between them, i.e. E1 in Table 2.
- for the `refine_nerfmm` target, all utilities in `nerfmm` target above are compatible with `refine_nerfmm` target, since it just refines a pretrained nerfmm system.
- for the `any_folder` target, it has its own `spiral.py` and `vis_learned_poses.py` utilities, as it does not compare with COLMAP. It does not have a `eval.py` file as this target is treated as a playground and does not split images to train/validation sets. It only provides novel view synthesis results via the `spiral.py` file.
    
---

## Table of Content
- [Environment](#Environment)
- [Get Data](#Get-Data)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Citation](#citation)

## Environment

We provide a `requirement.yml` file to set up a `conda` environment:

```sh
git clone https://github.com/ActiveVisionLab/nerfmm.git
cd nerfmm
conda env create -f environment.yml
```

Generally, our code should be able to run with any `pytorch >= 1.1` .

(Optional) Install `open3d` for visualisation. You might need a physical monitor to install this lib.
```sh
pip install open3d
```

## Get Data
We use the [NeRF-LLFF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) with two small structural changes: 
1. We remove their `image_4` and `image_8` folder and downsample images to any desirable resolution during data loading `dataloader/with_colmap.py`, by calling PyTorch's `interpolate` function.
2. We explicitly generate two txt files for train/val image ids. i.e. take every 8th image as the validation set, as in the official NeRF train/val split. The only difference is that we store them as txt files while NeRF split them during data loading. The file produces these two txt files is `utils/split_dataset.py`.

In addition to the NeRF-LLFF dataset, we provide two demo scenes to demonstrate how to use the `any_folder` target.
   
We pack the re-structured LLFF data and our data to a tar ball (~1.8G), to get it, run:
```shell
wget https://www.robots.ox.ac.uk/~ryan/nerfmm2021/nerfmm_release_data.tar.gz
```

Untar the data:
```
tar -xzvf path/to/the/tar.gz
```

## Training
We show how to:
1. train a `nerfmm` from scratch, i.e. initialise camera poses with identity matrices and focal lengths with image resolution:
    ```shell
   python tasks/nerf/train.py \
   --base_dir='path/to/nerfmm_release/data' \
   --scene_name='LLFF/fern'
    ```
2. train a `nerfmm` from COLMAP initialisation:
    ```shell
    python tasks/nerf/train.py \
   --base_dir='path/to/nerfmm_release/data' \
   --scene_name='LLFF/fern' \
   --start_refine_pose_epoch=1000 \
   --start_refine_focal_epoch=1000
    ```
   This command initialises a `nerfmm` target with COLMAP parameters, trains with them for 1000 epochs, and starts refining those parameters after 1000 epochs. 
3. train a `nerfmm` from a pretrained `nerfmm`:
    ```shell
    python tasks/refine_nerfmm/train.py \
   --base_dir='path/to/nerfmm_release/data' \
   --scene_name='LLFF/fern' --start_refine_epoch=1000 \
   --ckpt_dir='path/to/a/dir/contains/nerfmm/ckpts'
    ```
   This command initialises a `refine_nerfmm` target with a set of pretrained `nerfmm` parameters, trains with them for 1000 epochs, and starts refining those parameters after 1000 epochs.
4. train an `any_folder` from scratch given an image folder:
    ```shell
    python tasks/any_folder/train.py \
   --base_dir='path/to/nerfmm_release/data' \
   --scene_name='any_folder_demo/desk'
    ```
   This command trains an `any_folder` target using a provided demo scene `desk`. 

(Optional) set a symlink to the downloaded data:
```shell
mkdir data_dir  # do it in this nerfmm repo
cd data_dir
ln -s /path/to/downloaded/data ./nerfmm_release_data
cd ..
```
this can simplify the above training commands, for example:
```shell
python tasks/nerfmm/train.py
```

## Evaluation
### Compute image quality metrics
Call `eval.py` in `nerfmm` target:
```shell
python tasks/nerfmm/eval.py \
--base_dir='path/to/nerfmm_release/data' \
--scene_name='LLFF/fern' \
--ckpt_dir='path/to/a/dir/contains/nerfmm/ckpts'
```
This file can be used to evaluate a checkpoint trained with `refine_nerfmm` target. For some scenes, you might need to tweak with `--opt_eval_lr` option to get the best results. Common values for `opt_eval_lr` are 0.01 / 0.005 / 0.001 / 0.0005 / 0.0001. The default value is 0.001. Overall, it finds validation poses that can produce highest PSNR on validation set while freezing NeRF and focal lengths. We do this because the learned camera pose space is different from the COLMAP estimated camera pose space.

### Render novel views
Call `spiral.py` in each target. The `spiral.py` in `nerfmm` is compatible with `refine_nerfmm` target:
```shell
python spiral.py \
--base_dir='path/to/nerfmm_release/data' \
--scene_name='LLFF/fern' \
--ckpt_dir='path/to/a/dir/contains/nerfmm/ckpts'
```

### Visualise estimated poses in 3D
Call `vis_learned_poses.py` in each target. The `vis_learned_poses.py` in `nerfmm` is compatible with `refine_nerfmm` target:
```shell
python spiral.py \
--base_dir='path/to/nerfmm_release/data' \
--scene_name='LLFF/fern' \
--ckpt_dir='path/to/a/dir/contains/nerfmm/ckpts'
```

---

## Acknowledgement
[Shangzhe Wu](http://elliottwu.com) is supported by Facebook Research. [Weidi Xie](https://weidixie.github.io/weidi-personal-webpage/) is supported by Visual AI (EP/T028572/1).

The authors would like to thank
[Tim Yuqing Tang](https://scholar.google.co.uk/citations?user=kQB_dOoAAAAJ&hl=en) for insightful discussions and proofreading.
 
During our NeRF implementation, we referenced several open sourced NeRF implementations, and we thank their contributions. Specifically, we referenced functions from [nerf](https://github.com/bmild/nerf) and [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), and borrowed/modified code from [nerfplusplus](https://github.com/Kai-46/nerfplusplus) and [nerf_pl](https://github.com/kwea123/nerf_pl). We especially appreciate the detailed code comments and git issue answers in [nerf_pl](https://github.com/kwea123/nerf_pl).

## Citation
```
@article{wang2021nerfmm,
  title={Ne{RF}$--$: Neural Radiance Fields Without Known Camera Parameters},
  author={Zirui Wang and Shangzhe Wu and Weidi Xie and Min Chen and Victor Adrian Prisacariu},
  journal={arXiv preprint arXiv:2102.07064},
  year={2021}
}
```