# MagFace
MagFace: A Universal Representation for Face Recognition and Quality Assessment  
in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, **Oral** presentation.

![magface](raw/magface.png)

**Project Page**: https://irvingmeng.github.io/projects/magface/

**Paper**: [arXiv](https://arxiv.org/abs/2103.06627)

**知乎解读**: [https://zhuanlan.zhihu.com/p/475775106](https://zhuanlan.zhihu.com/p/475775106)

**A toy example**: [examples.ipynb](inference/examples.ipynb)

**Poster**: [GoogleDrive](https://drive.google.com/file/d/1S0hoQNDJC_H8b8ryuYyF7xjVLMorlBu1/view?usp=sharing), [BaiduDrive](https://pan.baidu.com/s/1Ji1fRtwfTzwm9egWGtarWQ) code: dt9e

**Beamer**: [GoogleDrive](https://drive.google.com/file/d/1MPj_ghD7c1igA_fe20ooMbOcD-OsK0jC/view?usp=sharing), [BaiduDrive](https://pan.baidu.com/s/1wt9eqCbn6forcoAz1ZVrAw), code: c16b

**Presentation**: 
  1. CVPR [5-minute presentation](https://www.bilibili.com/video/BV1Jq4y1j7ZH).
  2. Will release a detailed version later.

**NOTE**: This repo is forked from the original version of magface


## Installation
```bash
conda env create -f environment.yml
conda activate magface
```

## Face Recognition with OpenCV

This application uses OpenCV to capture images from your webcam. When you press the `1` key, the current frame is saved if a face is detected. The application then starts recognizing faces in subsequent frames. Release the saved face by press the `2` key.
1. Navigate to the project directory
```bash
  cd eval/eval_cam_recognition
```
2. Run the script
  - if using GPU
```bash
  python cam.py --resume {your model path}
```
  - if using CPU
```bash
  python cam.py --resume {your model path} --cpu_mode
```

## Face Capture with OpenCV
This application uses OpenCV to capture images from your webcam. It saves the captured face image and the embedded vector from the model in a local dataframe. The script then uses t-SNE for 2D visualization plot.

1. Navigate to the project directory
```bash
  cd eval/eval_cam_recognition
```
2. Run the script The dataframe will be saved in `{root}/df/`
  - if using GPU
```bash
  python capture.py --resume {your model path}
```
  - if using CPU
```bash
  python capture.py --resume {your model path} --cpu_mode
```
3. Run the notebook
- Navigate to `{root}/notebook/tsne.ipynb` to reduce dimensions.
4. Run the plot script
```bash
  python plot.py
```

## Model Zoo

| Parallel Method | Loss | Backbone | Dataset | Split FC? | Model | Log File |
| --- | --- | --- | --- | --- | --- | --- |
| DDP | MagFace | iResNet100 | MS1MV2 | Yes | [GoogleDrive](https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/view?usp=sharing), [BaiduDrive](https://pan.baidu.com/s/15iKz3wv6UhKmPGR6ltK4AA) code: wsw3 | **Trained by original codes** |
| DDP | MagFace | iResNet50 | MS1MV2 | Yes | [GoogleDrive](https://drive.google.com/file/d/1QPNOviu_A8YDk9Rxe8hgMIXvDKzh6JMG/view?usp=sharing), [BaiduDrive](https://pan.baidu.com/s/19FjwUyuPCTzLhGm3fvyPlw) code: idkx| [BaiduDrive](https://pan.baidu.com/s/1MGAmhtOangqr8nHxIFmNvg), code: 66j1 |
| DDP | Mag-CosFace | iResNet50 | MS1MV2 | Yes | [BaiduDrive](https://pan.baidu.com/s/1wZOanpWKealVd-4cMAu_tQ) code: rg2w| [BaiduDrive](https://pan.baidu.com/s/10EQjRydQLJMAU98q7lH10w), code: ejec |
| DP | MagFace | iResNet50 | MS1MV2 | No | [BaiduDrive](https://pan.baidu.com/s/1atuZZDkcCX3Bl14J8Ss_YQ) code: tvyv | [BaiduDrive](https://pan.baidu.com/s/1T6_TkEh9v9Vtf4Sw-chT2w), code: hpbt |
| DP | MagFace | iResNet18 | CASIA-WebFace | No | [GoogleDrive](https://drive.google.com/file/d/18pSIQOHRBQ-srrYfej20S5M8X8b_7zb9/view?usp=sharing), [BaiduDrive](https://pan.baidu.com/s/1N478xTfSow342WsP9LTRXA) code: fkja | [BaiduDrive](https://pan.baidu.com/s/1bdfE7W2ffUB8ehDaOt-tBw), code: qv2x |
| DP | ArcFace | iResNet18 | CASIA-WebFace | No | [BaiduDrive](https://pan.baidu.com/s/1M2M8u-GO6BnrxgYAOtXYEA) code: wq2w | [BaiduDrive](https://pan.baidu.com/s/1lp4wAlz85w2Y29DT8RqGfQ), code: 756e |


## Evaluation

### Face Recognition
Steps to evaluate modes on lfw/cfp/agedb:

1. download data from [GDrive](https://drive.google.com/file/d/1HBGwyTFnl4Bt4hl5BpLE3t__J84R72TX/view?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/1vmw_1kOnKIu10jm5xlAxAQ), code: z7hs
2. `cd eval/eval_recognition/` and extract the data in the folder
3. evaluate the model by with `eval.sh` (e.g., `./eval.sh magface_epoch_00025.pth official 100`)

Use `eval_ijb.sh` for evaluation on IJB-B ([Gdrive](https://drive.google.com/file/d/1eR1xUXNf16wLQAH0It8YpfyUN5SvHgcz/view?usp=sharing) or[BaiduDrive](https://pan.baidu.com/s/1br4I7EAmNwHKkxofqY6w0A) code: iiwa) and IJB-C ([Gdrive](https://drive.google.com/file/d/10728RcLaX-LEYUHYtCLcIaCLGOYnajXy/view?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/1BcPsBvzKOw0ONZlv_RuhpQ) code: q6md). **Please apply for permissions from [NIST](https://www.nist.gov/programs-projects/face-challenges) before your usage.**

### Quality Assessment
Steps to calculate face qualities ([examples.ipynb](inference/examples.ipynb) is a toy example).

1. extract features from faces with `inference/gen_feat.py`. 
2. calculate feature magnitudes with `np.linalg.norm()`. 

Plot the error-versus-reject curve: 

1. prepare the features (in the recognition step).
2. `cd eva/eval_quality` and run `eval_quality.sh` (e.g., `./eval_quality.sh  lfw`).

Note: model used in the quality assessment session of the paper can be found [here](https://drive.google.com/file/d/1vA1AVLblzdal_twXrLQhUKZ4ms9_hgqg/view?usp=sharing).


## Basic Training
1. install [requirements](raw/requirements.txt).
2. Align images to 112x112 pixels with 5 facial landmarks ([code](https://github.com/deepinsight/insightface/blob/cdc3d4ed5de14712378f3d5a14249661e54a03ec/python-package/insightface/utils/face_align.py)).
3. Prepare a training list with format `imgname 0 id 0` in each line (`id` starts from 0), as indicated [here](dataloader/dataloader.py#L31-L32). In the paper, we employ MS1MV2 as the training dataset which can be downloaded from InsightFace (MS1M-ArcFace in [DataZoo](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)).
Use [`rec2image.py`](https://github.com/deepinsight/insightface/blob/0b5cab57b6011a587386bb14ac01ff2d74af1ff9/recognition/common/rec2image.py) to extract images.
4. Modify parameters in `run.sh/run_dist.sh/run_dist_cos.sh` and run it.


## Parallel Training
**Note:** Use **Pytorch > 1.7** for this feature. Codes are mainly based on [torchshard](https://github.com/KaiyuYue/torchshard) from [Kaiyu Yue](http://kaiyuyue.com/).

How to run: 

1. Update NCCL info (can be found with the command `ifconfig`) and port info in [train_dist.py](run/train_dist.py#L290-292)
2. Set the number of gpus in [here](run/train_dist.py#L283). 
3. [Optional. Not tested yet!] If training with multi-machines, modify [node number](run/train_dist.py#L284).
4. [Optional. **Help needed** as NAN can be reached during training.] Enable fp16 training by setiing `--fp16 1` in run/run_dist.sh.
5. run run/run_dist.sh.


Parallel training (Sec. 5.1 in [ArcFace](https://arxiv.org/pdf/1801.07698v3.pdf)) can highly speed up training as well as reduce consumption of GPU memory. Here are some results.

| Parallel Method | Float Type | Backbone | GPU | Batch Size | FC Size | Split FC? | Avg. Throughput (images/sec) | Memory (MiB) | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | FP32 | iResNet50 | v100 x 8 | 512 |  85742 | No | 1099.41 | 8681 |
| DDP | FP32 | iResNet50 | v100 x 8 | 512 |  85742 | **Yes** | 1687.71 | 8137 |
| DDP | FP16 | iResNet50 | v100 x 8 | 512 |  85742 | **Yes** | 3388.66 | 5629 |
| DP | FP32 | iResNet100 | v100 x 8 | 512 |  85742 | No | 612.40 | 11825 |
| DDP | FP32 | iResNet100 | v100 x 8 | 512 |  85742 | **Yes** | 1060.16 | 10777 |
| DDP | FP16 | iResNet100 | v100 x 8 | 512 |  85742 | **Yes** | 2013.90 | 7319 |

## Tips for MagFace
1. In practical, one may want to finetune a existing model for either performance boosts or quality-aware ability. This is practicable (verified in our scenario) but requires a few modifications. Here are my recommended steps:
   - Generate features from a few samples by existing model and calculate their magnitudes.
   - Assume that magnitudes are distributed in `[x1, x2]`, then modify parameters to meet `l_a < x1, u_a > x2`.
   - In our scenario, we have a model trained by ArcFace which produces magnitudes around 1. `[l_a, u_a, l_m, u_m, l_g] =[1, 51, 0.45, 1, 5]` is a good choice.
  