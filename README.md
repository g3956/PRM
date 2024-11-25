

<div align="center">
  
# PRM:  Photometric Stereo based Large Reconstruction Model

<a href="https://tau-yihouxiang.github.io/projects/X-Ray/X-Ray.html"><img src="https://img.shields.io/badge/Project_Page-Online-EA3A97"></a>
<a href="https://arxiv.org/abs/2404.07191"><img src="https://img.shields.io/badge/ArXiv-2404.07191-brightgreen"></a> 
<a href="https://huggingface.co/LTT/PRM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>  <br>
<a href="https://huggingface.co/spaces/TencentARC/InstantMesh"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>
<a href="https://github.com/jtydhr88/ComfyUI-InstantMesh"><img src="https://img.shields.io/badge/Demo-ComfyUI-8A2BE2"></a>

</div>

---

An official implementation of PRM, a feed-forward framework for high-quality 3D mesh generation with photometric stereo images.


![image](https://github.com/g3956/PRM/blob/main/assets/teaser.png)

# 🚩 Features
- [x] Release inference and training code.
- [x] Release model weights.
- [x] Release huggingface gradio demo. Please try it at [demo](https://huggingface.co/spaces/TencentARC/InstantMesh) link.
- [x] Release ComfyUI demo.

# ⚙️ Dependencies and Installation

We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name PRM python=3.10
conda activate PRM
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# Install Triton 
pip install triton

# Install other requirements
pip install -r requirements.txt
```

# 💫 Inference

## Download the pretrained model

The pretrained model can be found [model card](https://huggingface.co/TencentARC/InstantMesh).

Our inference script will download the models automatically. Alternatively, you can manually download the models and put them under the `ckpts/` directory.

# 💻 Training

We provide our training code to facilitate future research. 
For training data, we used filtered Objaverse for training. Before training, you need to pre-processe the environment maps and GLB files into formats that fit our dataloader.
For preprocessing GLB files, please run
```bash
# GLB files to OBJ files
python train.py --base configs/instant-mesh-large-train.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```
then 
```bash
# OBJ files to mesh files that can be readed
python obj2mesh.py path_to_obj save_path
```
For preprocessing environment maps, please run
```bash
# Pre-process environment maps
python light2map.py path_to_env save_path
```


To train the sparse-view reconstruction models, please run:
```bash
# Training on Mesh representation
python train.py --base configs/PRM.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```
Note that you need to change to root_dir and light_dir to pathes that you save the preprocessed GLB files and environment maps.

# :books: Citation

If you find our work useful for your research or applications, please cite using this BibTeX:

```BibTeX
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

# 🤗 Acknowledgements

We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)
- [InstantMesh]([https://instant-3d.github.io/](https://github.com/TencentARC/InstantMesh))



