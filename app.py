import os
import imageio
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
import glm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.data.objaverse import load_mipmap
from src.utils import render_utils
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_glb
from src.utils.infer_util import remove_background, resize_foreground, images_to_video

import tempfile
from huggingface_hub import hf_hub_download


if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:0')
else:
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = device0

# Define the cache directory for model files
model_cache_dir = './ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)

def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False, fov=50):
    """
    Get the rendering camera parameters.
    """
    train_res = [512, 512]
    cam_near_far = [0.1, 1000.0]
    fovy = np.deg2rad(fov)
    proj_mtx = render_utils.perspective(fovy, train_res[1] / train_res[0], cam_near_far[0], cam_near_far[1])
    all_mv = []
    all_mvp = []
    all_campos = []
    if isinstance(elevation, tuple):
        elevation_0 = np.deg2rad(elevation[0])
        elevation_1 = np.deg2rad(elevation[1])
        for i in range(M//2):
            azimuth = 2 * np.pi * i / (M // 2)
            z = radius * np.cos(azimuth) * np.sin(elevation_0)
            x = radius * np.sin(azimuth) * np.sin(elevation_0)
            y = radius * np.cos(elevation_0)

            eye = glm.vec3(x, y, z)
            at = glm.vec3(0.0, 0.0, 0.0)
            up = glm.vec3(0.0, 1.0, 0.0)
            view_matrix = glm.lookAt(eye, at, up)
            mv = torch.from_numpy(np.array(view_matrix))
            mvp   = proj_mtx @ (mv)  #w2c
            campos = torch.linalg.inv(mv)[:3, 3]
            all_mv.append(mv[None, ...].cuda())
            all_mvp.append(mvp[None, ...].cuda())
            all_campos.append(campos[None, ...].cuda())
        for i in range(M//2):
            azimuth = 2 * np.pi * i / (M // 2)
            z = radius * np.cos(azimuth) * np.sin(elevation_1)
            x = radius * np.sin(azimuth) * np.sin(elevation_1)
            y = radius * np.cos(elevation_1)

            eye = glm.vec3(x, y, z)
            at = glm.vec3(0.0, 0.0, 0.0)
            up = glm.vec3(0.0, 1.0, 0.0)
            view_matrix = glm.lookAt(eye, at, up)
            mv = torch.from_numpy(np.array(view_matrix))
            mvp   = proj_mtx @ (mv)  #w2c
            campos = torch.linalg.inv(mv)[:3, 3]
            all_mv.append(mv[None, ...].cuda())
            all_mvp.append(mvp[None, ...].cuda())
            all_campos.append(campos[None, ...].cuda())
    else:
        # elevation = 90 - elevation
        for i in range(M):
            azimuth = 2 * np.pi * i / M
            z = radius * np.cos(azimuth) * np.sin(elevation)
            x = radius * np.sin(azimuth) * np.sin(elevation)
            y = radius * np.cos(elevation)

            eye = glm.vec3(x, y, z)
            at = glm.vec3(0.0, 0.0, 0.0)
            up = glm.vec3(0.0, 1.0, 0.0)
            view_matrix = glm.lookAt(eye, at, up)
            mv = torch.from_numpy(np.array(view_matrix))
            mvp   = proj_mtx @ (mv)  #w2c
            campos = torch.linalg.inv(mv)[:3, 3]
            all_mv.append(mv[None, ...].cuda())
            all_mvp.append(mvp[None, ...].cuda())
            all_campos.append(campos[None, ...].cuda())
    all_mv = torch.stack(all_mv, dim=0).unsqueeze(0).squeeze(2)
    all_mvp = torch.stack(all_mvp, dim=0).unsqueeze(0).squeeze(2)
    all_campos = torch.stack(all_campos, dim=0).unsqueeze(0).squeeze(2)
    return all_mv, all_mvp, all_campos


def render_frames(model, planes, render_cameras, camera_pos, env, materials, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    albedos = []
    pbr_spec_lights = []
    pbr_diffuse_lights = []
    normals = []
    alphas = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            out = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                camera_pos[:, i:i+chunk_size],
                [[env]*chunk_size],
                [[materials]*chunk_size],
                render_size=render_size,
            )
            frame = out['pbr_img']
            albedo = out['albedo']
            pbr_spec_light = out['pbr_spec_light']
            pbr_diffuse_light = out['pbr_diffuse_light']
            normal = out['normal']
            alpha = out['mask']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[i],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
        albedos.append(albedo)
        pbr_spec_lights.append(pbr_spec_light)
        pbr_diffuse_lights.append(pbr_diffuse_light)
        normals.append(normal)
        alphas.append(alpha)

    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    alphas = torch.cat(alphas, dim=1)[0]    
    albedos = torch.cat(albedos, dim=1)[0]
    pbr_spec_lights = torch.cat(pbr_spec_lights, dim=1)[0]
    pbr_diffuse_lights = torch.cat(pbr_diffuse_lights, dim=1)[0]
    normals = torch.cat(normals, dim=0).permute(0,3,1,2)[:,:3]
    return frames, albedos, pbr_spec_lights, pbr_diffuse_lights, normals, alphas



def images_to_video(images, output_path, fps=30):
    # images: (N, C, H, W)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='h264')


###############################################################################
# Configuration.
###############################################################################

seed_everything(0)

config_path = 'configs/PRM_inference.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True

device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
    cache_dir=model_cache_dir
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
print('Loading custom white-background unet ...')
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="LTT/PRM", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

# load reconstruction model
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="LTT/PRM", filename="final_ckpt.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device1)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device1, fovy=30.0)
model = model.eval()

print('Loading Finished!')


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background):

    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image


def generate_mvs(input_image, sample_steps, sample_seed):

    seed_everything(sample_seed)
    
    # sampling
    generator = torch.Generator(device=device0)
    z123_image = pipeline(
        input_image, 
        num_inference_steps=sample_steps, 
        generator=generator,
    ).images[0]

    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)     # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())

    return z123_image, show_image


def make_mesh(mesh_fpath, planes):

    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")
        
    with torch.no_grad():
        # get mesh

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]
        
        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
        save_obj(vertices, faces, vertex_colors, mesh_fpath)
        
        print(f"Mesh saved to {mesh_fpath}")

    return mesh_fpath, mesh_glb_fpath


def make3d(images):

    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=3.2, fov=30).to(device).to(device1)
    all_mv, all_mvp, all_campos = get_render_cameras(
                batch_size=1, 
                M=240, 
                radius=4.5, 
                elevation=(90, 60.0),
                is_flexicubes=IS_FLEXICUBES,
                fov=30
            )

    images = images.unsqueeze(0).to(device1)
    images = v2.functional.resize(images, (512, 512), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    print(mesh_fpath)
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")
    ENV = load_mipmap("env_mipmap/6")
    materials = (0.0,0.9)
    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get video
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 512
        
        frames = []
        frames, albedos, pbr_spec_lights, pbr_diffuse_lights, normals, alphas = render_frames(
                model, 
                planes, 
                render_cameras=all_mvp,
                camera_pos=all_campos,
                env=ENV,
                materials=materials,
                render_size=render_size, 
                chunk_size=chunk_size, 
                is_flexicubes=IS_FLEXICUBES,
            )
        normals = (torch.nn.functional.normalize(normals) + 1) / 2
        normals = normals * alphas + (1-alphas)
        all_frames = torch.cat([frames, albedos, pbr_spec_lights, pbr_diffuse_lights, normals], dim=3)
                
        images_to_video(
            all_frames,
            video_fpath,
            fps=30,
        )

        print(f"Video saved to {video_fpath}")

    mesh_fpath, mesh_glb_fpath = make_mesh(mesh_fpath, planes)

    return video_fpath, mesh_fpath, mesh_glb_fpath


import gradio as gr

_HEADER_ = '''
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/g3956/PRM' target='_blank'><b>PRM: Photometric Stereo based Large Reconstruction Model</b></a></h2>

**PRM** is a feed-forward framework for high-quality 3D mesh generation with fine-grained local details from a single image.

Code: <a href='https://github.com/g3956/PRM' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/2404.07191' target='_blank'>ArXiv</a>.
'''

_CITE_ = r"""
If PRM is helpful, please help to ⭐ the <a href='https://github.com/g3956/PRM' target='_blank'>Github Repo</a>. Thanks!
---
📝 **Citation**

If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

📋 **License**

Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/InstantMesh/blob/main/LICENSE) for details.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>jlin695@connect.hkust-gz.edu.cn</b>.
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Processed Image", 
                    image_mode="RGBA", 
                    width=256,
                    height=256,
                    type="pil", 
                    interactive=False
                )
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                    sample_steps = gr.Slider(
                        label="Sample Steps",
                        minimum=30,
                        maximum=100,
                        value=75,
                        step=5
                    )

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                    ],
                    inputs=[input_image],
                    label="Examples",
                    examples_per_page=20
                )

        with gr.Column():

            with gr.Row():

                with gr.Column():
                    mv_show_images = gr.Image(
                        label="Generated Multi-views",
                        type="pil",
                        width=379,
                        interactive=False
                    )

            with gr.Column():
                with gr.Column():
                    output_video = gr.Video(
                        label="video", format="mp4",
                        width=768,
                        autoplay=True,
                        interactive=False
                    )

            with gr.Row():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage.")
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")

            with gr.Row():
                gr.Markdown('''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')

    gr.Markdown(_CITE_)
    mv_images = gr.State()

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image],
    ).success(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed],
        outputs=[mv_images, mv_show_images],
    ).success(
        fn=make3d,
        inputs=[mv_images],
        outputs=[output_video, output_model_obj, output_model_glb]
    )

demo.queue(max_size=10)
demo.launch(server_port=1211)
