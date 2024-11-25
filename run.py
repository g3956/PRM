import os
import argparse
import glm
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
import torchvision
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.data.objaverse import load_mipmap
from src.utils import render_utils
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    center_looking_at_camera_pose,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video

def str_to_tuple(arg_str):
    try:
        return eval(arg_str)
    except:
        raise argparse.ArgumentTypeError("Tuple argument must be in the format (x, y)")
    

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


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--model_ckpt_path', type=str, default="", help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=100, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--materials', type=str_to_tuple, default=(1.0, 0.1), help=' metallic and roughness')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--fov', type=float, default=30, help='Render distance.')
parser.add_argument('--env_path', type=str, default='data/env_mipmap/2', help='environment map')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
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

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=50.0)
model = model.eval()

# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# process input files
if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, file) 
        for file in os.listdir(args.input_path) 
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
    ]
else:
    input_files = [args.input_path]
print(f'Total number of input images: {len(input_files)}')

###############################################################################
# Stage 1: Multiview generation.
###############################################################################

rembg_session = None if args.no_rembg else rembg.new_session()

outputs = []
for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

    # remove background optionally
    input_image = Image.open(image_file)
    if not args.no_rembg:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    # sampling
    output_image = pipeline(
        input_image, 
        num_inference_steps=args.diffusion_steps, 
    ).images[0]
    print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
    torchvision.utils.save_image(images, os.path.join(image_path, f'{name}.png'))
    sample = {'name': name, 'images': images}

# delete pipeline to save memory
# del pipeline

###############################################################################
# Stage 2: Reconstruction.
###############################################################################

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=3.2*args.scale, fov=30).to(device)
    chunk_size = 20 if IS_FLEXICUBES else 1

# for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 512, interpolation=3, antialias=True).clamp(0, 1)

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)
        
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )
        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")

        render_size = 512
        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution
            ENV = load_mipmap(args.env_path)
            materials = args.materials
            
            all_mv, all_mvp, all_campos = get_render_cameras(
                batch_size=1, 
                M=240, 
                radius=args.distance, 
                elevation=(90, 60.0),
                is_flexicubes=IS_FLEXICUBES,
                fov=args.fov
            )
            
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
                
            # breakpoint()
            save_video(
                all_frames,
                video_path_idx,
                fps=30,
            )
            print(f"Video saved to {video_path_idx}")
