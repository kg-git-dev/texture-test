from typing import *
from functools import wraps
import inspect
from numbers import Number
import numpy as np
import torch
import xatlas
from .random_utils import sphere_hammersley_sequence
from tqdm import tqdm
import pyvista as pv

from .render_utils import get_renderer
from ..representations import Strivec, Gaussian, MeshExtractResult
from PIL import Image
import trimesh

import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    BlendParams,
    HardPhongShader
)
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.transforms import Transform3d

import cv2
import matplotlib.pyplot as plt
import os


def suppress_traceback(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise
    return wrapper

def get_device(args, kwargs):
    device = None
    for arg in (list(args) + list(kwargs.values())):
        if isinstance(arg, torch.Tensor):
            if device is None:
                device = arg.device
            elif device != arg.device:
                raise ValueError("All tensors must be on the same device.")
    return device

def get_args_order(func, args, kwargs):
    """
    Get the order of the arguments of a function.
    """
    names = inspect.getfullargspec(func).args
    names_idx = {name: i for i, name in enumerate(names)}
    args_order = []
    kwargs_order = {}
    for name, arg in kwargs.items():
        if name in names:
            kwargs_order[name] = names_idx[name]
            names.remove(name)
    for i, arg in enumerate(args):
        if i < len(names):
            args_order.append(names_idx[names[i]])
    return args_order, kwargs_order

def broadcast_args(args, kwargs, args_dim, kwargs_dim):
    spatial = []
    for arg, arg_dim in zip(args + list(kwargs.values()), args_dim + list(kwargs_dim.values())):
        if isinstance(arg, torch.Tensor) and arg_dim is not None:
            arg_spatial = arg.shape[:arg.ndim-arg_dim]
            if len(arg_spatial) > len(spatial):
                spatial = [1] * (len(arg_spatial) - len(spatial)) + spatial
            for j in range(len(arg_spatial)):
                if spatial[-j] < arg_spatial[-j]:
                    if spatial[-j] == 1:
                        spatial[-j] = arg_spatial[-j]
                    else:
                        raise ValueError("Cannot broadcast arguments.")
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and args_dim[i] is not None:
            args[i] = torch.broadcast_to(arg, [*spatial, *arg.shape[arg.ndim-args_dim[i]:]])
    for key, arg in kwargs.items():
        if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
            kwargs[key] = torch.broadcast_to(arg, [*spatial, *arg.shape[arg.ndim-kwargs_dim[key]:]])
    return args, kwargs, spatial

@suppress_traceback
def batched(*dims):
    """
    Decorator that allows a function to be called with batched arguments.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, device=torch.device('cpu'), **kwargs):
            args = list(args)
            # get arguments dimensions
            args_order, kwargs_order = get_args_order(func, args, kwargs)
            args_dim = [dims[i] for i in args_order]
            kwargs_dim = {key: dims[i] for key, i in kwargs_order.items()}
            # convert to torch tensor
            device = get_device(args, kwargs) or device
            for i, arg in enumerate(args):
                if isinstance(arg, (Number, list, tuple)) and args_dim[i] is not None:
                    args[i] = torch.tensor(arg, device=device)
            for key, arg in kwargs.items():
                if isinstance(arg, (Number, list, tuple)) and kwargs_dim[key] is not None:
                    kwargs[key] = torch.tensor(arg, device=device)
            # broadcast arguments
            args, kwargs, spatial = broadcast_args(args, kwargs, args_dim, kwargs_dim)
            for i, (arg, arg_dim) in enumerate(zip(args, args_dim)):
                if isinstance(arg, torch.Tensor) and arg_dim is not None:
                    args[i] = arg.reshape([-1, *arg.shape[arg.ndim-arg_dim:]])
            for key, arg in kwargs.items():
                if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
                    kwargs[key] = arg.reshape([-1, *arg.shape[arg.ndim-kwargs_dim[key]:]])
            # call function
            results = func(*args, **kwargs)
            type_results = type(results)
            results = list(results) if isinstance(results, (tuple, list)) else [results]
            # restore spatial dimensions
            for i, result in enumerate(results):
                results[i] = result.reshape([*spatial, *result.shape[1:]])
            if type_results == tuple:
                results = tuple(results)
            elif type_results == list:
                results = list(results)
            else:
                results = results[0]
            return results
        return wrapper
    return decorator

@batched(2)
def extrinsics_to_view(
        extrinsics: torch.Tensor
    ) -> torch.Tensor:
    """
    OpenCV camera extrinsics to OpenGL view matrix

    Args:
        extrinsics (torch.Tensor): [..., 4, 4] OpenCV camera extrinsics matrix

    Returns:
        (torch.Tensor): [..., 4, 4] OpenGL view matrix
    """
    return extrinsics * torch.tensor([1, -1, -1, 1], dtype=extrinsics.dtype, device=extrinsics.device)[:, None]

@batched(2,0,0)
def intrinsics_to_perspective(
        intrinsics: torch.Tensor,
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor],
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
        near (float | torch.Tensor): [...] near plane to clip
        far (float | torch.Tensor): [...] far plane to clip
    Returns:
        (torch.Tensor): [..., 4, 4] OpenGL perspective matrix
    """
    N = intrinsics.shape[0]
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    ret = torch.zeros((N, 4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[:, 0, 0] = 2 * fx
    ret[:, 1, 1] = 2 * fy
    ret[:, 0, 2] = -2 * cx + 1
    ret[:, 1, 2] = 2 * cy - 1
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2. * near * far / (near - far)
    ret[:, 3, 2] = -1.
    return ret

@batched(1, 1, 1)
def extrinsics_look_at(
    eye: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """
    Get OpenCV extrinsics matrix looking at something

    Args:
        eye (torch.Tensor): [..., 3] the eye position
        look_at (torch.Tensor): [..., 3] the position to look at
        up (torch.Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (torch.Tensor): [..., 4, 4], extrinsics matrix
    """
    N = eye.shape[0]
    z = look_at - eye
    x = torch.cross(-up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    # x = torch.cross(y, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    R = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(R, eye[..., None])
    ret = torch.zeros((N, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = R
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.
    return ret

def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics

@batched(0,0,0,0)
def intrinsics_from_focal_center(
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix

    Args:
        focal_x (float | torch.Tensor): focal length in x axis
        focal_y (float | torch.Tensor): focal length in y axis
        cx (float | torch.Tensor): principal point in x axis
        cy (float | torch.Tensor): principal point in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    N = fx.shape[0]
    ret = torch.zeros((N, 3, 3), dtype=fx.dtype, device=fx.device)
    zeros, ones = torch.zeros(N, dtype=fx.dtype, device=fx.device), torch.ones(N, dtype=fx.dtype, device=fx.device)
    ret = torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))
    return ret

def intrinsics_from_fov_xy(
    fov_x: Union[float, torch.Tensor],
    fov_y: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix from field of view in x and y axis

    Args:
        fov_x (float | torch.Tensor): field of view in x axis
        fov_y (float | torch.Tensor): field of view in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    focal_x = 0.5 / torch.tan(fov_x / 2)
    focal_y = 0.5 / torch.tan(fov_y / 2)
    cx = cy = 0.5
    return intrinsics_from_focal_center(focal_x, focal_y, cx, cy)

def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    renderer = get_renderer(sample, **options)
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        else:
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
    return rets

def render_multiview(sample, resolution=512, nviews=30, r=2, fov=40):
    # r = 2
    # fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics

def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')
        if verbose:
            tqdm.write(f'After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    return vertices, faces



def bake_texture_and_return_mesh(
    app_rep,
    mesh,
    simplify: float = 0.90,
    texture_size: int = 1024,
    near: float = 0.1,
    far: float = 10.0,
    debug: bool = False,
    verbose: bool = True,
):
    """
    Bakes a texture onto `mesh` using multi-view renders of `app_rep`
    (Gaussian | Strivec) and returns a trimesh.Trimesh with UV + PBR texture.
    When `debug=True` diagnostic images/statistics are saved to ./debug_bake.
    """
    # ------------ 0. prep I/O paths ------------
    if debug:
        dbg_dir = "debug_bake"
        os.makedirs(dbg_dir, exist_ok=True)
        print(f"[DBG] saving debug artefacts to {dbg_dir}/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ 1. mesh pre-processing ------------
    V_np, F_np = mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy()
    V_np, F_np = postprocess_mesh(V_np, F_np,
                                  simplify=simplify > 0,
                                  simplify_ratio=simplify,
                                  verbose=verbose)
    V_np, F_np, UV_np = parametrize_mesh(V_np, F_np)

    # ------------ 2. render synthetic observations ------------
    # (we keep PyTorch3D renders on GPU; move copies to CPU only for debug PNGs)
    imgs, extrs, intrs = render_multiview(app_rep,
                                          resolution=1024,
                                          nviews=60)
    H = W = imgs[0].shape[0]
    masks_np = [np.any(rgb > 0, axis=-1) for rgb in imgs]

    # dump a few observations & masks
    if debug:
        for i in range(min(4, len(imgs))):
            cv2.imwrite(f"{dbg_dir}/obs_{i}.png", cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{dbg_dir}/mask_{i}.png", (masks_np[i] * 255).astype(np.uint8))

    # ------------ 3. move everything to torch.float32 ------------
    V     = torch.tensor(V_np,  dtype=torch.float32, device=device)
    F     = torch.tensor(F_np,  dtype=torch.int64,   device=device)
    UV    = torch.tensor(UV_np, dtype=torch.float32, device=device)
    faces_uvs = F                      # xatlas guarantees 1-to-1 UVs

    views = [extrinsics_to_view(torch.tensor(e, dtype=torch.float32,
                                             device=device))
             for e in (ex.cpu().numpy() for ex in extrs)]
    projs = [intrinsics_to_perspective(torch.tensor(k, dtype=torch.float32,
                                                    device=device), near, far)
             for k in (in_.cpu().numpy() for in_ in intrs)]

    obs_torch = [torch.tensor(img / 255.0, dtype=torch.float32, device=device)
                 for img in imgs]
    mask_bool = [torch.tensor(m, dtype=torch.bool, device=device) for m in masks_np]

    # ------------ 4. build empty‐textured mesh ------------
    tex_init = torch.zeros((1, texture_size, texture_size, 3),
                           dtype=torch.float32, device=device)
    mesh_p3d = Meshes(
        verts=[V], faces=[F],
        textures=TexturesUV(
            maps=tex_init.permute(0, 3, 1, 2),
            faces_uvs=[faces_uvs], verts_uvs=[UV])
    )

    rasteriser = MeshRasterizer(
        raster_settings=RasterizationSettings(
            image_size=(H, W), faces_per_pixel=1, blur_radius=0.0)
    )

    # ------------ 5. accumulators ------------
    tex_acc   = torch.zeros_like(tex_init[0])
    w_acc     = torch.zeros(texture_size, texture_size, 1, device=device)

    vis_total = 0  # for stats

    # ------------ 6. per-view bake ------------
    for vi, (rgb, msk, view_m, K) in enumerate(tqdm(
            list(zip(obs_torch, mask_bool, views, projs)),
            desc="Baking", disable=not verbose)):

        R, T = view_m[:3, :3], view_m[:3, 3]
        fx, fy, cx, cy = (K[0, 0] * W, K[1, 1] * H,
                          K[0, 2] * W, K[1, 2] * H)
        cam = PerspectiveCameras(
            device=device, in_ndc=False,
            R=R[None], T=T[None],
            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
            principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
            image_size=torch.tensor([[H, W]], dtype=torch.int32, device=device),
        )

        frags = rasteriser(mesh_p3d, cameras=cam)
        pix2f = frags.pix_to_face[0, ..., 0]
        bari  = frags.bary_coords[0, ..., 0, :]
        vis_mask = (pix2f >= 0) & msk      # combine rasteriser & user mask

        if debug and vi < 4:
            # cyan = rasteriser, red = user mask
            vis_png = np.zeros((H, W, 3), np.uint8)
            vis_png[..., 2] = vis_mask.cpu().numpy() * 255  # blue
            vis_png[..., 1] = msk.cpu().numpy() * 255       # green
            cv2.imwrite(f"{dbg_dir}/visible_{vi}.png", vis_png)

        if vis_mask.sum() == 0:
            continue

        vis_total += int(vis_mask.sum())

        fi = pix2f[vis_mask]
        bari_vis = bari[vis_mask]
        uv_tri   = UV[faces_uvs[fi]]
        uv_vis   = (bari_vis[..., None] * uv_tri).sum(-2)
        uv_vis[..., 1] = 1.0 - uv_vis[..., 1]

        tex_xy = (uv_vis * (texture_size - 1)).long()
        u, v   = tex_xy[:, 0], tex_xy[:, 1]
        clr    = rgb[vis_mask]

        tex_acc.index_put_((v, u), clr, accumulate=True)
        w_acc.index_put_((v, u),
                         torch.ones_like(v, dtype=torch.float32).unsqueeze(-1),
                         accumulate=True)

    # ------------ 7. final texture ------------
    covered = (w_acc > 0).sum().item()
    print(f"[INFO] Texels hit by at least one view: {covered}"
          f" / {texture_size*texture_size} ({covered/texture_size**2:.1%})")
    if vis_total == 0:
        raise RuntimeError("No pixel was ever considered visible; "
                           "check masks & camera matrices!")

    tex_final = tex_acc / w_acc.clamp(min=1e-6)
    if debug:
        # heat-map of coverage
        heat = (w_acc.squeeze() / w_acc.max()).cpu().numpy()
        heat = (plt_cm := cv2.applyColorMap((heat * 255).astype(np.uint8),
                                            cv2.COLORMAP_VIRIDIS))
        cv2.imwrite(f"{dbg_dir}/uv_scatter.png", heat)
        cv2.imwrite(f"{dbg_dir}/baked_texture.png",
                    cv2.cvtColor((tex_final.cpu().numpy()*255).astype(np.uint8),
                                 cv2.COLOR_RGB2BGR))

    # ------------ 8. debug render of baked mesh ------------
    if debug:
        # from pytorch3d.renderer import (
        #     RasterizationSettings, MeshRenderer, HardPhongShader,
        #     PointLights, BlendParams)
        R0, T0 = look_at_view_transform(dist=2.0, elev=0, azim=0, device=device)
        cam0 = PerspectiveCameras(device=device, R=R0, T=T0,
                                  fov=40.0, in_ndc=True)
        mesh_p3d.textures = TexturesUV(
            maps=tex_final.permute(2, 0, 1)[None],
            faces_uvs=[faces_uvs], verts_uvs=[UV])

        renderer_dbg = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cam0,
                raster_settings=RasterizationSettings(image_size=H)),
            shader=HardPhongShader(
                device=device, cameras=cam0,
                lights=PointLights(device=device, location=[[0, 2, 2]]),
                blend_params=BlendParams(background_color=(0, 0, 0)))
        )
        dbg_img = renderer_dbg(mesh_p3d)[0, ..., :3].cpu().numpy()
        cv2.imwrite(f"{dbg_dir}/render_debug.png",
                    cv2.cvtColor((dbg_img * 255).astype(np.uint8),
                                 cv2.COLOR_RGB2BGR))

    # ------------ 9. export to trimesh ------------
    tex_np = (tex_final.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    tex_img = Image.fromarray(tex_np)
    R_zup_to_yup = np.array([[1, 0, 0],
                             [0, 0,-1],
                             [0, 1, 0]], np.float32)
    verts_np = (V.cpu().numpy() @ R_zup_to_yup.T)
    visual = trimesh.visual.TextureVisuals(
        uv=UV.cpu().numpy(),
        image=tex_img,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_img, roughnessFactor=1.0))
    tri_mesh = trimesh.Trimesh(verts_np, F_np, visual=visual, process=False)

    print("[DONE] Bake complete.")
    return tri_mesh



