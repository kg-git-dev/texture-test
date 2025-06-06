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
import cv2
import utils3d
import nvdiffrast.torch as dr

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
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    texture_size: int = 1024,
    near: float = 0.1,
    far: float = 10.0,
    debug: bool = False,
    verbose: bool = True,
):
    """
    Bake texture to a mesh from multiple observations.

    Args:
    app_rep (Union[Strivec, Gaussian]): Appearance representation.
    mesh (MeshExtractResult): Extracted mesh.
    simplify (float): Ratio of faces to remove in simplification.
    texture_size (int): Size of the texture.
    near (float): Near plane of the camera.
    far (float): Far plane of the camera.
    debug (bool): Whether to print debug information.
    verbose (bool): Whether to print progress.
    """
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    # mesh postprocess
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        verbose=verbose,
    )

    vertices, faces, uvs = parametrize_mesh(vertices, faces)
    
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)
    
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]

    vertices = torch.tensor(vertices).cuda()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    uvs = torch.tensor(uvs).cuda()
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    masks = [torch.tensor(m>0).bool().cuda() for m in masks]
    views = [extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
    projections = [intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]    

    rastctx = utils3d.torch.RastContext(backend='cuda')
    observations = [observations.flip(0) for observations in observations]
    masks = [m.flip(0) for m in masks]
    _uv = []
    _uv_dr = []
    for observation, view, projection in tqdm(zip(observations, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV'):
        with torch.no_grad():
            rast = utils3d.torch.rasterize_triangle_faces(
                rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
            )
            _uv.append(rast['uv'].detach())
            _uv_dr.append(rast['uv_dr'].detach())

    texture = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).cuda())
    optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
    
    def tv_loss(texture):
        return torch.nn.functional.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :]) + \
                torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])

    lambda_tv = 1e-2
    total_steps = 2500
    with tqdm(total=total_steps, disable=not verbose, desc='Texture baking (opt): optimizing') as pbar:
        for step in range(total_steps):
            optimizer.zero_grad()
            selected = np.random.randint(0, len(views))
            uv, uv_dr, observation, mask = _uv[selected], _uv_dr[selected], observations[selected], masks[selected]
            render = dr.texture(texture, uv, uv_dr)[0]
            loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
            if lambda_tv > 0:
                loss += lambda_tv * tv_loss(texture)
            loss.backward()
            optimizer.step()
            # annealing
            optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, total_steps, 1e-2, 1e-5)
            pbar.set_postfix({'loss': loss.item()})
            pbar.update()
    texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    mask = 1 - utils3d.torch.rasterize_triangle_faces(
        rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
    )['mask'][0].detach().cpu().numpy().astype(np.uint8)
    texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    # After mesh rendering
    texture = Image.fromarray(texture)

    # rotate mesh (from z-up to y-up)
    vertices = vertices.cpu().numpy()  # Move to CPU and convert to NumPy
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Convert faces and uvs to NumPy for trimesh
    faces = faces.cpu().numpy()
    uvs = uvs.cpu().numpy()

    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
    )
    mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    
    return mesh

