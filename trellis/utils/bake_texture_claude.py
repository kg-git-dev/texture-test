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
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    BlendParams
)
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.transforms import Transform3d


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
    Bake texture to a mesh from multiple observations using PyTorch3D.

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    # mesh postprocess
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        verbose=verbose,
    )

    # Parametrize mesh for UV mapping
    vertices, faces, uvs = parametrize_mesh(vertices, faces)
    
    # Render multiview observations
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)
    
    # Process observations and camera parameters
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]

    # Convert to tensors
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces.astype(np.int32), dtype=torch.long, device=device)
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
    observations = [torch.tensor(obs / 255.0, dtype=torch.float32, device=device) for obs in observations]
    masks = [torch.tensor(m, dtype=torch.bool, device=device) for m in masks]
    
    # Convert camera parameters to PyTorch3D format
    R_list = []
    T_list = []
    
    for extr in extrinsics:
        # Convert OpenCV extrinsics to PyTorch3D format
        # OpenCV: [R|t] where world_point = R * camera_point + t
        # PyTorch3D: camera_point = R * world_point + T
        R = torch.tensor(extr[:3, :3], dtype=torch.float32, device=device)
        t = torch.tensor(extr[:3, 3], dtype=torch.float32, device=device)
        
        # Convert to PyTorch3D convention
        R_pytorch3d = R.transpose(0, 1)  # Inverse rotation
        T_pytorch3d = -torch.matmul(R_pytorch3d, t)  # Corresponding translation
        
        R_list.append(R_pytorch3d)
        T_list.append(T_pytorch3d)
    
    R = torch.stack(R_list, dim=0)
    T = torch.stack(T_list, dim=0)
    
    # Create cameras for each view
    cameras = FoVPerspectiveCameras(
        R=R,
        T=T,
        fov=40.0,  # This should match your render_multiview fov parameter
        device=device
    )
    
    # Initialize texture map
    texture_map = torch.zeros((texture_size, texture_size, 3), dtype=torch.float32, device=device)
    weight_map = torch.zeros((texture_size, texture_size, 1), dtype=torch.float32, device=device)
    
    # Create mesh for rasterization
    mesh_pytorch3d = Meshes(
        verts=[vertices],
        faces=[faces],
        textures=TexturesUV(
            maps=torch.ones((1, texture_size, texture_size, 3), device=device),
            faces_uvs=[faces],
            verts_uvs=[uvs]
        )
    )
    
    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=1024,  # Should match your observation resolution
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None
    )
    
    # Debug: Save initial mesh and UV information
    if debug:
        print(f"\n=== DEBUGGING TEXTURE BAKING ===")
        print(f"Mesh info:")
        print(f"  - Vertices: {vertices.shape} (min: {vertices.min(dim=0)[0]}, max: {vertices.max(dim=0)[0]})")
        print(f"  - Faces: {faces.shape} (min: {faces.min()}, max: {faces.max()})")
        print(f"  - UVs: {uvs.shape} (min: {uvs.min(dim=0)[0]}, max: {uvs.max(dim=0)[0]})")
        print(f"Camera info:")
        print(f"  - Number of views: {len(observations)}")
        print(f"  - Observation resolution: {observations[0].shape if observations else 'N/A'}")
        print(f"  - R matrix sample: {R[0] if len(R) > 0 else 'N/A'}")
        print(f"  - T vector sample: {T[0] if len(T) > 0 else 'N/A'}")
        
        # Save UV visualization
        uv_vis = torch.zeros((texture_size, texture_size, 3), device=device)
        # Color UVs based on their original UV coordinates
        for face_idx in range(min(100, faces.shape[0])):  # Sample first 100 faces
            face_uvs = uvs[faces[face_idx]]
            for uv in face_uvs:
                u, v = (uv * (texture_size - 1)).long()
                u, v = torch.clamp(u, 0, texture_size - 1), torch.clamp(v, 0, texture_size - 1)
                v_flipped = texture_size - 1 - v
                uv_vis[v_flipped, u] = torch.tensor([uv[0], uv[1], 0.5], device=device)
        
        uv_vis_img = (uv_vis.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(uv_vis_img).save("debug_uv_mapping.png")
        print(f"  - Saved UV mapping visualization to debug_uv_mapping.png")
    
    # Process each view with progress tracking
    progress_bar = tqdm(
        range(len(observations)), 
        desc="Baking texture from views",
        disable=not verbose,
        unit="view",
        leave=True
    )
    
    total_valid_pixels_all_views = 0
    
    for view_idx in progress_bar:
        # Update progress bar description with current view info
        progress_bar.set_description(f"Processing view {view_idx+1}/{len(observations)}")
        
        # Get current view's camera
        current_camera = FoVPerspectiveCameras(
            R=R[view_idx:view_idx+1],
            T=T[view_idx:view_idx+1],
            fov=40.0,
            device=device
        )
        
        # Debug: Check camera transformation
        if debug and view_idx < 3:  # Debug first 3 views
            print(f"\nView {view_idx} camera debug:")
            print(f"  - R: {current_camera.R[0]}")
            print(f"  - T: {current_camera.T[0]}")
            print(f"  - Original extrinsics:\n{extrinsics[view_idx]}")
            print(f"  - Original intrinsics:\n{intrinsics[view_idx]}")
        
        # Rasterize mesh from current viewpoint
        rasterizer = MeshRasterizer(cameras=current_camera, raster_settings=raster_settings)
        fragments = rasterizer(mesh_pytorch3d)
        
        # Get pixel to face mapping
        pix_to_face = fragments.pix_to_face[0, ..., 0]  # [H, W]
        bary_coords = fragments.bary_coords[0, ..., 0, :]  # [H, W, 3]
        
        # Debug: Check rasterization results
        if debug and view_idx < 3:
            visible_faces = (pix_to_face >= 0).sum().item()
            print(f"  - Visible faces in view: {visible_faces} / {pix_to_face.numel()}")
            print(f"  - Pix_to_face range: {pix_to_face.min().item()} to {pix_to_face.max().item()}")
            print(f"  - Bary_coords range: {bary_coords.min().item()} to {bary_coords.max().item()}")
        
        # Get current observation and mask
        obs = observations[view_idx]  # [H, W, 3]
        mask = masks[view_idx]  # [H, W]
        
        # Debug: Check observation and mask
        if debug and view_idx < 3:
            obs_nonzero = (obs > 0).any(dim=-1).sum().item()
            mask_true = mask.sum().item()
            print(f"  - Observation non-zero pixels: {obs_nonzero} / {obs.numel()//3}")
            print(f"  - Mask true pixels: {mask_true} / {mask.numel()}")
            print(f"  - Observation range: {obs.min().item()} to {obs.max().item()}")
            
            # Save observation and mask images for first few views
            obs_img = (obs.cpu().numpy() * 255).astype(np.uint8)
            mask_img = (mask.cpu().numpy() * 255).astype(np.uint8)
            pix_to_face_vis = ((pix_to_face >= 0).float().cpu().numpy() * 255).astype(np.uint8)
            
            Image.fromarray(obs_img).save(f"debug_observation_{view_idx}.png")
            Image.fromarray(mask_img).save(f"debug_mask_{view_idx}.png")
            Image.fromarray(pix_to_face_vis).save(f"debug_rasterization_{view_idx}.png")
            print(f"  - Saved debug images: debug_observation_{view_idx}.png, debug_mask_{view_idx}.png, debug_rasterization_{view_idx}.png")
        
        # Create valid pixel mask (visible faces and valid observations)
        valid_pixels = (pix_to_face >= 0) & mask
        
        if not valid_pixels.any():
            progress_bar.set_postfix({"valid_pixels": 0, "status": "skipped"})
            if debug and view_idx < 3:
                print(f"  - WARNING: No valid pixels found!")
                print(f"    - Visible faces: {(pix_to_face >= 0).sum().item()}")
                print(f"    - Mask coverage: {mask.sum().item()}")
                print(f"    - Intersection: {valid_pixels.sum().item()}")
            continue
        
        # Update progress with valid pixel count
        num_valid = valid_pixels.sum().item()
        total_valid_pixels_all_views += num_valid
        progress_bar.set_postfix({"valid_pixels": num_valid, "status": "processing"})
        
        if debug and view_idx < 3:
            print(f"  - Valid pixels found: {num_valid}")
        
        # Get valid face indices and barycentric coordinates
        valid_faces = pix_to_face[valid_pixels]  # [N_valid]
        valid_bary = bary_coords[valid_pixels]   # [N_valid, 3]
        valid_colors = obs[valid_pixels]         # [N_valid, 3]
        
        # Debug: Check valid data
        if debug and view_idx < 3:
            print(f"  - Valid faces range: {valid_faces.min().item()} to {valid_faces.max().item()}")
            print(f"  - Valid bary range: {valid_bary.min().item()} to {valid_bary.max().item()}")
            print(f"  - Valid colors range: {valid_colors.min().item()} to {valid_colors.max().item()}")
        
        # Get UV coordinates for valid pixels
        face_uvs = uvs[faces[valid_faces]]  # [N_valid, 3, 2]
        
        # Interpolate UV coordinates using barycentric coordinates
        uv_coords = torch.sum(face_uvs * valid_bary.unsqueeze(-1), dim=1)  # [N_valid, 2]
        
        # Debug: Check UV interpolation
        if debug and view_idx < 3:
            print(f"  - UV coords range: {uv_coords.min(dim=0)[0]} to {uv_coords.max(dim=0)[0]}")
            print(f"  - Face UVs shape: {face_uvs.shape}")
        
        # Convert UV coordinates to texture pixel coordinates
        tex_coords = (uv_coords * (texture_size - 1)).long()
        tex_coords = torch.clamp(tex_coords, 0, texture_size - 1)
        
        # Debug: Check texture coordinate conversion
        if debug and view_idx < 3:
            print(f"  - Texture coords range: {tex_coords.min(dim=0)[0]} to {tex_coords.max(dim=0)[0]}")
            print(f"  - Texture coords shape: {tex_coords.shape}")
            
            # Create a visualization of where pixels are being mapped
            tex_vis = torch.zeros((texture_size, texture_size, 3), device=device)
            for i in range(min(100, len(tex_coords))):  # Sample first 100 pixels
                u, v = tex_coords[i]
                v_flipped = texture_size - 1 - v
                tex_vis[v_flipped, u] = valid_colors[i]
            
            tex_vis_img = (tex_vis.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(tex_vis_img).save(f"debug_texture_mapping_{view_idx}.png")
            print(f"  - Saved texture mapping visualization: debug_texture_mapping_{view_idx}.png")
        
        # Accumulate colors in texture map with sub-progress for large pixel counts
        if len(tex_coords) > 10000:  # Show sub-progress for views with many pixels
            pixel_progress = tqdm(
                range(len(tex_coords)), 
                desc=f"  Accumulating pixels for view {view_idx+1}",
                disable=not verbose,
                leave=False,
                unit="pixel"
            )
            for i in pixel_progress:
                u, v = tex_coords[i]
                # Flip V coordinate (OpenGL convention)
                v = texture_size - 1 - v
                
                color = valid_colors[i]
                weight = 1.0
                
                # Weighted average for blending
                current_weight = weight_map[v, u, 0]
                if current_weight > 0:
                    # Blend with existing color
                    total_weight = current_weight + weight
                    texture_map[v, u] = (texture_map[v, u] * current_weight + color * weight) / total_weight
                    weight_map[v, u, 0] = total_weight
                else:
                    # First color for this pixel
                    texture_map[v, u] = color
                    weight_map[v, u, 0] = weight
            pixel_progress.close()
        else:
            # Fast path for views with fewer pixels
            for i in range(len(tex_coords)):
                u, v = tex_coords[i]
                # Flip V coordinate (OpenGL convention)
                v = texture_size - 1 - v
                
                color = valid_colors[i]
                weight = 1.0
                
                # Weighted average for blending
                current_weight = weight_map[v, u, 0]
                if current_weight > 0:
                    # Blend with existing color
                    total_weight = current_weight + weight
                    texture_map[v, u] = (texture_map[v, u] * current_weight + color * weight) / total_weight
                    weight_map[v, u, 0] = total_weight
                else:
                    # First color for this pixel
                    texture_map[v, u] = color
                    weight_map[v, u, 0] = weight
        
        progress_bar.set_postfix({"valid_pixels": num_valid, "status": "completed"})
    
    progress_bar.close()
    
    # Debug: Final statistics
    if debug or verbose:
        print(f"\n=== TEXTURE BAKING SUMMARY ===")
        print(f"Total valid pixels across all views: {total_valid_pixels_all_views}")
        print(f"Final texture coverage: {(weight_map[..., 0] > 0).sum().item()} / {texture_size * texture_size} pixels")
        coverage_percent = (weight_map[..., 0] > 0).sum().item() / (texture_size * texture_size) * 100
        print(f"Coverage percentage: {coverage_percent:.2f}%")
        
        if total_valid_pixels_all_views == 0:
            print("\n!!! WARNING: NO VALID PIXELS FOUND ACROSS ALL VIEWS !!!")
            print("Possible causes:")
            print("1. Camera transformation issues (OpenCV ↔ PyTorch3D conversion)")
            print("2. Mesh not visible in any camera view (scale/position mismatch)")
            print("3. UV coordinate issues (outside [0,1] range)")
            print("4. Observation mask too restrictive")
            print("5. Rasterization settings incompatible")
            print("6. Face indexing mismatch between mesh and UV coordinates")
        
        # Save final texture state
        final_texture_img = (texture_map.cpu().numpy() * 255).astype(np.uint8)
        weight_vis = (weight_map[..., 0].cpu().numpy() * 255 / (weight_map[..., 0].max().item() + 1e-8)).astype(np.uint8)
        weight_vis = np.stack([weight_vis, weight_vis, weight_vis], axis=-1)
        
        Image.fromarray(final_texture_img).save("debug_final_texture.png")
        Image.fromarray(weight_vis).save("debug_weight_map.png")
        print(f"Saved final texture and weight map to debug_final_texture.png and debug_weight_map.png")
    
    # Handle unsampled pixels by filling with nearest neighbors using memory-efficient approach
    unsampled_mask = (weight_map[..., 0] == 0)
    if unsampled_mask.any():
        num_unsampled = unsampled_mask.sum().item()
        
        with tqdm(
            total=num_unsampled, 
            desc="Filling unsampled texture pixels",
            disable=not verbose,
            unit="pixel",
            leave=False
        ) as inpaint_progress:
            
            # Get coordinates
            coords = torch.meshgrid(
                torch.arange(texture_size, device=device),
                torch.arange(texture_size, device=device),
                indexing='ij'
            )
            coords = torch.stack(coords, dim=-1).float()  # [H, W, 2]
            
            sampled_coords = coords[~unsampled_mask]  # [N_sampled, 2]
            sampled_colors = texture_map[~unsampled_mask]  # [N_sampled, 3]
            
            inpaint_progress.set_postfix({
                "unsampled": num_unsampled, 
                "sampled": len(sampled_coords)
            })
            
            if len(sampled_coords) > 0:
                # Use iterative dilation approach instead of distance computation for memory efficiency
                current_texture = texture_map.clone()
                current_mask = ~unsampled_mask
                
                # Iteratively fill pixels by dilating from known pixels
                kernel_size = 3
                iterations = 0
                max_iterations = max(texture_size // 4, 50)  # Reasonable upper bound
                
                while unsampled_mask.any() and iterations < max_iterations:
                    iterations += 1
                    prev_unsampled_count = unsampled_mask.sum().item()
                    
                    # Create dilation kernel (3x3 neighborhood)
                    new_texture = current_texture.clone()
                    new_mask = current_mask.clone()
                    
                    # Check 8-connected neighborhood for each unsampled pixel
                    unsampled_indices = torch.where(unsampled_mask)
                    
                    for idx in range(len(unsampled_indices[0])):
                        y, x = unsampled_indices[0][idx].item(), unsampled_indices[1][idx].item()
                        
                        # Check 3x3 neighborhood
                        neighbors_colors = []
                        neighbors_weights = []
                        
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                    
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < texture_size and 0 <= nx < texture_size and 
                                    current_mask[ny, nx]):
                                    
                                    # Weight by inverse distance
                                    dist = (dy**2 + dx**2)**0.5
                                    weight = 1.0 / dist
                                    neighbors_colors.append(current_texture[ny, nx] * weight)
                                    neighbors_weights.append(weight)
                        
                        # If we found sampled neighbors, interpolate
                        if neighbors_colors:
                            total_weight = sum(neighbors_weights)
                            interpolated_color = sum(neighbors_colors) / total_weight
                            new_texture[y, x] = interpolated_color
                            new_mask[y, x] = True
                    
                    # Update current state
                    current_texture = new_texture
                    current_mask = new_mask
                    unsampled_mask = ~current_mask
                    
                    # Update progress
                    newly_filled = prev_unsampled_count - unsampled_mask.sum().item()
                    inpaint_progress.update(newly_filled)
                    inpaint_progress.set_postfix({
                        "iteration": iterations,
                        "remaining": unsampled_mask.sum().item(),
                        "filled_this_iter": newly_filled
                    })
                    
                    # Early termination if no progress
                    if newly_filled == 0:
                        break
                
                # Update the final texture
                texture_map = current_texture
                
                # Handle any remaining unsampled pixels with fallback color
                remaining_unsampled = unsampled_mask.sum().item()
                if remaining_unsampled > 0:
                    # Use average color of sampled pixels as fallback
                    fallback_color = sampled_colors.mean(dim=0)
                    texture_map[unsampled_mask] = fallback_color
                    inpaint_progress.update(remaining_unsampled)
                    inpaint_progress.set_postfix({
                        "fallback_filled": remaining_unsampled,
                        "status": "completed"
                    })
    
    # Create final textured mesh
    texture_map_batch = texture_map.unsqueeze(0)  # [1, H, W, 3]
    
    textured_mesh = Meshes(
        verts=[vertices],
        faces=[faces],
        textures=TexturesUV(
            maps=texture_map_batch,
            faces_uvs=[faces],
            verts_uvs=[uvs]
        )
    )
    
    if debug:
        # Save texture map for debugging
        texture_img = (texture_map.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(texture_img).save("debug_texture.png")
        if verbose:
            print("Saved debug texture to debug_texture.png")
    
    # Convert to trimesh object
    if verbose:
        print("Converting to trimesh format...")
    
    # Convert tensors to numpy
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    uvs_np = uvs.cpu().numpy()
    texture_np = (texture_map.cpu().numpy() * 255).astype(np.uint8)
    
    # Rotate mesh from z-up to y-up coordinate system
    vertices_np = vertices_np @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    
    # Create texture image
    texture_img = Image.fromarray(texture_np)
    
    # Create PBR material with the baked texture
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=texture_img,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
    )
    
    # Create trimesh with texture
    trimesh_obj = trimesh.Trimesh(
        vertices=vertices_np, 
        faces=faces_np, 
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )
    
    # Save test GLB for debugging
    if debug or verbose:
        test_glb_path = "test.glb"
        trimesh_obj.export(test_glb_path)
        if verbose:
            print(f"Saved test GLB to {test_glb_path}")
    
    # Convert back to your mesh format
    # Create a new MeshExtractResult with the textured mesh
    result_mesh = MeshExtractResult(
        vertices=textured_mesh.verts_packed(),
        faces=textured_mesh.faces_packed(),
        vertex_attrs=None,  # We'll store texture info separately
        res=mesh.res if hasattr(mesh, 'res') else 64
    )
    
    # Store additional texture information as custom attributes
    result_mesh.uvs = uvs
    result_mesh.texture_map = texture_map
    result_mesh.pytorch3d_mesh = textured_mesh
    result_mesh.trimesh_obj = trimesh_obj  # Store the trimesh object
    
    if verbose:
        print(f"Texture baking completed. Final mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    
    return result_mesh




    

