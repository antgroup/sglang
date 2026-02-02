# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import os

# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from einops import rearrange

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPVisionConfig,
    T5Config,
)
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [
            torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds_tensor


@dataclass
class WanI2VCommonConfig(PipelineConfig):
    # for all wan i2v pipelines
    def adjust_num_frames(self, num_frames):
        vae_scale_factor_temporal = self.vae_config.arch_config.scale_factor_temporal
        if num_frames % vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames // vae_scale_factor_temporal * vae_scale_factor_temporal + 1
            )
            return num_frames
        return num_frames


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    task_type: ModelTaskType = ModelTaskType.T2V
    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)

    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 3.0

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    # WanConfig-specific added parameters

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class TurboWanT2V480PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [988, 932, 852, 608]
    )


@dataclass
class WanT2V720PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 14B 720P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class WanI2V480PConfig(WanT2V480PConfig, WanI2VCommonConfig):
    """Base configuration for Wan I2V 14B 480P pipeline architecture."""

    max_area: int = 480 * 832
    # WanConfig-specific parameters with defaults
    task_type: ModelTaskType = ModelTaskType.I2V
    # Precision for each component
    image_encoder_config: EncoderConfig = field(default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"

    image_encoder_extra_args: dict = field(
        default_factory=lambda: dict(
            output_hidden_states=True,
        )
    )

    def postprocess_image(self, image):
        return image.hidden_states[-2]

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class WanI2V720PConfig(WanI2V480PConfig):
    """Base configuration for Wan I2V 14B 720P pipeline architecture."""

    max_area: int = 720 * 1280
    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class TurboWanI2V720Config(WanI2V720PConfig):
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [996, 932, 852, 608]
    )
    boundary_ratio: float | None = 0.9

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class FastWan2_1_T2V_480P_Config(WanT2V480PConfig):
    """Base configuration for FastWan T2V 1.3B 480P pipeline architecture with DMD"""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )


@dataclass
class Wan2_2_TI2V_5B_Config(WanT2V480PConfig, WanI2VCommonConfig):
    flow_shift: float | None = 5.0
    task_type: ModelTaskType = ModelTaskType.TI2V
    expand_timesteps: bool = True
    # ti2v, 5B
    vae_stride = (4, 16, 16)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        F = num_frames
        z_dim = self.vae_config.arch_config.z_dim
        vae_stride = self.vae_stride
        oh = batch.height
        ow = batch.width
        shape = (batch_size, z_dim, F, oh // vae_stride[1], ow // vae_stride[2])
        return shape

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.expand_timesteps = self.expand_timesteps


@dataclass
class FastWan2_2_TI2V_5B_Config(Wan2_2_TI2V_5B_Config):
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )


@dataclass
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    flow_shift: float | None = 12.0
    boundary_ratio: float | None = 0.875

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class Wan2_2_I2V_A14B_Config(WanI2V480PConfig):
    flow_shift: float | None = 5.0
    boundary_ratio: float | None = 0.900

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dit_config.boundary_ratio = self.boundary_ratio


# =============================================
# ============= Causal Self-Forcing =============
# =============================================
@dataclass
class SelfForcingWanT2V480PConfig(WanT2V480PConfig):
    is_causal: bool = True
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True


# lingbot world
@dataclass
class LingBotWorldI2VPConfig(WanI2V720PConfig):
    boundary_ratio: float | None = 0.947
    flow_shift: float | None = 10.0
    vae_stride = (4, 8, 8)

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        dit_cond_dict = None
        action_path = "/ossfs/workspace/lingbot-world/examples/00/"
        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))  # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = min(batch.num_frames, len_c2ws)
            c2ws = c2ws[:frame_num]
            Ks = torch.from_numpy(
                np.load(os.path.join(action_path, "intrinsics.npy"))
            ).float()

            # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
            Ks = get_Ks_transformed(
                Ks,
                height_org=480,
                width_org=832,
                height_resize=batch.height,
                width_resize=batch.width,
                height_final=batch.height,
                width_final=batch.width,
            )
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(batch.latents.device)
            Ks = Ks.to(batch.latents.device)
            c2ws_plucker_emb = get_plucker_embeddings(
                c2ws_infer, Ks, batch.height, batch.width
            )
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                c1=int(batch.height // batch.latents.shape[2]),
                c2=int(batch.width // batch.latents.shape[3]),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "b (f h w) c -> b c f h w",
                f=batch.latents.shape[1],
                h=batch.latents.shape[2],
                w=batch.latents.shape[3],
            ).to(self.param_dtype)
            dit_cond_dict = {
                "c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0),
            }
        return {"dit_cond_dict": dit_cond_dict}


def get_Ks_transformed(
    Ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
):
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # [f, 1]

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    cx_final = cx_resize - crop_offset_x
    cy_final = cy_resize - crop_offset_y

    Ks_transformed = torch.zeros_like(Ks)
    Ks_transformed[:, 0:1] = fx_resize
    Ks_transformed[:, 1:2] = fy_resize
    Ks_transformed[:, 2:3] = cx_final
    Ks_transformed[:, 3:4] = cy_final

    return Ks_transformed


from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    # interpolate translation
    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    # interpolate rotation
    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    # ensure there is no sudden change in qw
    quats = src_quat_vec.as_quat().copy()  # [N, 4]
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # compute pose between i and i+1
        relative_poses_framewise = torch.bmm(
            SE3_inverse(relative_poses[:-1]), relative_poses[1:]
        )
        relative_poses[1:] = relative_poses_framewise
    if (
        normalize_trans
    ):  # note refer to camctrl2: "we scale the coordinate inputs to roughly 1 standard deviation to simplify model learning."
        translations = relative_poses[:, :3, 3]  # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        # only normlaize when moving
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    Rot = T[:, :3, :3]  # [B,3,3]
    trans = T[:, :3, 3:]  # [B,3,1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(
        T.shape[0], 1, 1
    )
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
):
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(
        n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype
    )  # [f, h*w, 2]
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # [f, 1]

    i = grid_xy[..., 0]  # [f, h*w]
    j = grid_xy[..., 1]  # [f, h*w]
    zs = torch.ones_like(i)  # [f, h*w]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)  # [f, h*w, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)  # [f, h*w, 3]

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)  # [f, h*w, 3]
    rays_o = c2ws_mat[:, :3, 3]  # [f, 3]
    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [f, h*w, 3]
    # rays_dxo = torch.cross(rays_o, rays_d, dim=-1) # [f, h*w, 3]
    # note refer to: apt2
    plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)  # [f, h*w, 6]
    plucker_embeddings = plucker_embeddings.view(
        [n_frames, height, width, 6]
    )  # [f*h*w, 6]
    return plucker_embeddings


@torch.no_grad()
def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device="cuda",
    dtype=torch.float32,
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias  # [h*w, 2]
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1)  # [f, h*w, 2]
    return grid_xy
