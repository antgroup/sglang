from collections import deque
from typing import Dict, List, Optional, Union

import torch
from diffusers.utils import is_ftfy_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

if is_ftfy_available():
    import ftfy

import html

import regex as re


class KreaRealtimeVideoBeforeDenoisingStage(PipelineStage):
    def __init__(self, text_encoder, tokenizer, transformer, vae) -> None:
        super().__init__()
        self.vae = vae
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.video_processor = VideoProcessor(vae_scale_factor=8)

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
        # step1 TextEncoder
        device = get_local_torch_device()
        strength = 1 if batch.input_video is None else 0.3
        num_inference_steps = 4
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            batch.prompt, device, 1, False, batch.negative_prompt
        )
        transformer_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        batch.prompt_embeds = prompt_embeds.contiguous()
        block_idx = batch.block_idx
        num_frames_per_block = self.transformer.config.arch_config.num_frames_per_block
        kv_cache_num_frames = self.transformer.config.arch_config.kv_cache_num_frames
        frame_seq_length = self.transformer.config.arch_config.frame_seq_length
        # step2 set timesteps
        batch.timesteps, batch.sigmas = self.prepare_timesteps(5, strength, 4)
        # step3 prepare latents
        # video to video
        if (
            batch.session.input_frames_cache is not None
            and batch.input_video is not None
        ):
            batch.session.input_frames_cache.extend(batch.input_video)
            video = (
                self.video_processor.preprocess(
                    list(batch.session.input_frames_cache),
                    batch.height,
                    batch.width,
                )
                .unsqueeze(0)
                .to(self.vae.device, self.vae.dtype)
            )
            batch.current_start_frame = block_idx * num_frames_per_block
            init_latents = self.encode_frames(
                video,
                self.vae.dtype,
                self.vae.device,
                None,
            )
            init_latents = init_latents[:, :, -num_frames_per_block:]

            strength = batch.timesteps[0] / 1000.0
            noise = randn_tensor(
                init_latents.shape,
                device=self.transformer.device,
                dtype=transformer_dtype,
                generator=batch.generator,
            )

            init_latents = init_latents * (1.0 - strength) + noise * strength
            init_latents = init_latents.to(transformer_dtype).contiguous()

            batch.latents = init_latents
        else:
            # text to video
            init_latents = self.prepare_latents(
                1,
                server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial,
                server_args.pipeline_config.dit_config.arch_config.num_channels_latents,
                batch.height,
                batch.width,
                batch.num_blocks,
                num_frames_per_block,
                transformer_dtype,
                self.transformer.device,
                batch.generator,
                batch.latents,
            )
            init_latents = init_latents.contiguous()
            start_frame = block_idx * num_frames_per_block
            end_frame = start_frame + num_frames_per_block

            # Extract single block from full latent buffer
            # final_latents shape: [B, C, total_frames, H, W]
            # Extract frames along the time dimension (dim=2)
            batch.latents = init_latents[:, :, start_frame:end_frame, :, :]
            batch.current_start_frame = start_frame
        # step4 setup kvcache
        # Get existing caches if they exist
        kv_cache = batch.session.kv_cache
        crossattn_cache = batch.session.crossattn_cache
        if crossattn_cache is None or batch.update_prompt_embeds:
            batch.session.crossattn_cache = _initialize_crossattn_cache(
                len(self.transformer.blocks),
                self.transformer.num_attention_heads,
                self.transformer.num_attention_heads
                * self.transformer.attention_head_dim,
                crossattn_cache,
                1,
                transformer_dtype,
                self.transformer.device,
            )

        batch.local_attn_size = kv_cache_num_frames + num_frames_per_block
        for block in self.transformer.blocks:
            block.attn1.local_attn_size = -1
        for block in self.transformer.blocks:
            block.attn1.num_frame_per_block = num_frames_per_block

        batch.session.kv_cache = _initialize_kv_cache(
            len(self.transformer.blocks),
            self.transformer.num_attention_heads,
            self.transformer.num_attention_heads * self.transformer.attention_head_dim,
            kv_cache,
            1,
            transformer_dtype,
            self.transformer.device,
            batch.local_attn_size,
            frame_seq_length,
        )
        # step5 recomputeKVCache
        if batch.block_idx != 0:
            context_frames = self.get_context_frames(
                kv_cache_num_frames, num_frames_per_block, batch
            )
            block_mask = self.transformer._prepare_blockwise_causal_attn_mask(
                self.transformer.device,
                num_frames=context_frames.shape[2],
                frame_seqlen=frame_seq_length,
                num_frame_per_block=num_frames_per_block,
                local_attn_size=-1,
            )
            self.transformer.block_mask = block_mask
            context_timestep = torch.zeros(
                (context_frames.shape[0], context_frames.shape[2]),
                device=self.transformer.device,
                dtype=torch.int64,
            )
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=batch,
            ):
                self.transformer(
                    hidden_states=context_frames.to(transformer_dtype),
                    timestep=context_timestep,
                    encoder_hidden_states=batch.prompt_embeds.to(transformer_dtype),
                    kv_cache=batch.session.kv_cache,
                    crossattn_cache=batch.session.crossattn_cache,
                    current_start=0,  # when updating the kv cache with block_mask the current_start is unused
                    cache_start=None,
                )
            self.transformer.block_mask = None
        return batch

    def get_context_frames(self, kv_cache_num_frames, num_frames_per_block, batch):
        current_kv_cache_num_frames = kv_cache_num_frames
        total_frames_generated = (batch.block_idx - 1) * num_frames_per_block

        if total_frames_generated < current_kv_cache_num_frames:
            context_frames = batch.session.current_denoised_latents[
                :, :, :current_kv_cache_num_frames
            ]

        else:
            context_frames = batch.session.current_denoised_latents
            context_frames = context_frames[:, :, 1:][
                :, :, -current_kv_cache_num_frames + 1 :
            ]
            first_frame_latent = self.prepare_frame_latents(
                frames=batch.session.frame_cache_context[0].half()
            )
            first_frame_latent = first_frame_latent.to(batch.latents)
            context_frames = torch.cat((first_frame_latent, context_frames), dim=2)

        return context_frames

    def prepare_frame_latents(self, frames):
        self.vae._enc_feat_map = [None] * 55
        latents = retrieve_latents(self.vae.encode(frames), sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std

        return latents

    def encode_frames(
        self,
        video: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if latents is not None:
            return latents.to(device, dtype)

        if not hasattr(self.vae, "_enc_feat_map"):
            self.vae.clear_cache()
        else:
            self.vae._enc_feat_map = [None] * 55

        init_latents = [
            retrieve_latents(
                self.vae.encode(vid.unsqueeze(0).transpose(2, 1)),
                sample_mode="argmax",
            )
            for vid in video
        ]
        init_latents = torch.cat(init_latents, dim=0).to(dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device, dtype)
        init_latents = (init_latents - latents_mean) * latents_std

        return init_latents

    def encode_prompt(
        self,
        prompt: str,
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        prepare_unconditional_embeds: bool = True,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            prepare_unconditional_embeds (`bool`):
                whether to use prepare unconditional embeddings or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of text tokens to be used for the generation process.
        """
        device = device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt, max_sequence_length, device
            )

        if prepare_unconditional_embeds and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                negative_prompt, max_sequence_length, device
            )

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        if prepare_unconditional_embeds:
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_videos_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

        return prompt_embeds, negative_prompt_embeds

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int,
        device: torch.device,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        return prompt_embeds

    def prepare_timesteps(self, shift, strength, num_inference_steps):
        sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas.to(self.transformer.device) * 1000.0
        zero_padded_timesteps = torch.cat(
            [
                timesteps,
                torch.tensor([0], device=self.transformer.device),
            ]
        )
        denoising_steps = torch.linspace(
            strength * 1000,
            0,
            num_inference_steps,
            dtype=torch.float32,
        ).to(torch.long)
        timesteps = zero_padded_timesteps[1000 - denoising_steps]

        return timesteps, sigmas

    def prepare_latents(
        self,
        batch_size: int,
        vae_scale_factor: float = 8.0,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_blocks: int = 9,
        num_frames_per_block: int = 3,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = num_blocks * num_frames_per_block
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // vae_scale_factor,
            int(width) // vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(
            shape,
            generator=generator,
            device=self.transformer.device,
            dtype=dtype,
        )
        return latents


class KreaRealtimeVideoDenoisingStage(PipelineStage):
    def __init__(self, transformer, scheduler, vae):
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.video_processor = VideoProcessor(vae_scale_factor=8)

    def forward(self, batch: Req, server_args: ServerArgs):
        latents = batch.latents
        prompt_embeds = batch.prompt_embeds
        kv_cache = batch.session.kv_cache
        crossattn_cache = batch.session.crossattn_cache
        current_start_frame = batch.current_start_frame
        timesteps = batch.timesteps
        sigmas = batch.sigmas
        num_inference_steps = batch.num_inference_steps
        kv_cache_num_frames = self.transformer.config.arch_config.kv_cache_num_frames
        num_frames_per_block = self.transformer.config.arch_config.num_frames_per_block
        frame_seq_length = self.transformer.config.arch_config.frame_seq_length
        self.transformer_dtype = PRECISION_TO_TYPE[
            server_args.pipeline_config.dit_precision
        ]
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        # Iterate over each timestep
        for i, t in enumerate(timesteps):
            # Step 1: predict noise
            noise_pred = self.predict_noise(
                latents=latents,
                timestep=t,
                timestep_index=i,
                prompt_embeds=prompt_embeds,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=current_start_frame,
                kv_cache_num_frames=kv_cache_num_frames,
                num_frames_per_block=num_frames_per_block,
                seq_length=32760,
                frame_seq_length=frame_seq_length,
                batch=batch,
            )

            # Step 2: update latents
            latents = self.update_latents(
                latents=latents,
                noise_pred=noise_pred,
                timestep=t,
                all_timesteps=timesteps,
                sigmas=sigmas,
            )

            # Step 3: if not the last step, add noise for the next timestep
            # This is a common practice in samplers like DDIM
            if i < (num_inference_steps - 1):
                next_timestep = timesteps[i + 1]

                # Prepare noise
                noise = randn_tensor(
                    latents.transpose(1, 2).squeeze(0).shape,
                    device=latents.device,
                    dtype=latents.dtype,
                    generator=batch.generator,
                )

                # Add noise to latents
                latents = (
                    self.add_noise(
                        sample=latents.transpose(1, 2).squeeze(0),
                        noise=noise,
                        timestep=next_timestep.expand(
                            latents.shape[0], num_frames_per_block
                        ),
                        all_timesteps=timesteps,
                        sigmas=sigmas,
                    )
                    .unsqueeze(0)
                    .transpose(1, 2)
                )
        batch.latents = latents

        if batch.session.frame_cache_context is None:
            frame_cache_len = 1 + (kv_cache_num_frames - 1) * 4
            batch.session.frame_cache_context = deque(maxlen=frame_cache_len)

        # Disable clearing cache
        if batch.block_idx == 0:
            self.vae.clear_cache()
            self.vae.clear_cache = lambda: None
            self.vae._feat_map = [None] * 55

        if batch.block_idx != 0:
            self.vae._feat_map = batch.decoder_cache

        latents = batch.latents.to(self.vae.device)

        # Create tensors directly on target device and dtype to avoid redundant conversions
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)

        latents = latents / latents_std + latents_mean
        latents = latents.to(vae_dtype)

        videos = self.vae.decode(latents)

        batch.session.decoder_cache = self.vae._feat_map
        batch.session.frame_cache_context.extend(videos.split(1, dim=2))
        videos = self.video_processor.postprocess_video(videos, output_type="pil")

        output_batch = OutputBatch(
            output=videos,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            metrics=batch.metrics,
        )
        return output_batch

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        all_timesteps: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to a sample.

        Uses the formula: noisy_sample = (1 - sigma) * sample + sigma * noise

        Args:
            sample: clean sample
            noise: noise tensor
            timestep: current timestep
            all_timesteps: all timesteps
            sigmas: sigma lookup table

        Returns:
            The noisy sample.
        """
        # Ensure sigmas is on the correct device
        sigmas = sigmas.to(all_timesteps.device)

        # Handle multi-dimensional timestep
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)

        # Find the sigma corresponding to each timestep
        timestep_id = torch.argmin(
            (all_timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1)

        # Add noise
        noisy_sample = (
            (1 - sigma.double()) * sample.double() + sigma.double() * noise.double()
        ).type_as(noise)

        return noisy_sample

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        timestep_index: int,
        prompt_embeds: torch.Tensor,
        kv_cache: torch.Tensor,
        crossattn_cache: torch.Tensor,
        current_start_frame: int,
        kv_cache_num_frames: int,
        num_frames_per_block: int,
        seq_length: int,
        frame_seq_length: int,
        batch: Req,
    ) -> torch.Tensor:
        """
        Predict noise using the Transformer.

        Args:
            latents: Current latent representation [B, C, F, H, W]
            timestep: Current timestep
            timestep_index: Current timestep index
            prompt_embeds: Text embeddings
            kv_cache: Transformer KV cache
            crossattn_cache: Cross-attention cache
            current_start_frame: Start frame index of the current video block
            kv_cache_num_frames: Number of frames stored in the KV cache
            num_frames_per_block: Number of frames per block
            seq_length: Total sequence length
            frame_seq_length: Sequence length per frame
            batch: Request batch

        Returns:
            noise_pred: Predicted noise
        """
        # Compute the effective start frame (not exceeding cache capacity)
        start_frame = min(current_start_frame, kv_cache_num_frames)
        prompt_embeds = prompt_embeds.to(self.transformer_dtype)
        # Call the Transformer to predict noise
        with set_forward_context(
            current_timestep=timestep_index,
            attn_metadata=None,
            forward_batch=batch,
        ):
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep.expand(latents.shape[0], num_frames_per_block),
                encoder_hidden_states=prompt_embeds,
                kv_cache=kv_cache,
                seq_len=seq_length,
                crossattn_cache=crossattn_cache,
                current_start=start_frame * frame_seq_length,
                cache_start=None,
            )

        return noise_pred

    def update_latents(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        all_timesteps: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update latents based on the predicted noise.

        Uses: latents = latents - sigma * noise_pred

        Args:
            latents: Current latent representation
            noise_pred: Predicted noise
            timestep: Current timestep
            all_timesteps: Tensor of all timesteps
            sigmas: Sigma values corresponding to each timestep

        Returns:
            Updated latents
        """
        # Find the index corresponding to the current timestep
        timestep_id = torch.argmin((all_timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id]

        # Use float64 for numerical stability
        latents_dtype = latents.dtype
        latents = (latents.double() - sigma_t.double() * noise_pred.double()).to(
            latents_dtype
        )

        return latents


def _initialize_crossattn_cache(
    num_transformer_blocks,
    num_heads,
    dim,
    crossattn_cache_existing: Optional[List[Dict]],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Initialize a Per-GPU cross-attention cache for the Wan model.
    Mirrors causal_inference.py:315-338
    """
    crossattn_cache = []

    k_shape = [batch_size, 512, num_heads, dim // num_heads]
    v_shape = [batch_size, 512, num_heads, dim // num_heads]

    # Check if we can reuse existing cache
    if (
        crossattn_cache_existing
        and len(crossattn_cache_existing) > 0
        and list(crossattn_cache_existing[0]["k"].shape) == k_shape
        and list(crossattn_cache_existing[0]["v"].shape) == v_shape
    ):
        for i in range(num_transformer_blocks):
            crossattn_cache_existing[i]["k"].zero_()
            crossattn_cache_existing[i]["v"].zero_()
            crossattn_cache_existing[i]["is_init"] = False
        return crossattn_cache_existing
    else:
        # Create new cache
        for _ in range(num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(k_shape, dtype=dtype, device=device).contiguous(),
                    "v": torch.zeros(v_shape, dtype=dtype, device=device).contiguous(),
                    "is_init": False,
                }
            )
        return crossattn_cache


def _initialize_kv_cache(
    num_transformer_blocks,
    num_heads,
    dim,
    kv_cache_existing: Optional[List[Dict]],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    local_attn_size: int,
    frame_seq_length: int,
):
    """
    Initialize a Per-GPU KV cache for the Wan model.
    Mirrors causal_inference.py:279-313
    """
    kv_cache = []

    # Calculate KV cache size
    if local_attn_size != -1:
        # Use the local attention size to compute the KV cache size
        kv_cache_size = local_attn_size * frame_seq_length
    else:
        # Use the default KV cache size
        kv_cache_size = 32760

    # Get transformer config
    k_shape = [batch_size, kv_cache_size, num_heads, dim // num_heads]
    v_shape = [batch_size, kv_cache_size, num_heads, dim // num_heads]

    # Check if we can reuse existing cache
    if (
        kv_cache_existing
        and len(kv_cache_existing) > 0
        and list(kv_cache_existing[0]["k"].shape) == k_shape
        and list(kv_cache_existing[0]["v"].shape) == v_shape
    ):
        for i in range(num_transformer_blocks):
            kv_cache_existing[i]["k"].zero_()
            kv_cache_existing[i]["v"].zero_()
            kv_cache_existing[i]["global_end_index"] = 0
            kv_cache_existing[i]["local_end_index"] = 0
        return kv_cache_existing
    else:
        # Create new cache
        for _ in range(num_transformer_blocks):
            kv_cache.append(
                {
                    "k": torch.zeros(k_shape, dtype=dtype, device=device).contiguous(),
                    "v": torch.zeros(v_shape, dtype=dtype, device=device).contiguous(),
                    "global_end_index": 0,
                    "local_end_index": 0,
                }
            )
        return kv_cache


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text
