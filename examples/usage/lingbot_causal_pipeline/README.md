# Experimental LingBot causal pipelines

This opt-in path runs consecutive LingBot video chunks with **same-timestep
noisy KV history**. It changes the existing clean-history inference algorithm.
The original LingBot checkpoints have not been quality-validated for this path;
compare generated videos before using it for serving.

Two static layouts share one execution protocol:

- `timestep`: keep each denoising forward together, assigning consecutive
  timesteps to ranks (TPP).
- `block`: partition the expanded timestep × Transformer-layer chain across
  ranks. Boundaries can fall inside a forward. This implements the stage-wise
  partitioning idea discussed by PhysisServe, not its VirtualRank optimizer or
  cross-stream batching. No PhysisServe source code is included.

The default `off` mode keeps the existing LingBot path.

## Launch

Use a Linux NVIDIA CUDA SGLang Diffusion installation and an existing LingBot
**causal DMD** checkpoint, such as
`robbyant/lingbot-world-v2-14b-causal-fast-diffusers`. The experimental path is
single-node, single-stream and eager; it does not compose with TP, SP, DP, CFG,
FSDP, layerwise offload, KV quantization, dynamic batching or unmerged LoRA.
Every denoising rank currently loads the full DiT weights, even in `block` mode.
Only KV storage is limited to the layers and timesteps owned by that rank.
Account for model weights, KV, conditioning and VAE memory when selecting GPUs.

Four timestep ranks, with VAE decoding on the last rank (four GPUs total):

```bash
sglang serve --model-path robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --num-gpus 4 --performance-mode manual \
  --causal-pipeline-mode timestep --causal-pipeline-parallel-size 4 \
  --causal-pipeline-window 4 --causal-vae-placement shared
```

Four timestep ranks plus one dedicated VAE rank (five GPUs total):

```bash
sglang serve --model-path robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --num-gpus 5 --performance-mode manual \
  --causal-pipeline-mode timestep --causal-pipeline-parallel-size 4 \
  --causal-pipeline-window 4 --causal-vae-placement dedicated
```

For a layer boundary inside each timestep, use eight denoising ranks:

```bash
sglang serve --model-path robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --num-gpus 8 --performance-mode manual \
  --causal-pipeline-mode block --causal-pipeline-parallel-size 8 \
  --causal-pipeline-window 8 --causal-vae-placement shared
```

`--causal-vae-rank` selects a logical denoising rank for shared decode. It
defaults to the last denoising rank. Dedicated decode uses the next rank after
all denoisers; that rank loads only the VAE. With dedicated placement,
`num_gpus = causal_pipeline_parallel_size + 1`.

## Submit a window

The existing `/v1/realtime_video/generate` WebSocket accepts the same LingBot
init and camera/prompt events. When the mode is enabled, the HTTP process
snapshots at most `--causal-pipeline-window` consecutive chunks and submits
them together. The final window is shortened to respect `max_chunks`.

Run the adjacent client from the repository root, using an image file:

```bash
python examples/usage/lingbot_causal_pipeline/client.py \
  --image examples/assets/example_image.png \
  --prompt 'Move slowly through a sunlit room.' \
  --chunks 8 --output lingbot-pipeline.mp4
```

Inside a window, each rank processes chunks in order. Hidden states travel
between layer partitions; latent states travel across timestep boundaries.
KV remains local and is indexed by session, timestep and layer. Noise is seeded
by `(seed, chunk_index, timestep)`, independently of layout, rank count and
window size.

Dedicated VAE decode consumes chunks while denoising continues. Neighbor sends
provide bounded backpressure. Shared VAE decode starts after the denoising
window has drained, avoiding a communication cycle when the chosen VAE rank
is upstream of the final denoiser. Both preserve the causal VAE state and output
order across windows.

**Delivery is windowed in this first implementation.** Decoded frames are held
on the host until the window completes, then sent in chunk order. New control
events apply to the next window. A slow WebSocket client prevents the next
window from being submitted. Cancellation stops new submissions; the in-flight
bounded window finishes before the queued session-release request is handled.
This is not yet a continuously admitted, per-completion streaming executor.

`chunk_stats.scheduler_forward_ms` reports the enclosing window's RPC time for
each chunk, with `scheduler_forward_scope="window"` and `window_chunks` identifying
its scope. It is not an individual denoise-stage duration. Do not interpret burst
arrival intervals within a returned window as steady-state DiT throughput.

The existing `profile` and `profile_all_stages` request options capture a window
under its first chunk's request ID. Denoising step metrics describe each rank's
local timestep/layer slice; they are not full-model or end-to-end chunk timings.
Worker `total_duration_ms` is measured once for the window, before output
materialization. Each chunk's metrics and performance report carry that same
duration with `total_duration_scope="window"` and `window_chunks`; do not sum
these repeated window totals as individual chunk execution times.

## Validation

CPU contract tests:

```bash
python test/registered/unit/multimodal_gen/test_causal_pipeline.py
python test/registered/unit/multimodal_gen/test_lingbot_causal_pipeline.py
```

Before reporting GPU performance or quality:

1. Keep prompts, first frames, seeds, dimensions, sampling schedule and
   attention backend fixed. Compare `off` with single-rank noisy-history mode
   for image quality, temporal consistency, prompt adherence and long-run drift.
2. Compare the same noisy-history model at degree 1 with timestep and block
   layouts. Check denoised latents and decoded frames across several windows,
   including prompt changes, final partial windows and session restarts. Do not
   assume bit-exact equality across different device kernels without measuring.
3. Run shared VAE on the first and last denoising ranks and run dedicated VAE.
   Exercise disconnects, invalid requests and a new session after failure.
4. Measure first-frame latency, complete-request time, denoising intervals and
   decoded output throughput separately. Include VAE and host/WebSocket costs
   in the end-to-end numbers.

GPU execution and checkpoint quality have not been validated by the local CPU
tests. Selective layer loading, continuous admission, asynchronous output
delivery, cross-stream batching and cost-based virtual-rank placement remain
future work.
