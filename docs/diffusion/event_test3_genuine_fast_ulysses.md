# event_test3 genuine fast-ulysses 部署与实测

## 结论与范围

这里的 `fast_ulysses` 指
[triple-mu/fast-ulysses](https://github.com/triple-mu/fast-ulysses) 的
CUDA/NVSHMEM 扩展，不是 SGLang 自带的 `sgl_p2p` kernel。生产日志必须同时出现：

```text
Initialized genuine fast_ulysses backend ...
genuine fast_ulysses collective active ...
```

本适配只覆盖单机、Ulysses group 等于 `WORLD`、2～8 卡、uniform split、
`[B,S,H,D]`、BF16/FP16、`head_dim=2` 的同步 A2A。LingBot 8 卡 sequence
shard 满足这些条件；variable split、CUDA graph capture 和 subgroup 不会静默回退，
强制选择该 backend 时会直接报错。

## 为什么不能只安装上游包

上游按 tag 复用 NVSHMEM symmetric output buffer。某个 rank 较快时，可能在另一
rank 仍读取上一代结果时开始覆盖同一 tag。同步 A2A 末尾的 barrier 只保证本代
写入完成，不保证所有 rank 的消费者都完成。

`scripts/fast_ulysses/patches/0001-stream-ordered-pre-write-barrier.patch`
给上游 binding 增加 `pre_write_barrier`。最保守的 tag pool size 1 会在每次复用
前执行 barrier。生产配置使用 32 个轮转 tag；第一次循环只分配不复用，之后每
32 次调用在 wrap 点执行一次 barrier：

1. 本 rank 上一代消费者先完成；
2. 所有 rank 到达 pre-write barrier；
3. 上一轮全部 tag 都可以安全复用；
4. 上游原有 post-write barrier 继续保证本代结果可读。

barrier 排在调用者当前 CUDA stream 上；每个 rank 到达 wrap 点时，该 stream
已经消费完前一轮所有输出。所有 rank 到齐后，下一轮才开始复用。LingBot 两类
实测 buffer 使用 32 个 tag 约占每卡 0.72 GiB，低于上游默认 2 GiB symmetric
pool。这样既保留 borrowed buffer、避免每次 clone，也把全局 barrier 从每次
collective 降到每个 slot 每 32 次一次。

另一个独立问题是原 backend router 在 40 层 × 4 denoise step 的热路径中反复
构造 transaction/spec。固定选择 `nccl` 或 `fast_ulysses` 时，现在只对每种
fast shape/mode 做一次跨 rank 预检，后续直接进入 transport；`auto` 和
`sgl_p2p` 仍使用 transaction，保持同一 attention 内不切换 backend 的语义。

## 已验证环境

| 项目 | 值 |
|---|---|
| SGLang branch | `codex/event-test3-ulysses-a2a-backend` |
| SGLang commit 基线 | `7165f649b541c62ef25ef596b77a15c153c2378a` |
| GPU | 8 × NVIDIA L20X，全 NVLink，PyTorch capability `sm90` |
| PyTorch | 2.9.1+cu128 |
| CUDA toolkit | 12.6.85 |
| `sgl_kernel` | 0.4.1，原环境已有 |
| fast-ulysses | 0.1.0，commit `6e5dcb24dc44e781ac3091d1d9b3f9fef314fb87` |
| NVSHMEM | NVIDIA RPM 3.7.2/cu12 |
| 模型 | `/home/admin/lingbot-world-fast-diffusers` |

原 `/opt/conda/deep_ep_deps` 中 NVSHMEM 是 3.1.7，不能作为这次构建的
`NVSHMEM_HOME`。

## 一次性安装

在 Alibaba Cloud Linux 3 / RHEL8 兼容环境中：

```bash
cd /home/admin/sglang

dnf install -y \
  libnvshmem3-cuda-12-3.7.2-1 \
  libnvshmem3-devel-cuda-12-3.7.2-1 \
  libnvshmem3-static-cuda-12-3.7.2-1

bash scripts/fast_ulysses/install.sh \
  2>&1 | tee /home/admin/fast_ulysses_experiment/logs/install.log
```

static 包不能省略：NVIDIA 的 CMake config 即使最终 target 只链接 host library，
也会检查 `libnvshmem_device.a` 是否存在。

安装脚本会：

- 固定 checkout 上游 commit；
- 创建 `/home/admin/nvshmem-3.7.2-cu12` 规范化 prefix；
- 应用 stream-ordered pre-write barrier 和 setuptools 75 兼容补丁；
- 只为 `sm90` 编译；
- editable install 到当前 Python 环境；
- 验证 `UlyssesGroup.pre_write_barrier` 存在。

## CUDA driver 搜索顺序

安装 RPM 会执行 `ldconfig`。该镜像的配置把
`/usr/local/cuda/compat/libcuda.so.1` 排在宿主 570 driver 前，直接启动 Python
可能报 CUDA error 803。不要使用 compat stub；启动与测试脚本会精确预加载宿主
driver：

```bash
export LD_PRELOAD="/usr/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="/home/admin/nvshmem-3.7.2-cu12/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

验证扩展确实链接 3.7.2：

```bash
FAST_ULYSSES_SO="$(find /home/admin/fast-ulysses/fast_ulysses \
  -maxdepth 1 -name '_C*.so' -print -quit)"
LD_PRELOAD=/usr/lib64/libcuda.so.1 \
LD_LIBRARY_PATH=/home/admin/nvshmem-3.7.2-cu12/lib \
ldd "${FAST_ULYSSES_SO}" | grep nvshmem
```

期望路径是：

```text
/home/admin/nvshmem-3.7.2-cu12/lib/libnvshmem_host.so.3
```

## 正确性与 microbenchmark

```bash
bash /home/admin/sglang/scripts/fast_ulysses/run_cpu_tests.sh
bash /home/admin/sglang/scripts/fast_ulysses/run_microbenchmark.sh \
  --iterations 200 --warmup 20 --skew-iterations 40
```

8 卡脚本同时检查：

- fast output 与 NCCL bitwise 一致；
- mode 0 + mode 1 round-trip 恢复输入；
- rank 0 故意延迟消费时，同 tag 多代复用仍正确；
- LingBot 真实 packed QKV 形状 `[1,585,40,384]` 的往返耗时。

最终 tag pool=32 记录：

| 指标 | NCCL | genuine fast-ulysses |
|---|---:|---:|
| critical-rank round-trip mean（200 次） | 0.2510 ms | 0.2157 ms |
| critical-rank mean speedup | 1.00× | 1.16× |
| critical-rank p50 | 0.2391 ms | 0.2123 ms |

另一次 `skew-iterations=40` 的运行跨过 32-tag wrap，rank 0 在消费端故意延迟，
仍通过 bitwise 与多代复用校验。`auto` 的真实 packed-QKV round-trip 比显式
`sm` 和 `ce` 更快，因此生产配置保留 `auto`。microbenchmark 仍不能替代整模型
steady-state A/B。

## 生产启动参数

删除旧的三项 `sgl_p2p` 开关：

```bash
unset SGLANG_ENABLE_ULYSSES_P2P_A2A
unset LINGBOT_FORCE_P2P
unset LINGBOT_ULYSSES_JIT
```

保留原有 RIFE、SR、warmup、offload 和 TLS 参数，增加：

```bash
export LD_PRELOAD="/usr/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="/home/admin/nvshmem-3.7.2-cu12/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NVSHMEM_DISABLE_NVLS=1
export NVSHMEM_REMOTE_TRANSPORT=none
export NCCL_NVLS_ENABLE=0
export SGLANG_FAST_ULYSSES_TAG_POOL_SIZE=32
```

并给 `sglang serve` 增加：

```text
--ulysses-a2a-backend fast_ulysses
--ulysses-a2a-transfer auto
```

`auto` 只在上游 SM/TMA kernel 间实测选优，不会选择 CE。若要测 copy engine，
显式使用 `--ulysses-a2a-transfer ce`，但必须重新做整模型 A/B。

仓库内已提供与原参数一致的生产启动入口；它只启动服务，不运行测试客户端：

```bash
bash /home/admin/sglang/scripts/fast_ulysses/start_lingbot_production.sh
```

完整的、与原生产参数一致的可复现实验入口：

```bash
bash scripts/fast_ulysses/run_lingbot_variant.sh fast_ulysses 90
```

NCCL 对照：

```bash
bash scripts/fast_ulysses/run_lingbot_variant.sh nccl 90
```

两个脚本都会拒绝占用已有的 8001 服务，只清理自己启动的 process group，并把
server/client 日志写到 `/home/admin/fast_ulysses_experiment/logs`。

## LingBot A/B 与历史日志

服务端使用同一台 8×L20X 主机、同一模型和启动参数；客户端为
`/ossfs/workspace/test_fusion_ling.py`，每个 variant 连续跑 3 轮 90 秒。
下表只统计去掉每个 session 的 chunk 0 后、`mode=moving` 的 steady-state
样本。为了降低运行顺序和温度漂移的影响，最终 Fast 被同一适配分支的前后两次
NCCL 运行夹住：

| variant | moving 样本数 | denoise mean | scheduler mean | deploy total mean | deploy total p50 | deploy total p95 |
|---|---:|---:|---:|---:|---:|---:|
| NCCL（Fast 前） | 85 | 349.266 ms | 563.473 ms | 692.926 ms | 684.40 ms | 757.14 ms |
| genuine Fast | 88 | **333.929 ms** | **538.977 ms** | **667.789 ms** | **652.07 ms** | 743.65 ms |
| NCCL（Fast 后） | 86 | 347.024 ms | 555.787 ms | 684.610 ms | 673.72 ms | **743.32 ms** |
| 前后 NCCL 均值 | — | 348.145 ms | 559.630 ms | 688.768 ms | — | — |

Fast 相对前后 NCCL 均值：

- denoise mean 降低 **4.08%**；
- scheduler mean 降低 **3.69%**；
- deploy total mean 降低 **3.05%**。

按单侧对照计算，deploy total mean 的收益范围是 **2.46%～3.63%**，因此 3.05%
比只挑一次 NCCL 更适合作为这台机器上的结论。p95 为 743.65 ms 对 743.32 ms，
没有观察到尾延迟收益；收益主要在 mean 和 p50。

同机还 checkout 了原生 `event_test3` commit
`3fc065f99cd6e12e81861c9847d46025f959a895`，它的 moving denoise /
scheduler / total 分别是 341.451 / 548.466 / 676.873 ms。最终 Fast 分别快
2.20% / 1.73% / 1.34%。

用户提供的历史 `bf16_server.log` 来自旧环境，按相同 moving-only 口径为：

| variant | denoise mean | scheduler mean | deploy total mean |
|---|---:|---:|---:|
| historical `event_test3` / NCCL | 324.400 ms | 535.219 ms | 662.660 ms |
| 当前环境 genuine Fast | 333.929 ms | 538.977 ms | 667.789 ms |

当前环境 Fast 的 total 比历史基线慢 0.77%，而同机原生 `event_test3` 比历史基线
慢 2.15%；这说明跨环境的绝对值不能直接当成 backend 回归。最终结论以同机、
同轮次的 NCCL–Fast–NCCL 夹逼为准。

历史 `fast_ulysess.log` 实际选择的是 `backend=sgl_p2p`，没有 genuine marker，
所以它不是 triple-mu fast-ulysses 的结果。真正生效时每个 rank 都会记录上游
模块路径、world size、transfer、tag pool 和 safety policy。

关键原始日志：

```text
/home/admin/fast_ulysses_experiment/logs/lingbot-nccl-20260724-121015.server.log
/home/admin/fast_ulysses_experiment/logs/lingbot-fast_ulysses-20260724-121919.server.log
/home/admin/fast_ulysses_experiment/logs/lingbot-nccl-20260724-122603.server.log
/home/admin/fast_ulysses_experiment/logs/microbenchmark-auto-20260724-121723.log
/home/admin/fast_ulysses_experiment/logs/microbenchmark-auto-20260724-121841.log
/home/admin/fast_ulysses_experiment/logs/cpu-tests-final.log
```

统一解析命令：

```bash
python scripts/fast_ulysses/parse_lingbot_log.py \
  /path/to/bf16_server.log \
  /path/to/new-nccl.server.log \
  /path/to/new-fast-ulysses.server.log
```

## 回退

生产回退不需要卸载任何包或改模型：

```text
--ulysses-a2a-backend nccl
```

显式 backend 是 strict 的。`fast_ulysses` 不可用或语义不支持时会启动/请求失败，
不会悄悄回退为 NCCL 后仍声称 fast 生效。
