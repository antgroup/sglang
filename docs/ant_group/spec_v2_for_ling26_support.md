 ▐▛███▜▌   Claude Code v2.1.84
▝▜█████▛▘  glink/claude-opus-4-6 · API Usage Billing
  ▘▘ ▝▝    ~/Documents/theta_proj/ant_sglang

❯ SGLANG_ENABLE_SPEC_V2=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m 
 sglang.launch_server --model-path                                              
/home/admin/ling_max_2_6_sft_with_mtp_all_fp8/ --host 0.0.0.0 --port 8188       
--trust-remote-code --tp-size 8 --enable-cache-report --log-level info          
--max-running-requests 16 --mem-fraction-static 0.925 --enable-metrics          
--chunked-prefill-size 8192 --load-balance-method round_robin --dist-timeout    
1800 --allow-auto-output-truncate --tool-call-parser qwen25                     
--speculative-algorithm EAGLE --speculative-num-steps 3                         
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4                     
--mamba-scheduler-strategy extra_buffer --cuda-graph-max-bs 96 --context-length 
 262144 --json-model-override-args                                              
'{"linear_backend":"seg_la","torch_dtype":"bfloat16","architectures":["BailingM 
oeV2_5ForCausalLM"],"model_type":"bailing_hybrid"}' --mamba-full-memory-ratio   
1.4 我启动了BailingMoeV2_5ForCausalLM架构的模型，打开了                         
SGLANG_ENABLE_SPEC_V2=1，启动报错：2026-04-03 14:57:25.716 ERROR 496007 [ TP4   
scheduler.py:3475] Scheduler hit an exception: Traceback (most recent call      
last):                                                                          
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File            
"/opt/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",     
line 3471, in run_scheduler_process                                             
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]                   
scheduler.run_event_loop()                                                      
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File            
"/opt/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",     
line 1248, in run_event_loop                                                    
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]                   
dispatch_event_loop(self)                                                       
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File            
"/opt/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",     
line 3347, in dispatch_event_loop                                               
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]                   
scheduler.event_loop_overlap()                                                  
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File            
"/opt/conda/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, 
 in decorate_context                                                            
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]     return        
func(*args, **kwargs)                                                           
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File            
"/opt/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",     
line 1309, in event_loop_overlap                                                
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]     batch_result  
= self.run_batch(batch)                                                         
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File            
"/opt/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",     
line 2568, in run_batch                                                         
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]     batch_result  
= self.model_worker.forward_batch_generation(                                   
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File "/opt/cond 
a/lib/python3.10/site-packages/sglang/srt/speculative/eagle_worker_v2.py", line 
 736, in forward_batch_generation                                               
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]                   
self.draft_worker._draft_extend_for_decode(                                     
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File "/opt/cond 
a/lib/python3.10/site-packages/sglang/srt/speculative/eagle_worker_v2.py", line 
 565, in _draft_extend_for_decode                                               
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]     forward_batch 
 = draft_input.prepare_for_extend_to_fill_draft_kvcache(                        
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File "/opt/cond 
a/lib/python3.10/site-packages/sglang/srt/speculative/eagle_info_v2.py", line   
220, in prepare_for_extend_to_fill_draft_kvcache                                
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]                   
draft_model_runner.attn_backend.init_forward_metadata(forward_batch)            
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File            
"/opt/conda/lib/python3.10/site-packages/sglang/srt/layers/attention/hybrid_lin 
ear_attn_backend.py", line 737, in init_forward_metadata                        
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]                   
attn_backend.init_forward_metadata(forward_batch)                               
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File            
"/opt/conda/lib/python3.10/site-packages/sglang/srt/layers/attention/linear/lig 
htning_backend.py", line 76, in init_forward_metadata                           
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]                   
self.forward_metadata = BailingLinearMetadata.prepare_mixed(                    
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]   File            
"/opt/conda/lib/python3.10/site-packages/sglang/srt/layers/attention/linear/lin 
ear_metadata.py", line 59, in prepare_mixed                                     
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]                   
query_start_loc = query_start_loc[: num_prefills + 1]                           
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475] TypeError:        
'NoneType' object is not subscriptable                                          
2026-04-03 14:57:25.716 ERROR 496007 [ TP4 scheduler.py:3475]                   
                                                                                

⏺ Explore(Explore linear_metadata and eagle_info_v2)
  ⎿  Done (13 tool uses · 76.4k tokens · 2m 10s)
  (ctrl+o to expand)

  Read 2 files (ctrl+o to expand)

⏺ Now let me see the init_forward_metadata in HybridLinearAttnBackend to
  understand the full flow.

  Read 1 file (ctrl+o to expand)

⏺ The issue is clear. In HybridLinearAttnBackend.init_forward_metadata (line
  735-737), it calls init_forward_metadata on both sub-backends unconditionally.
   For DRAFT_EXTEND_V2 mode, the linear attention backend's _forward_metadata
  intentionally sets query_start_loc = None (line 178), but then passes it to
  BailingLinearMetadata.prepare_mixed() which tries to subscript it at line 59.

  The comment at line 174-177 already says mamba metadata should be skipped for
  DRAFT_EXTEND_V2. The fix should skip calling the linear attention backend for
  this mode.

  Read 1 file (ctrl+o to expand)

⏺ The fix should be in HybridLinearAttnBackend.init_forward_metadata — skip the
  linear backend when the forward mode is DRAFT_EXTEND_V2, since only
  full-attention layers run during draft extend.

⏺ Update(python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py)
  ⎿  Added 5 lines
      733          self.attn_backend_list = [full_attn_backend, linear_at
           tn_backend]
      734  
      735      def init_forward_metadata(self, forward_batch: ForwardBatc
           h):
      736 +        if forward_batch.forward_mode.is_draft_extend_v2():   
      737 +            # DRAFT_EXTEND_V2 only runs full-attn layers in th
          +e draft model,                                                
      738 +            # so skip linear/mamba backend metadata which requ
          +ires query_start_loc.                                         
      739 +            self.full_attn_backend.init_forward_metadata(forwa
          +rd_batch)                                                     
      740 +            return                                            
      741          for attn_backend in self.attn_backend_list:
      742              attn_backend.init_forward_metadata(forward_batch)
      743  

⏺ Now let me also clean up the now-dead code path in _forward_metadata that sets
   query_start_loc = None for DRAFT_EXTEND_V2, since we're skipping the linear
  backend entirely.

  Read 1 file (ctrl+o to expand)

⏺ The _forward_metadata method is shared between the mamba backend and the
  lightning backend (via inheritance from MambaAttnBackendBase). Now that we
  skip calling the linear backend for DRAFT_EXTEND_V2 at the
  HybridLinearAttnBackend level, the query_start_loc = None path in
  _forward_metadata will no longer be reached from the linear backend. However,
  it might still be needed by the mamba backend in other hybrid configurations —
   so I'll leave it as-is to avoid breaking other paths.

  The fix is clean: when forward_mode is DRAFT_EXTEND_V2, we only initialize the
   full-attention backend since the draft model only runs full-attention layers.
   This prevents the linear/mamba backend from trying to process None metadata.

  Root cause: HybridLinearAttnBackend.init_forward_metadata() was calling both
  sub-backends unconditionally. For DRAFT_EXTEND_V2, the linear backend's
  _forward_metadata intentionally set query_start_loc = None (since mamba layers
   don't run during draft extend), but then
  BailingLinearMetadata.prepare_mixed() tried to subscript it at line 59,
  causing the TypeError: 'NoneType' object is not subscriptable.

  Fix: Skip the linear attention backend in init_forward_metadata when the
  forward mode is DRAFT_EXTEND_V2, since only full-attention layers execute
  during EAGLE v2 draft extension.

✻ Baked for 3m 19s

❯ 好了？                                                                        

⏺ 是的，修复完成。改动在
  python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py 第 735-741
  行，只加了 4 行代码：

  if forward_batch.forward_mode.is_draft_extend_v2():
      self.full_attn_backend.init_forward_metadata(forward_batch)
      return

  DRAFT_EXTEND_V2 模式下 draft model 只跑 full-attention 层，不需要 linear/mamba
   的 metadata。之前无条件调用两个 backend，导致 linear backend 拿到
  query_start_loc = None 后在 BailingLinearMetadata.prepare_mixed()
  里崩溃。现在直接跳过 linear backend 即可。

❯ 从原理上再给我讲讲这个bug的背景吧                                             

⏺ 三层背景

  1. 混合注意力架构（Hybrid Attention）

  BailingMoeV2_5 是一个混合架构模型——它的 decoder 层不是清一色的 full
  attention，而是部分层用标准的 full attention（如 FlashAttention），部分层用
  linear attention（Lightning Attention）。这两种 attention 的 KV cache
  管理方式完全不同：

  - Full attention：需要存储每个 token 的 K、V 向量，推理时做 softmax attention
  - Linear attention：维护的是一个固定大小的 recurrent state（类似 Mamba/SSM 的
  hidden state），不存显式的 KV cache

  所以 SGLang 为这类模型设计了 HybridLinearAttnBackend，内部持有两个子
  backend，在每次 forward 前需要分别为两个 backend 准备各自的 metadata（比如
  query_start_loc、mamba_cache_indices 等）。

  2. EAGLE 投机解码 V2（Speculative Decoding）

  EAGLE 的核心思路是：用一个轻量的 draft model 快速预测后续若干 token，然后用
  target model 一次性验证这些 token，命中的就直接采纳，从而用一次 target forward
   换来多个 token 的输出。

  V2 的流程中有一个关键步骤——DRAFT_EXTEND_V2：draft model 需要把刚被 target
  model 验证通过的 token "回填" 到自己的 KV cache
  中，以便下一轮继续预测。这本质上是一个 extend（prefill）操作，而不是普通的
  decode。

  关键点：EAGLE 的 draft model 通常只包含 full attention 层。它是对 target model
   的简化版本，不包含 linear attention / mamba 层。所以在 DRAFT_EXTEND_V2
  阶段，linear attention backend 根本不会被执行。

  3. Bug 的本质：抽象层的信息断裂

  HybridLinearAttnBackend.init_forward_metadata()
      ├── full_attn_backend.init_forward_metadata()    ✅ 正常
      └── linear_attn_backend.init_forward_metadata()  💥 崩溃

  问题出在抽象层级之间的信息不一致：

  - HybridLinearAttnBackend 作为统一入口，不区分 forward mode，无条件调用两个子
  backend
  - 子 backend 的 _forward_metadata() 里已经意识到 DRAFT_EXTEND_V2 不需要 linear
   metadata，所以把 query_start_loc 设为 None（注释也写了 "can be skipped"）
  - 但下游的 BailingLinearMetadata.prepare_mixed() 不知道这个约定，拿到 None
  直接做切片就崩了

  本质上，"DRAFT_EXTEND_V2 不需要 linear
  attention"这个知识散落在了中间层（_forward_metadata 用 None
  做信号），但没有在入口处（HybridLinearAttnBackend）拦截，也没有在出口处（prepa
  re_mixed）防御。修复放在入口层最干净——从源头就不调用不需要的 backend。

✻ Crunched for 32s

❯ /model                                                                        
  ⎿  Set model to glink/claude-opus-4-6 (default)

❯ 但之前SGLANG_ENABLE_SPEC_V2=0的时候，走了老的spec，为啥不会崩溃               

⏺ 好问题，让我看一下 v1 的路径。

⏺ Explore(Compare EAGLE v1 vs v2 draft extend)
  ⎿  Done (35 tool uses · 83.8k tokens · 7m 32s)
  (ctrl+o to expand)

⏺ 核心区别在于 v1 和 v2 用了不同的 ForwardMode，而这两个 mode 在 linear
  attention backend 里的处理路径完全不同。

  EAGLE v1：ForwardMode.DRAFT_EXTEND

  v1 做 draft extend 时设的是 ForwardMode.DRAFT_EXTEND。这个 mode 在
  _forward_metadata 里没有被特殊对待——它走的是普通 extend 的 else 分支：

  # _forward_metadata 中的分支逻辑
  if forward_batch.forward_mode.is_decode_or_idle():
      query_start_loc = torch.arange(...)          # decode 路径
  elif forward_batch.forward_mode.is_extend(...):
      if forward_batch.forward_mode.is_draft_extend_v2():
          query_start_loc = None                   # ← v2 专属，设为 None
      elif forward_batch.forward_mode.is_target_verify():
          query_start_loc = torch.arange(...)      # verify 路径
      else:
          query_start_loc = torch.empty(...)       # ← v1 走这里，正常构造 
  tensor

  v1 的 DRAFT_EXTEND 被当成普通 extend 处理，query_start_loc 是一个正常的
  tensor，下游自然不会崩。

  EAGLE v2：ForwardMode.DRAFT_EXTEND_V2

  v2 引入了一个新的设计意图：draft model 只有 full attention 层，linear/mamba
  层在 draft extend 阶段根本不跑，所以不需要为 linear backend 准备 metadata。

  代码里体现这个意图的方式是：在 _forward_metadata 中把 query_start_loc 设为
  None
  作为"不需要"的信号。但问题是，HybridLinearAttnBackend.init_forward_metadata()
  仍然无条件地调了两个 backend，None 就流到了
  BailingLinearMetadata.prepare_mixed() 并崩溃。

  一句话总结

  v1 对 linear backend "无害"是因为它复用了普通 extend 路径，metadata 
  正常构造；v2 做了优化想跳过 linear backend，但只在数据层（设 
  None）表达了这个意图，没在控制流层（入口处）拦截调用，导致 None 
  泄漏到了不该到的地方。

✻ Worked for 8m 5s

claude --resume 9b7f4a07-5577-4520-9ea1-c35b9dbef188                                                                                                                            
