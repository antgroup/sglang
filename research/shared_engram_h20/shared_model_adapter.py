"""Opt-in research adapter; preserve native private-TP and CP collectives.

Install before importing deepseek_v4. Owner connection/registration is completed
at finish_load, outside CUDA graph capture. Default serving remains unmodified.
"""
import atexit
import base64
import json
import hashlib
import os
import signal
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from engram_transport import Context
from owner_control import OwnerClient
from row264_kernels import gather_row264


@contextmanager
def _reader_device_scope(mapping, gpu):
    """Select this visible GPU's NIC only while creating its native contexts."""
    key = 'GPUNETIO_DEVICE_API_RDMA'
    previous = os.environ.get(key)
    if mapping is None:
        yield previous
        return
    devices = [device.strip() for device in mapping.split(',')]
    if any(not device for device in devices) or not 0 <= gpu < len(devices):
        raise ValueError('reader device map must name every visible GPU NIC')
    os.environ[key] = devices[gpu]
    try:
        yield devices[gpu]
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@triton.jit
def _read_wait_stamp(pointer):
    tick=tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;",constraints="=l",args=[],dtype=tl.uint64,is_pure=False,pack=1)
    tl.store(pointer,tick)


@triton.jit
def _scatter_fetched(w, s, ids, destinations, lo, hi, N: tl.constexpr,
                     ROWS: tl.constexpr, COLS: tl.constexpr,
                     WEIGHT_STRIDE: tl.constexpr = 256, SCALE_STRIDE: tl.constexpr = 8, reps=None, MAPPED:tl.constexpr=False):
    q = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids + q)
    if (idx >= lo) & (idx < hi):
        x = tl.arange(0, 256)
        source_q=tl.load(reps+q) if MAPPED else q
        values = tl.load(w + source_q * WEIGHT_STRIDE + x).to(tl.float32)
        exp = tl.load(s + source_q * SCALE_STRIDE + x // 32).to(tl.int32)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        scale = tl.where(exp == 0, 2.0**-127, scale)
        token = q // COLS
        dst = tl.load(destinations + token // ROWS).to(tl.pointer_type(tl.bfloat16))
        offset = ((token % ROWS) * COLS + q % COLS) * 256
        tl.store(dst + offset + x, (values * scale + 0.0).to(tl.bfloat16))


_runtime = None
_quarantined = []


class ReaderRuntime:
    def __init__(self):
        config = json.loads(Path(os.environ["ENGRAM_SHARED_READY"]).read_text())
        self.client = OwnerClient(config["host"], config["port"],
                                  config["checkpoint_id"], config["generation"])
        self.lease = self.client.attach()
        self.context = None
        self.closed = False
        try:
            self._prepare()
            if os.environ.get("ENGRAM_SHARED_CONTROLLED_DRAIN") == "1":
                if signal.getsignal(signal.SIGUSR2) != signal.SIG_DFL:
                    raise RuntimeError("refuse to overwrite an existing SIGUSR2 handler")
                signal.signal(signal.SIGUSR2, self._controlled_drain)
        except BaseException:
            try:
                if self.context is not None:
                    self.context.close()
                self.lease.release()
            except BaseException:
                # A failed device retirement must retain scratch and its lease.
                _quarantined.append(self)
            raise
        atexit.register(self.close)
        print(json.dumps({"event":"shared-engram-reader-ready", "pid":os.getpid(),
                          "gpu":torch.cuda.current_device(), "scratch_bytes":self.arena.numel(),
                          "reader_rdma_device":self.reader_device,
                          "registered_table_bytes":sum(self.metadata["lengths"]),
                          "reader_full_table_replica_bytes":0}), flush=True)

    def _prepare(self):
        md = self.lease.metadata
        self.metadata = md
        self.storage_layout = md.get("storage_layout", "split")
        if self.storage_layout not in ("split", "row264"):
            raise ValueError("unknown shared Engram storage layout")
        # Validate actual serving checkpoint, not merely a same-shape table.
        model = Path(os.environ["ENGRAM_SHARED_MODEL"])
        for name, field in (("config.json", "config_sha256"),
                            ("model.safetensors.index.json", "index_sha256")):
            if hashlib.sha256((model/name).read_bytes()).hexdigest() != md[field]:
                raise ValueError("shared owner/serving checkpoint identity mismatch: " + name)
        if self.storage_layout == "row264":
            cfg = json.loads((model/"config.json").read_text())
            cfg = cfg.get("text_config",cfg)
            rows = cfg["engram_num_embeddings"]
            specs = md.get("table_specs",())
            if len(specs)!=2 or tuple(md["lengths"])!=tuple(int(n)*264 for n in rows):
                raise ValueError("row264 MR count/length mismatch")
            expected = {"weight":{"offset":0,"bytes_per_row":256,"dtype":"F8_E4M3"},
                        "scale":{"offset":256,"bytes_per_row":8,"dtype":"F8_E8M0"}}
            if any(s.get("layer")!=layer or s.get("component_layout")!=expected
                   for s,layer in zip(specs,(1,14))):
                raise ValueError("row264 component layout mismatch")
        self.max_queries = int(os.environ.get("ENGRAM_SHARED_MAX_QUERIES", "196608"))
        self.prefetch_enabled = os.environ.get("ENGRAM_SHARED_PREFETCH")=="1"
        if self.prefetch_enabled and self.storage_layout!='row264':
            raise ValueError('early prefetch requires row264')
        self.p_off = 64
        self.s_off = self.p_off + self.max_queries * 256 + 64
        self.dump = (self.s_off + self.max_queries * 8 + 127) // 64 * 64
        gpu = torch.cuda.current_device()
        self.span = self.dump+64
        self.arena = torch.empty(
            self.span * (2 if self.prefetch_enabled else 1),
            dtype=torch.uint8,
            device="cuda",
        )
        self.layer_stats = torch.zeros((2 if self.prefetch_enabled else 1,8), dtype=torch.uint64, device="cuda")
        self.stats = self.layer_stats[0]
        self.timing_dir=os.environ.get('ENGRAM_RESEARCH_WAIT_TIMING_DIR')
        self.timing_seq=0
        self.read_wait_times=torch.zeros((2,2),device='cuda',dtype=torch.uint64) if self.timing_dir else None
        self.sequence = torch.arange(self.max_queries, dtype=torch.int64, device="cuda")
        self.ranges = {}
        self.prefetch_ids = None
        self.prefetch_issued=set()
        self.defer_layer14=os.environ.get("ENGRAM_SHARED_DEFER_LAYER14")=="1"
        self.defer_control_path=os.environ.get("ENGRAM_RESEARCH_DEFER_CONTROL")
        self.forward_cp_expected = False
        if self.prefetch_enabled:
            self.comm_stream = torch.cuda.Stream(device=gpu)
            self.ready_events = {layer:torch.cuda.Event() for layer in (1,14)}
            self.consumed_events = {layer:torch.cuda.Event() for layer in (1,14)}
        # A separate base per model instance prevents same-node endpoint reuse.
        port = int(os.environ["ENGRAM_SHARED_READER_PORT_BASE"]) + gpu*max(4,int(os.environ.get("ENGRAM_SHARED_READ_LANES","1")))
        lanes=int(os.environ.get("ENGRAM_SHARED_READ_LANES","1"))
        planned=os.environ.get('ENGRAM_SHARED_READ_PLAN')=='1'
        with _reader_device_scope(os.environ.get('ENGRAM_SHARED_READER_DEVICES'), gpu) as self.reader_device:
            if planned:
                if not self.prefetch_enabled:raise ValueError('planned reader is P-only')
                from read_plan import ReadPlan
                graph_sizes=tuple(int(n) for n in os.environ.get('ENGRAM_SHARED_READ_GRAPH_SIZES','49152').split(','))
                if not graph_sizes or any(n<=0 or n>self.max_queries for n in graph_sizes):raise ValueError('invalid planned READ graph capacity')
                self.context=ReadPlan(os.environ['ENGRAM_SHARED_LIBRARY'],gpu,port,lanes,self.max_queries,graph_sizes)
            elif lanes!=1:
                if not self.prefetch_enabled:raise ValueError('multiple READ lanes require P prefetch')
                from read_lanes import ReadLanes
                self.context=ReadLanes(os.environ["ENGRAM_SHARED_LIBRARY"],gpu,port,lanes)
            else:
                self.context = Context(os.environ["ENGRAM_SHARED_LIBRARY"], gpu, port)
        if self.storage_layout == "row264" and not self.context.supports_row264:
            raise RuntimeError("selected runtime lacks row264 capability")
        self.context.register([self.arena], host=False)
        self.context.connect_raw(base64.b64decode(md["native_md"], validate=True),
                                 md["addresses"], md["lengths"], md["owner_gpu"])
        # ReadPlan infers one normal span per layer from the two async spans.
        if hasattr(self.context, "graph_sizes"):
            self.context.span = self.span

    def fetch(self, indices, layer, lo, hi):
        if self.closed:
            raise RuntimeError("shared Engram reader drained; new admission must stay stopped")
        ids = indices.reshape(-1).contiguous()
        n = ids.numel()
        if n > self.max_queries:
            raise ValueError(f"shared Engram scratch capacity {n}>{self.max_queries}")
        li = (1, 14).index(layer)
        if self.prefetch_enabled:
            if torch.cuda.is_current_stream_capturing():
                # ReadPlan QP16 shares context 0 with any direct fallback post.
                # Expected prefill capture uses SGLang's Engram capture stub;
                # reaching this path is unsupported and must fail closed.
                from sglang.srt.layers.remote_engram import (
                    _reject_prefetch_graph_capture,
                )

                _reject_prefetch_graph_capture(True, True)
            if self.prefetch_ids is None:
                if self.forward_cp_expected:
                    raise RuntimeError('CP forward did not execute early hash prefetch hook')
                # Tiny/non-CP forwards (including health requests) never run
                # the full-CP hash hook. Keep the same producer stream even
                # when a synchronous read is required.
                self.comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.comm_stream):
                    ids.record_stream(self.comm_stream)
                    self.context.fetch(ids,n,li*self.span+self.p_off,0,li*self.span+self.dump,
                        512,self.layer_stats[li],weight_index=li,scale_index=li,
                        row_lo=lo,row_hi=hi,row264=True)
                    self.ready_events[layer].record()
            elif self.prefetch_ids[layer].numel()!=n or self.ranges[layer]!=(lo,hi):
                raise RuntimeError('prefetch consumer shape/range was not prepared')
            if self.prefetch_ids is not None and layer not in self.prefetch_issued:
                self.issue_prefetch_layer(layer) # Safe synchronous fallback for non-scatter paths.
            if self.read_wait_times is not None:_read_wait_stamp[(1,)](self.read_wait_times[li,0],num_warps=1)
            torch.cuda.current_stream().wait_event(self.ready_events[layer])
            if self.read_wait_times is not None:_read_wait_stamp[(1,)](self.read_wait_times[li,1],num_warps=1)
            if self.prefetch_ids is not None:
                torch._assert_async((ids==self.prefetch_ids[layer]).all(),'prefetched Engram IDs differ')
            torch._assert_async(self.layer_stats[li].view(torch.int64)[0]==0,'prefetched Engram GET failed')
            offset=li*self.span+self.p_off
            raw=self.arena[offset:offset+n*264].view(n,264)
            return ids,raw[:,:256].view(torch.float8_e4m3fn),raw[:,256:]
        packed = self.storage_layout == "row264"
        self.context.fetch(ids, n, self.p_off, 0 if packed else self.s_off, self.dump, 512,
                           self.stats, weight_index=li if packed else 2 * li,
                           scale_index=li if packed else 2 * li + 1,
                           row_lo=lo, row_hi=hi, row264=packed)
        torch._assert_async(self.stats.view(torch.int64)[0] == 0, "shared Engram GET failed")
        if packed:
            raw = self.arena[self.p_off:self.p_off+n*264].view(n,264)
            return ids,raw[:,:256].view(torch.float8_e4m3fn),raw[:,256:]
        return (ids, self.arena[self.p_off:self.p_off+n*256].view(n,256).view(torch.float8_e4m3fn),
                self.arena[self.s_off:self.s_off+n*8].view(n,8))

    def begin_prefetch(self, full_hash_ids, local_rows, cp_size):
        if not self.prefetch_enabled or self.closed:
            raise RuntimeError('prefetch runtime unavailable')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('full-hash prefetch must start before custom graph capture')
        if set(self.ranges)!={1,14} or full_hash_ids.ndim!=3 or full_hash_ids.shape[1]!=2:
            raise ValueError('both trained layer ranges/full hash tensor are required')
        if cp_size not in (4,8) or full_hash_ids.shape[0]>cp_size*local_rows:
            raise ValueError('unsupported CP/padded hash shape')
        n=cp_size*local_rows*full_hash_ids.shape[2]
        if n==0:return
        self.forward_cp_expected=True
        if n>self.max_queries:raise ValueError('prefetch exceeds registered scratch')
        # Match native cat([hash[r::CP] padded to hidden_rows] for r in CP).
        ordered=full_hash_ids.new_zeros((cp_size*local_rows,*full_hash_ids.shape[1:]))
        for rank in range(cp_size):
            part=full_hash_ids[rank::cp_size]
            ordered[rank*local_rows:rank*local_rows+part.shape[0]].copy_(part)
        ids={layer:ordered[:,li,:].contiguous().reshape(-1) for li,layer in enumerate((1,14))}
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        self.prefetch_ids=ids
        with torch.cuda.stream(self.comm_stream):
            for layer in ((1,) if self.defer_layer14 else (1,14)):
                self._enqueue_prefetch_layer(layer)

    def _enqueue_prefetch_layer(self,layer):
        li=(1,14).index(layer);ids=self.prefetch_ids[layer];lo,hi=self.ranges[layer]
        ids.record_stream(self.comm_stream)
        self.context.fetch(ids,ids.numel(),li*self.span+self.p_off,0,li*self.span+self.dump,
            512,self.layer_stats[li],weight_index=li,scale_index=li,row_lo=lo,row_hi=hi,row264=True)
        self.ready_events[layer].record();self.prefetch_issued.add(layer)

    def issue_prefetch_layer(self,layer):
        if self.closed:raise RuntimeError('cannot prefetch after drain')
        if self.prefetch_ids is None or layer in self.prefetch_issued:return
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):self._enqueue_prefetch_layer(layer)

    def start_forward(self, cp_extend):
        if cp_extend and self.defer_control_path:
            # Opt-in experiment control, changed atomically only while idle.
            # No file access in the default model path.
            value=Path(self.defer_control_path).read_text().strip()
            if value not in ('0','1'):raise ValueError('invalid deferred READ experiment policy')
            self.defer_layer14=value=='1'
        if cp_extend:self.timing_seq=getattr(self,'timing_seq',0)+1
        self.prefetch_ids=None
        self.prefetch_issued=set()
        self.forward_cp_expected=bool(cp_extend)

    def report_wait_timing(self):
        if not self.timing_dir or self.prefetch_ids is None or self.prefetch_ids[1].numel()!=393216:return
        # Readback is after the second consumer was queued. It cannot alter this
        # forward's first-Engram wait; do not use these runs for serving timing.
        timing=self.read_wait_times.cpu().tolist();stats=self.layer_stats.cpu().tolist()
        record=dict(gpu=torch.cuda.current_device(),sequence=self.timing_seq,policy=int(self.defer_layer14),queries=393216,wait_ns=[b-a for a,b in timing],layer_stats=stats)
        path=Path(self.timing_dir)/('frame'+str(self.timing_seq)+'-gpu'+str(record['gpu'])+'.json')
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record));temp.replace(path)

    def record_consumed(self, layer):
        if self.prefetch_enabled:
            if torch.cuda.is_current_stream_capturing():
                from sglang.srt.layers.remote_engram import (
                    _reject_prefetch_graph_capture,
                )

                _reject_prefetch_graph_capture(True, True)
            self.consumed_events[layer].record(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(self.consumed_events[layer])
                self.context.record_consumed()
        else:
            self.context.record_consumed()

    def close(self):
        if not self.closed:
            # Native close retires CUDA/NIC consumers before the owner lease.
            self.context.close()
            self.lease.release()
            self.closed = True

    def _controlled_drain(self, *_):
        # Research controller stops admission and waits for all HTTP requests
        # before signaling every rank. Keep workers alive until all release:
        # an early rank exit could make the parent kill undrained peer ranks.
        self.close()
        print(json.dumps({"event":"shared-engram-reader-drained", "pid":os.getpid(),
                          "lease_released":True}), flush=True)


def install():
    import sglang.srt.layers.engram as native
    from sglang.srt.layers.remote_engram import RemoteEngramTable

    class SharedEngramEmbedding(native.EngramEmbedding):
        def _init_host_table(self, num_embeddings, dim, layer_id):
            if os.environ.get("SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT") != "private":
                raise ValueError("shared RDMA adapter currently preserves private TP layout only")
            if dim != 256 or layer_id not in (1, 14):
                raise ValueError("unsupported shared Engram checkpoint layout")
            self.remote_table = RemoteEngramTable(num_embeddings, dim, layer_id)
            self.host_table = SimpleNamespace(layout="private")
            # Parameter names/loaders are retained, with no consumer table copy.
            self.weight = torch.nn.Parameter(torch.empty((0, dim), dtype=torch.float8_e4m3fn),
                                             requires_grad=False)
            self.scale = torch.nn.Parameter(torch.empty((0, dim//32), dtype=torch.float8_e8m0fnu),
                                            requires_grad=False)

        def _load_rows(self, param, loaded_weight):
            self.remote_table.check_weight(param, loaded_weight, self.weight)

        def finish_load(self, label=""):
            self.remote_table.finish_load(self.row_start, self.rows)
            print(
                json.dumps(
                    {
                        "event": "shared-engram-layer-loaded",
                        "pid": os.getpid(),
                        "layer": self.remote_table.layer_id,
                        "reader_parameter_elements": self.weight.numel()
                        + self.scale.numel(),
                    }
                ),
                flush=True,
            )

        def _owned_rows(self, indices, *, in_id_order=False):
            return self.remote_table.lookup(
                indices, self.row_start, self.rows, self.new_rows
            )

        def _lookup(self, indices):
            # Native CP fallback reaches here after its TP all-reduce.
            out = super()._lookup(indices)
            self.remote_table.after_reduction()
            return out

        def _cp_scatter_owned_rows(self, all_indices, *, rows):
            parallel = native.get_parallel()
            if (
                not self._cp_owner_scatter
                or self.rows == 0
                or all_indices.dim() != 2
                or not native.cp_group_is_tp_group()
                or torch.cuda.is_current_stream_capturing()
                or all_indices.shape[0] != parallel.attn_cp_size * rows
            ):
                return None
            cols = all_indices.shape[1]
            comm = native.get_cp_owner_scatter(
                parallel.attn_cp_group, user="engram", width=cols * self.dim
            )
            if comm is None or rows > comm.max_rows:
                return None
            ids, weight, scale = self.remote_table.fetch_rows(
                all_indices, self.row_start, self.rows
            )
            reps = (
                self.remote_table.runtime.context.representatives(
                    (1, 14).index(self.remote_table.layer_id)
                )
                if hasattr(self.remote_table.runtime.context, "representatives")
                else None
            )
            _scatter_fetched[(ids.numel(),)](
                weight,
                scale,
                ids,
                comm.dst_table,
                self.row_start,
                self.row_start + self.rows,
                ids.numel(),
                rows,
                cols,
                WEIGHT_STRIDE=weight.stride(0),
                SCALE_STRIDE=scale.stride(0),
                reps=reps if reps is not None else ids,
                MAPPED=reps is not None,
                num_warps=4,
            )
            self.remote_table.record_consumed()
            result = comm.finish(rows).view(rows, cols, self.dim)
            self.remote_table.after_reduction()
            return result

    native.EngramEmbedding = SharedEngramEmbedding


def prefetch_full_hash(full_hash_ids, local_rows, cp_size):
    if _runtime is None:
        raise RuntimeError('Engram owner was not initialized before prefetch')
    _runtime.begin_prefetch(full_hash_ids,local_rows,cp_size)


def prepare_forward(cp_extend):
    if _runtime is None or not _runtime.prefetch_enabled:
        raise RuntimeError('prefetch runtime is not initialized')
    # This hook is outside the custom graph and executes on each forward.
    # Clearing a Python reference is safe: query tensors record comm_stream.
    _runtime.start_forward(cp_extend)
