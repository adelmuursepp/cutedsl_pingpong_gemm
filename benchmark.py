"""Benchmark base GemmSM90 (cooperative) vs the per-tile ping-pong design.

vanilla : GemmSM90 with atom_layout_mn=(2,1), tile=(128, 256). Both consumer
          warpgroups collaborate on each tile.
pingpong: GemmPingPong with atom_layout_mn=(1,1), tile=(64, 256). Each consumer
          warpgroup processes its own tile, alternating mainloop/epilogue via
          named-barrier gates so one WG is in mma while the other is in epi.

Constraint: pingpong's per-tile design needs k_iters < 2*ab_stage
to avoid 1-bit phase aliasing in the AB pipeline. With ab_stage=3 and cta_K=64,
this means K < 384. Larger K silently produces wrong output.
"""
import time
import torch

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args
from cdsl_fn_utils import compile_cutedsl, STREAM
from gemm_base import GemmSM90
from gemm_pingpong import GemmPingPong


# Cooperative baseline: 2 consumer WGs share one tile, full atom_layout MMA.
gemm_vanilla = GemmSM90(
    tile_shape_mn=(128, 256),
    epi_tile_mn=(128, 32),
    cluster_shape_mnk=(2, 1, 1),
    atom_layout_mn=(2, 1),
    ab_stage=3,
    epi_stage=2,
    reuse_ab=False,
    is_persistent=True,
    gemm_n_prologue=0,
)

# Per-tile ping-pong: each consumer WG owns its own tile. tile_M MUST be 64
# (= atom_layout[0] * 64) or the MMA atom only covers half the tile.
gemm_pingpong = GemmPingPong(
    tile_shape_mn=(64, 256),
    epi_tile_mn=(64, 128),
    cluster_shape_mnk=(2, 1, 1),
    atom_layout_mn=(1, 1),
    ab_stage=3,
    epi_stage=2,
    reuse_ab=False,
    is_persistent=True,
)

CTA_K = 64  # = mma_k * mma_inst_tile_k = 16 * 4 (set in base.populate_mma_atom)
PINGPONG_AB_STAGE = 3
MAX_VALID_K_FOR_PINGPONG = (2 * PINGPONG_AB_STAGE - 1) * CTA_K  # 5 * 64 = 320


def torch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b.t()


def _run_one(m, n, k, to_csv):


    a = get_normal_bernoulli((m, k))
    b = get_normal_bernoulli((n, k))
    c = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    tensors = (a, b)

    compiled_vanilla = compile_cutedsl((a, b, c), gemm_vanilla)
    compiled_pingpong = compile_cutedsl((a, b, c), gemm_pingpong)

    ref = a.to(torch.float64) @ b.to(torch.float64).t()

    def cdsl_vanilla(a_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device="cuda")
        compiled_vanilla(a_, b_, o, STREAM)
        return o

    def cdsl_pingpong(a_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device="cuda")
        compiled_pingpong(a_, b_, o, STREAM)
        return o

    torch_out    = ExperimentOutput("gemm_torch",    m, n, k)
    vanilla_out  = ExperimentOutput("gemm_vanilla",  m, n, k)
    pingpong_out = ExperimentOutput("gemm_pingpong", m, n, k)

    pingpong_out.run(cdsl_pingpong, tensors, ref)
    time.sleep(2)
    vanilla_out.run(cdsl_vanilla, tensors, ref)
    time.sleep(2)
    torch_out.run(torch_gemm, tensors, ref)

    if to_csv:
        print(ExperimentOutput.list_to_csv(torch_out.values()))
        print(ExperimentOutput.list_to_csv(vanilla_out.values()))
        print(ExperimentOutput.list_to_csv(pingpong_out.values()))
    else:
        flops = 2 * m * n * k
        vanilla_tflops = flops / (vanilla_out.ms_median / 1e3) / 1e12
        pingpong_tflops = flops / (pingpong_out.ms_median / 1e3) / 1e12
        k_iters = k // CTA_K
        print(f"\n--- m={m} n={n} k={k}  (k_iters={k_iters}) ---")
        print(f"  vanilla   {vanilla_out.ms_median:.3f} ms  {vanilla_tflops:.0f} TFLOPs   speedup vs torch:   {torch_out.ms_median / vanilla_out.ms_median:.3f}x")
        print(f"  pingpong  {pingpong_out.ms_median:.3f} ms  {pingpong_tflops:.0f} TFLOPs   speedup vs vanilla: {vanilla_out.ms_median / pingpong_out.ms_median:.3f}x")


if __name__ == "__main__":
    args = get_args()
    m, n, k = args.m, args.n, args.k

    if args.to_csv:
        print(ExperimentOutput.list_to_csv(ExperimentOutput.header()))

    base_k = k
    sweep_sizes = [
        (m, n, base_k),
        (m, n, max(base_k // 4, CTA_K)),
        (m // 4, n, base_k),
    ]
    seen = set()
    for size in sweep_sizes:
        if size not in seen and size[2] >= CTA_K:
            seen.add(size)
            _run_one(*size, args.to_csv)
