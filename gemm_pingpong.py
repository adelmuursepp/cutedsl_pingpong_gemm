"""Ping-pong GEMM SM90 kernel, per-tile mainloop/epilogue alternation.
Inheritance: subclasses GemmSM90, reuses produce_mainloop / consume_mainloop
unchanged, overrides make_ab_pipeline + kernel + a per-WG epilogue.
"""
from enum import IntEnum

import torch

import cutlass
from cutlass import Int32, const_expr
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from gemm_base import GemmSM90, THREADS_PER_WG
from tile_scheduler import SimpleTileScheduler
from cdsl_fn_utils import compile_cutedsl, STREAM


class PingPongBarrier(IntEnum):
    """Named-barrier IDs.

    Mma{0,1} / Epi{0,1} are 256-thread gates both consumer warpgroups arrive on
    they enforce the alternation between WG0 and WG1 across mainloop / epilogue

    EpiSync{0,1} are per-warpgroup 128-thread barriers used inside the epilogue
    to sync between the stmatrix-into-smem step and the TMA-store-out step

    Barrier id 0 is reserved by hardware for sync_threads()
    """
    Mma0 = 1
    Mma1 = 2
    Epi0 = 3
    Epi1 = 4
    EpiSync0 = 5
    EpiSync1 = 6


class GemmPingPong(GemmSM90):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.atom_layout_mnk == (1, 1, 1), \
            "Gemm ping-pong needs atom_layout_mn=(1, 1)"
        self.mma_warpgroups = 2
        self.threads_per_cta = (self.mma_warpgroups + 1) * THREADS_PER_WG
        self.ab_load_warp_id = self.mma_warpgroups * 4
        self.consumer_warps_per_wg = 4

    # AB pipeline: per-stage consumer arrives = ONE warpgroup, not both.
    @cute.jit
    def make_ab_pipeline(self, mbar_ptr: cute.Pointer, cta_layout_vmnk: cute.Layout):
        num_producers = 1
        mcast_size = self.mcast_ctas_a + self.mcast_ctas_b - 1
        num_consumers = self.consumer_warps_per_wg * mcast_size
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_producers)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_consumers)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=self.ab_stage,
            tx_count=self.tma_ab_load_bytes,
            producer_group=producer_group,
            consumer_group=consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
        )

    # Ping-pong gates. Each Mma{i}/Epi{i} barrier is a 256-thread gate
    # both warpgroups (128 threads each) arrive on it 
    #  The kickoff (WG0 self-arrives once on itsown gates) 
    # replaces the missing "first arrive from the other side" so WG0 isn't deadlocked on its first sync.
    def pingpong_barrier_arrive(self, target_wg: Int32, stage: str):
        assert stage in ("mma", "epi")
        base_id = int(PingPongBarrier.Mma0) if stage == "mma" else int(PingPongBarrier.Epi0)
        cute.arch.barrier_arrive(
            barrier_id=base_id + target_wg,
            number_of_threads=2 * THREADS_PER_WG,
        )

    def pingpong_barrier_sync(self, my_wg: Int32, stage: str):
        assert stage in ("mma", "epi")
        base_id = int(PingPongBarrier.Mma0) if stage == "mma" else int(PingPongBarrier.Epi0)
        cute.arch.barrier(
            barrier_id=base_id + my_wg,
            number_of_threads=2 * THREADS_PER_WG,
        )

    # Kernel entry. Producer is unchanged (loads ALL tiles for this CTA in
    # scheduler order). Consumer block runs the ping-pong: 
    # each WG owns every other tile and alternates mma/epi via the named-barrier gates.
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        mA: cute.Tensor,
        mB: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params,
        cluster_layout_mnk: cute.Layout,
        epi_smem_layout: cute.ComposedLayout,
        epi_copy: cute.CopyAtom,
        epi_mC: cute.Tensor,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.ab_load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            storage.mainloop_pipeline_barriers.data_ptr(),
            cute.make_layout((1, *cluster_layout_mnk.shape)),
        )
        pipeline_init_arrive()
        pipeline_init_wait()

        sA = self.get_smem_field(storage, "sA", a_smem_layout_staged)
        sB = self.get_smem_field(storage, "sB", b_smem_layout_staged)
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)

        tile_scheduler = SimpleTileScheduler.create(tile_sched_params)

        # Producer warpgroup (warps 8..11) 
        if warp_idx >= self.ab_load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            if warp_idx == self.ab_load_warp_id:
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1)
                b_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0)
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    gA_mk = cute.local_tile(
                        mA,
                        cute.select(self.cta_tile_shape_mnk, [0, 2]),
                        (tile_coord_mnkl[0], None),
                    )
                    k_iters = cute.size(gA_mk, mode=[2])

                    ab_producer_state = self.produce_mainloop(
                        k_iters, ab_pipeline, ab_producer_state,
                        tma_atom_a, tma_atom_b, mA, sA, mB, sB,
                        tile_coord_mnkl, block_in_cluster_coord_mnk,
                        cluster_layout_mnk, a_mcast_mask, b_mcast_mask,
                    )
                    tile_scheduler.fetch_next_work()
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()

                ab_pipeline.producer_tail(ab_producer_state)

        # Consumer warpgroups (warps 0..7) 
        # WG0 owns tiles {bidz, bidz+2N, bidz+4N, ...} and WG1 owns the
        # interleaved {bidz+N, bidz+3N, ...}, where N = num_persistent_clusters
        # They alternate via Mma{0,1}/Epi{0,1} so that exactly one WG holds the
        # mma resource and exactly one holds the epilogue smem at any time
        if warp_idx < self.ab_load_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)

            tidx_full, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx_full // THREADS_PER_WG)
            # Rebase tidx so each WG slices the per-WG-sized tiled_mma identically.
            tidx = tidx_full % THREADS_PER_WG

            thr_mma = tiled_mma.get_slice(tidx)
            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
            acc_shape = tiled_mma.partition_shape_C(
                cute.select(self.cta_tile_shape_mnk, mode=[0, 1])
            )
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            # WG1: skip the first tile (it's WG0's), advance scheduler once and
            # also advance the AB pipeline state by k_iters_first so WG1 starts
            # consuming WG1's first tile, not WG0's.
            #
            # k_iters_skip < 2 * ab_stage (else 1-bit phase aliases and WG1 reads wrong data). 
            # The __main__ shape enforces that constraint.
            work_tile = tile_scheduler.initial_work_tile_info()


            if warp_group_idx == 1:
                gA_mk_first = cute.local_tile(
                    mA,
                    cute.select(self.cta_tile_shape_mnk, [0, 2]),
                    (work_tile.tile_idx[0], None),
                )
                k_iters_skip = cute.size(gA_mk_first, mode=[2])
                for _ in cutlass.range(k_iters_skip, unroll=1):
                    ab_consumer_state.advance()

                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # Kickoff: WG0 self-arrives so its first sync('mma') and sync('epi')
            # release immediately 
            # WG1 starts blocked on Mma1/Epi1 until WG0 finishes its first phase and arrives on the WG1-side gates.
            if warp_group_idx == 0:
                self.pingpong_barrier_arrive(Int32(0), "mma")
                self.pingpong_barrier_arrive(Int32(0), "epi")

            iter_count = Int32(0)
            while work_tile.is_valid_tile:
                tile_coord_mnk = (
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                )
                gA_mk = cute.local_tile(
                    mA,
                    cute.select(self.cta_tile_shape_mnk, [0, 2]),
                    (tile_coord_mnk[0], None),
                )
                k_iters = cute.size(gA_mk, mode=[2])


                # Wait for permission to enter the mainloop.
                self.pingpong_barrier_sync(warp_group_idx, "mma")


                ab_consumer_state, tiled_mma = self.consume_mainloop(
                    k_iters, tiled_mma, accumulators, ab_pipeline,
                    ab_consumer_state, tCrA, tCrB, tidx, sA, sB,
                )


                # Skip the other WG's k_iters worth of stages.
                for _ in cutlass.range(k_iters, unroll=1):
                    ab_consumer_state.advance()

                # Hand the mainloop gate to the other WG so it can start its mma.
                self.pingpong_barrier_arrive(Int32(1) - warp_group_idx, "mma")

                # Wait for permission to enter the epilogue (sD smem).
                self.pingpong_barrier_sync(warp_group_idx, "epi")

                self.epilogue_pingpong(
                    tiled_mma, epi_mC, epi_copy, sD, accumulators,
                    tile_coord_mnk, tidx, warp_idx, warp_group_idx,
                )

                # Hand the epilogue gate to the other WG.
                self.pingpong_barrier_arrive(Int32(1) - warp_group_idx, "epi")

                # Advance scheduler twice: skip the other WG's next tile.
                tile_scheduler.advance_to_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                iter_count = iter_count + Int32(1)


        return


    @cute.jit
    def epilogue_pingpong(
        self, tiled_mma, epi_mC, epi_copy, sD, accumulators,
        tile_coord_mnk, tidx, warp_idx, warp_group_idx,
    ):

        epi_sync_id = int(PingPongBarrier.EpiSync0) + warp_group_idx
        epi_sync_nthreads = self.consumer_warps_per_wg * cute.arch.WARP_SIZE

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
            self.dtype,
        )
        tiled_copy_r2s = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        gC_mnl = cute.local_tile(epi_mC, self.cta_tile_shape_mnk, tile_coord_mnk, proj=(1, 1, None))
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        rD_shape = cute.shape(thr_copy_r2s.partition_S(sD))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sD, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile_mn)
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            epi_copy, 0, cute.make_layout(1),
            sepi_for_tma_partition, tCgC_for_tma_partition,
        )
        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
        epi_tile_shape = tCgC_for_tma_partition.shape[1]
        epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))

        # TMA-store pipeline 
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )

        is_tma_warp = warp_idx == (warp_group_idx * 4)

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.dtype))


            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)])
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            cute.arch.barrier(barrier_id=epi_sync_id, number_of_threads=epi_sync_nthreads)

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            if is_tma_warp:

                cute.copy(epi_copy, bSG_sD[(None, epi_buffer)], bSG_gD[(None, gmem_coord)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            cute.arch.barrier(barrier_id=epi_sync_id, number_of_threads=epi_sync_nthreads)

        if is_tma_warp:
            c_pipeline.producer_tail()



if __name__ == "__main__":
    torch.manual_seed(0)
    # NOTE: K=64 is the absolute minimum (k_iters=1), needed for correctness.
    M, N, K = 512, 512, 64

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    C_ref = (A.float() @ B.float().T).to(torch.bfloat16)

    gemm = GemmPingPong(
        tile_shape_mn=(64, 256),
        epi_tile_mn=(64, 128),
        cluster_shape_mnk=(2, 1, 1),
        atom_layout_mn=(1, 1),
        ab_stage=3,
        epi_stage=2,
        reuse_ab=False,
        is_persistent=True,
    )
    compiled = compile_cutedsl((A, B, C), gemm)
    compiled(A, B, C, STREAM)
    torch.cuda.synchronize()

    abs_err = (C.float() - C_ref.float()).abs().max().item()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: M={M} N={N} K={K} bf16")
    print(f"max abs error vs torch: {abs_err:.4f}")

    cta_m, cta_n = 64, 256
    n_tiles_m, n_tiles_n = M // cta_m, N // cta_n
    diff = (C.float() - C_ref.float()).abs()
    print(f"\nPer-tile max abs diff ({n_tiles_m} M-tiles x {n_tiles_n} N-tiles):")
    for m_tile in range(n_tiles_m):
        for n_tile in range(n_tiles_n):
            m0, n0 = m_tile * cta_m, n_tile * cta_n
            d = diff[m0:m0 + cta_m, n0:n0 + cta_n].max().item()
            print(f"  tile ({m_tile},{n_tile})  M[{m0}:{m0 + cta_m}] N[{n0}:{n0 + cta_n}]  max diff = {d:.4f}")

    print(f"\nC[0:4, 0:4] from kernel:\n{C[0:4, 0:4]}")
    print(f"C_ref[0:4, 0:4]:\n{C_ref[0:4, 0:4]}")
