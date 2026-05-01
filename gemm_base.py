# Copied over from fused_kernels_cutedsl_lib/python/cutedsl_kernels/gemm/attempt1
import argparse
from typing import Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch
from triton.testing import do_bench

import cutlass
from cutlass import Boolean, Int32, const_expr
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

from tile_scheduler import SimpleTileSchedulerArguments, SimpleTileScheduler, RasterOrder, get_max_active_clusters
from cute_dsl_utils import ParamsBase
from my_utils import make_smem_layout_epi
from cdsl_helpers import shared, mma
from cdsl_fn_utils import make_fake_tensor

THREADS_PER_WG = 128

@cute.jit
def print0(x):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

@cute.jit
def printwg(x):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx%128 == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx%128 == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

class GemmSM90:
    def __init__(
        self,
        tile_shape_mn: Tuple[int, int],
        epi_tile_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        atom_layout_mn: Tuple[int, int],
        ab_stage: int = 2,
        epi_stage: int = 2,
        raster_order: RasterOrder = RasterOrder.AlongN,
        reuse_ab: bool = True,
        is_persistent: bool = False,
        gemm_n_prologue: int = 0,
        ):
        self.acc_dtype = cutlass.Float32
        self.raster_order = raster_order
        self.scheduler_group_size = Int32(8)
        self.cluster_shape_mnk = cluster_shape_mnk
        self.cluster_layout_mnk = None
        self.cta_tile_shape_mnk = (*tile_shape_mn, -1) # K-dim decided later
        tile_M, tile_N = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]

        # Atom layout
        assert atom_layout_mn[0] in [1, 2, 3] and atom_layout_mn[1] in [1, 2]
        assert atom_layout_mn[0] == 1 or tile_M % (atom_layout_mn[0] * 64) == 0
        assert atom_layout_mn[1] == 1 or tile_N % (atom_layout_mn[1] * 32) == 0
        self.atom_layout_mnk = (*atom_layout_mn, 1)

        # Multicast
        # How many times we have to multicast
        self.mcast_ctas_a = self.cluster_shape_mnk[1]
        self.mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.mcast_ctas_a > 1
        self.is_b_mcast = self.mcast_ctas_b > 1

        # Kernel config
        self.occupancy = 1
        self.mma_warpgroups = math.prod(self.atom_layout_mnk)
        assert self.mma_warpgroups in [1, 2, 3]
        self.threads_per_cta = (self.mma_warpgroups + 1) * THREADS_PER_WG
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes("sm_90")
        self.ab_load_warp_id = self.mma_warpgroups * 4
        
        # Registers
        self.num_regs_load, self.num_regs_mma = 40, 232

        self.ab_stage = ab_stage
        self.epi_stage = epi_stage

        # These are set when we run __call__
        self.dtype = cutlass.BFloat16
        self.a_layout, self.b_layout = None, None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024
        self.tma_ab_load_bytes = 0

        # Epilogue-related stuff
        self.epi_stage = epi_stage
        self.epi_tile_mn = epi_tile_mn
        self.reuse_ab = reuse_ab
        self.epi_smem_layout_staged = None
        self.epi_smem_size = 0 # populate later
        assert not (self.atom_layout_mnk[1] > 1) or self.epi_tile_mn[1] == self.cta_tile_shape_mnk[1], 'When atom layout n > 1, we need epi tile n = cta tile n'

        # Persistent
        self.is_persistent = is_persistent
        self.max_active_clusters = get_max_active_clusters(math.prod(cluster_shape_mnk))

        self.gemm_n_prologue = gemm_n_prologue

        # Checks
        assert not (self.reuse_ab and self.is_persistent), "Persistent kernel can't reuse AB for epilogue"

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream):
        # Populate fields
        self.populate_dtypes_and_layouts(a, b, c)
        self.populate_mma_atom()
        self.populate_smem_layouts()
        self.populate_shared_storage()

        # Epilogue TMA
        # Some GEMM's have D = A*B+C, naming is a bit messed up for this example
        tma_atom_d, tma_tensor_d = self._get_tma_epi_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile_mn,
        )

        # Convert cute.Tensors into TMA-compatible formats
        self.tma_ab_load_bytes = 0
        tma_atom_a, tma_tensor_a, _abytes = self._get_tma_load_and_tensors_incr_bytes(a, self.a_smem_layout_staged, (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]), self.mcast_ctas_a, self.dtype)
        tma_atom_b, tma_tensor_b, _bbytes = self._get_tma_load_and_tensors_incr_bytes(b, self.b_smem_layout_staged, (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]), self.mcast_ctas_b, self.dtype)
        self.tma_ab_load_bytes = _abytes + _bbytes

        # Tile scheduler arguments and grid
        ts_args = self.get_tile_scheduler_args(c)
        ts_params = SimpleTileScheduler.to_underlying_arguments(ts_args)
        grid = SimpleTileScheduler.get_grid_shape(ts_params, self.max_active_clusters)

        self.kernel(
            tma_atom_a, tma_atom_b,
            tma_tensor_a, tma_tensor_b,
            self.tiled_mma,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            ts_params,
            self.cluster_layout_mnk,
            self.epi_smem_layout_staged,
            tma_atom_d, tma_tensor_d
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=self.cluster_shape_mnk, stream=stream) # min_blocks_per_mp=1 only if kernel is large
    
    @cute.kernel
    def kernel(self,
               tma_atom_a: cute.CopyAtom,
               tma_atom_b: cute.CopyAtom,
               mA: cute.Tensor, mB: cute.Tensor,
               tiled_mma: cute.TiledMma,
               a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout,
               tile_sched_params: ParamsBase,
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

        ab_pipeline = self.make_ab_pipeline(storage.mainloop_pipeline_barriers.data_ptr(), cute.make_layout((1, *cluster_layout_mnk.shape)))
        pipeline_init_arrive()
        pipeline_init_wait()

        # Assign SMEM pointers
        sA = self.get_smem_field(storage, 'sA', a_smem_layout_staged)
        sB = self.get_smem_field(storage, 'sB', b_smem_layout_staged)
        # sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        # Pointer for epilogue
        sD = None
        if cutlass.const_expr(self.reuse_ab):
            sD_ptr = cute.recast_ptr(sA.iterator, epi_smem_layout.inner, dtype=self.dtype)
            sD = cute.make_tensor(sD_ptr, epi_smem_layout.outer)
        else:
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)

        tile_scheduler = SimpleTileScheduler.create(tile_sched_params)

        if warp_idx >= self.ab_load_warp_id: # Producer
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            if warp_idx == self.ab_load_warp_id: # Only TMA warp enters
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)

                # Multicast mask: slices along a mode to specify the CTAs to cast to
                a_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1)
                b_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0)
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx

                    # NOTE this part ignores the L dimension, no batched GEMM for now
                    gA_mk = cute.local_tile(mA, cute.select(self.cta_tile_shape_mnk, [0, 2]), (tile_coord_mnkl[0], None))
                    k_iters = cute.size(gA_mk, mode=[2]) # M, K, restK
                    
                    ab_producer_state = self.produce_mainloop(k_iters, ab_pipeline, ab_producer_state, tma_atom_a, tma_atom_b, mA, sA, mB, sB, tile_coord_mnkl, block_in_cluster_coord_mnk, cluster_layout_mnk, a_mcast_mask, b_mcast_mask) # TODO
                    tile_scheduler.fetch_next_work()
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx < self.ab_load_warp_id: # Consumer
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // THREADS_PER_WG)
            
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage) # TODO

            thr_mma = tiled_mma.get_slice(tidx)

            # NOTE these make descriptors for sA and sB
            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
            acc_shape = tiled_mma.partition_shape_C(
                cute.select(self.cta_tile_shape_mnk, mode=[0, 1])
            )
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                tile_coord_mnk = (work_tile.tile_idx[0], work_tile.tile_idx[1], work_tile.tile_idx[2])
                gA_mk = cute.local_tile(mA, cute.select(self.cta_tile_shape_mnk, [0, 2]), (tile_coord_mnk[0], None))
                k_iters = cute.size(gA_mk, mode=[2]) # m, k, restK

                # You have to return this, it doesn't let you modify a var inside a diff scope
                # SSA -- modifying a value means reassigning it, so you can't go into this fn scope and only modify the value there
                # you have to return it
                ab_consumer_state, tiled_mma = self.consume_mainloop(k_iters, tiled_mma, accumulators, ab_pipeline, ab_consumer_state, tCrA, tCrB, tidx, sA, sB)

                # Epilogue ##################################################
                self.epilogue(tiled_mma, epi_mC, epi_copy, sD, accumulators, tile_coord_mnk, tidx, warp_idx)

                # NOTE no need to fetch_next_work, that's done by the producer
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

        return

    # Main stuff
    # -----------------------------
    @cute.jit
    def epilogue(self, tiled_mma, epi_mC, epi_copy, sD, accumulators, tile_coord_mnk, tidx, warp_idx):
        # NOTE other gemm examples do a cluster arrive/wait, not sure why
        # We use a NamedBarrier since we can't syncthreads(only want to sync consumers)
        epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(1),
            num_threads=self.mma_warpgroups * 4 * cute.arch.WARP_SIZE
        )
        if const_expr(self.reuse_ab):
            epilogue_barrier.arrive_and_wait()
        
        # NOTE: In cutlass example, they initialize the copy differently, not sure why
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.dtype,
        )
        tiled_copy_r2s = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        # gC_mnl stores where our output tile should be
        gC_mnl = cute.local_tile(epi_mC, self.cta_tile_shape_mnk, tile_coord_mnk, proj=(1, 1, None))
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) # sD follows epi_smem_layout (m, n, stages)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        # Need to make accumulators that represents one stage of the epilogue, tRS_rAcc is all stages
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sD)) # registers needed for one epi tile
        # register layout, but for one stage of the epilogue
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        # sD has 4 stages, gD has 8 indices
        sepi_for_tma_partition = cute.group_modes(sD, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile_mn) # this just happens to be the right shape
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            epi_copy,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tCgC_for_tma_partition,
        )

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
        epi_tile_shape = tCgC_for_tma_partition.shape[1] # the layout of epi tiles
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        # this sets up a bulk wait pipeline(wait_group and commit_group)
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                # Take a slice of the accumulators
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
            
            # Type conversion
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.dtype))

            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            # R2S stmatrix
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            epilogue_barrier.arrive_and_wait() # Make sure stmatrix is done

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx) # e.g. (0, 0) to (7, 0)
            if warp_idx == 0:
                cute.copy(
                    epi_copy,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                c_pipeline.producer_commit() # commit_group
                c_pipeline.producer_acquire() # wait_group(stages-1)
            epilogue_barrier.arrive_and_wait() # Don't start next stmatrix yet
        
        if warp_idx == 0:
            c_pipeline.producer_tail() # wait_group(0)

    @cute.jit
    def produce_mainloop(
        self, k_iters: Int32, pipe: pipeline.PipelineAsync, state: pipeline.PipelineState, 
        tma_atom_a: cute.TiledCopy, tma_atom_b: cute.TiledCopy, 
        mA: cute.Tensor, sA: cute.Tensor, mB: cute.Tensor, sB: cute.Tensor, 
        tile_coord_mnkl: cute.Coord, block_in_cluster_coord_mnk: cute.Coord, cluster_layout_mnk: tuple, a_mcast_mask, b_mcast_mask
        ):
        peek_ab_empty_status = Boolean(True)
        if 0 < k_iters:
            peek_ab_empty_status = pipe.producer_try_acquire(state)

        for k_tile in cutlass.range(k_iters, unroll=1, unroll_full=False):
            pipe.producer_acquire(state, peek_ab_empty_status) # wait empty arrive full
            shared.tma_copy(tma_atom_a, mA, sA, self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2], tile_coord_mnkl[0], k_tile, pipe, state, block_in_cluster_coord_mnk[1], cute.make_layout(cute.slice_(cluster_layout_mnk, (0, None, 0)).shape), a_mcast_mask)
            shared.tma_copy(tma_atom_b, mB, sB, self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2], tile_coord_mnkl[1], k_tile, pipe, state, block_in_cluster_coord_mnk[0], cute.make_layout(cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape), b_mcast_mask)
            pipe.producer_commit(state)
            state.advance()

            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_iters:
                peek_ab_empty_status = pipe.producer_try_acquire(state)
        
        # NOTE: don't call producer tail here, you call it after all loads(from persistent) are done
        return state

    @cute.jit
    def consume_mainloop(self, k_iters: Int32, tiled_mma: cute.TiledMma, accumulators: cute.Tensor, pipe: pipeline.PipelineAsync, read_state: pipeline.PipelineState, tCrA: cute.Tensor, tCrB: cute.Tensor, tidx: Int32, sA: cute.Tensor, sB: cute.Tensor):
        release_state = read_state.clone() # NOTE: read is where we are reading from, release is when we are finished with the tile
        num_prologue_mma = min(self.gemm_n_prologue, k_iters)
        num_k_blocks = cute.size(tCrA, mode=[2])
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

        peek_ab_full_status = Boolean(True)
        if 0 < k_iters:
            peek_ab_full_status = pipe.consumer_try_wait(read_state)
        
        accumulate_O = False
        for k_tile in cutlass.range(num_prologue_mma):
            pipe.consumer_wait(read_state, peek_ab_full_status)
            mma.accumulating_gemm_ss(tidx, tiled_mma, sA, sB, accumulators, read_state, read_state, accumulate_O, -1)
            accumulate_O = True
            read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_iters:
                peek_ab_full_status = pipe.consumer_try_wait(read_state)

        for k_tile in cutlass.range(num_prologue_mma, k_iters, unroll=1, unroll_full=False):
            pipe.consumer_wait(read_state, peek_ab_full_status)
            mma.accumulating_gemm_ss(tidx, tiled_mma, sA, sB, accumulators, read_state, read_state, accumulate_O, -1)
            accumulate_O = True
            cute.nvgpu.warpgroup.wait_group(self.gemm_n_prologue)
            pipe.consumer_release(release_state)
            read_state.advance()
            release_state.advance()

            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_iters:
                peek_ab_full_status = pipe.consumer_try_wait(read_state)
        
        cute.nvgpu.warpgroup.wait_group(0)
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            pipe.consumer_release(release_state)
            release_state.advance()
        return read_state, tiled_mma

    # More runtime stuff
    # -----------------------------
    # def tma_partition(self, cluster_coord, tma_atom: cute.CopyAtom, sMatrix: cute.Tensor, gMatrix: cute.Tensor):
    #     s_tma = cute.group_modes(sMatrix, 0, 2)
    #     g_tma = cute.group_modes(gMatrix, 0, 2)

    #     # (TMA, pipe_stages) and (TMA, k)
    #     shared_layout, global_layout = cute.nvgpu.cpasync.tma_partition(
    #         tma_atom,
    #         cluster_coord,
    #         s_tma,
    #         g_tma,
    #     )
    #     return shared_layout, global_layout

    @cute.jit
    def make_ab_pipeline(self, mbar_ptr: cute.Pointer, cta_layout_vmnk: cute.Layout):
        num_producers = 1
        mcast_size = self.mcast_ctas_a + self.mcast_ctas_b - 1
        num_warps = self.mma_warpgroups * 4
        num_consumers = num_warps * mcast_size # IMPORTANT!!!

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_producers)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_consumers)
        # reminder: CTA layout is only used for syncing
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=self.ab_stage,
            tx_count=self.tma_ab_load_bytes,
            producer_group=producer_group,
            consumer_group=consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
        )

    def get_tile_scheduler_args(self, mC: cute.Tensor):
        batch_size = mC.shape[2] if cute.rank(mC.layout) == 3 else 1
        problem_shape_ntile_mnl = (
            cute.ceil_div(mC.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(mC.shape[1], self.cta_tile_shape_mnk[1]),
            batch_size,
        )
        tile_sched_args = SimpleTileSchedulerArguments(
            problem_shape_ntile_mnl,
            self.raster_order,
            self.scheduler_group_size,
            self.cluster_shape_mnk,
            self.is_persistent,
        )
        return tile_sched_args

    def _get_tma_load_and_tensors_incr_bytes(self, global_tensor: cute.Tensor, smem_layout_staged: cute.ComposedLayout, smem_tile: tuple[int, int], mcast_dim: bool, dtype: Type[cutlass.Numeric]) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp() if mcast_dim == 1 else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        # TMA tensor is just like a normal tensor but with 1@1, 1@0 etc. so the TMA can consume it
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            global_tensor,
            smem_layout,
            smem_tile, # CTA tiler
            num_multicast=mcast_dim
        )
        # self.tma_ab_load_bytes += cute.size_in_bytes(dtype, smem_layout)
        return tma_atom, tma_tensor, cute.size_in_bytes(dtype, smem_layout)
    
    @staticmethod
    def _get_tma_epi_atoms_and_tensors(
            tensor_d: cute.Tensor,
            epi_smem_layout_staged: cute.ComposedLayout,
            epi_tile: Tuple[int, int],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile) # change it to TMA-usable format with 1@0, 1@1 etc.

        op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Tiles D with d_cta_v_layout, prepares to copy to d
        tma_atom_d, tma_tensor_d = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor_d, epi_smem_layout, d_cta_v_layout
        )
        return tma_atom_d, tma_tensor_d

    # Easy Population Helpers
    # -------------------------------------
    def populate_dtypes_and_layouts(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        assert a.element_type == b.element_type == c.element_type == self.dtype
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self.cluster_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
    
    @cute.jit
    def populate_mma_atom(self):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1])
        )
        mma_k = 16
        mma_inst_tile_k = 4
        self.cta_tile_shape_mnk = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], mma_k * mma_inst_tile_k)
    
    def populate_smem_layouts(self):
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout, self.cta_tile_shape_mnk, self.dtype, self.ab_stage
        )

        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout, self.cta_tile_shape_mnk, self.dtype, self.ab_stage
        )

        self.epi_smem_layout_staged = make_smem_layout_epi(self.dtype, self.c_layout, self.epi_tile_mn, self.epi_stage)

        if not self.reuse_ab:
            self.epi_smem_size = cute.cosize(self.epi_smem_layout_staged)
    
    @cute.jit
    def populate_shared_storage(self):
        SharedStorage = type("SharedStorage", (), dict())
        SharedStorage.__annotations__['mainloop_pipeline_barriers'] = cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
        self.add_memrange(SharedStorage, 'sA', self.dtype, self.a_smem_layout_staged, self.buffer_align_bytes)
        self.add_memrange(SharedStorage, 'sB', self.dtype, self.b_smem_layout_staged, self.buffer_align_bytes)
        SharedStorage.__annotations__['sD'] = cute.struct.Align[cute.struct.MemRange[self.dtype, self.epi_smem_size], self.buffer_align_bytes]

        self.shared_storage = cute.struct(SharedStorage)
    
    def memrange(self, dtype, smem_layout, align):
        return cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(smem_layout)], align]
    
    def add_memrange(self, ss, name_field, dtype, smem_layout, align):
        ss.__annotations__[name_field] = cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(smem_layout)], align]
    
    def get_smem_field(self, storage, field_name, layout):
        return getattr(storage, field_name).get_tensor(layout.outer, swizzle=layout.inner)

if __name__ == "__main__":
    print('Starting...')
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    # args = parser.parse_args()
    # IS_NCU = args.mode == 'ncu'
    # IS_DEBUG = args.mode == 'debug'
    # IS_SPEED = args.mode == 'speed'

    # m, n, k = 65536, 4096, 8192
    # flops = 2 * m * n * k

    # def get_tflops(time_ms):
    #     return (flops / (time_ms / 1e3)) / 1e12

    # dtype = cutlass.BFloat16
    # div = math.gcd(128 // dtype.width, k)
    # divn = math.gcd(128 // dtype.width, n)
    # a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    # b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    # c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    # ref = a @ b.t()
    # convert_from_dlpack = lambda tensor: (
    #     from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
    #         mode=0, stride_order=(0, 1)
    #     )
    # )
    # # a_cute, b_cute, c_cute = [convert_from_dlpack(x) for x in (a, b, c)]
    # current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # gemm = GemmSM90(tile_shape_mn=(128, 256), 
    #                 epi_tile_mn=(128, 32),
    #                 cluster_shape_mnk=(2, 1, 1), 
    #                 atom_layout_mn=(2, 1),
    #                 ab_stage=3,
    #                 reuse_ab=False,
    #                 is_persistent=True)
    # a_c = make_fake_tensor(dtype, (m, k), div)
    # b_c = make_fake_tensor(dtype, (n, k), div)
    # c_c = make_fake_tensor(dtype, (m, n), divn)
    # compiled_gemm = cute.compile(gemm, a_c, b_c, c_c, current_stream, options='--enable-tvm-ffi')
    # compiled_gemm(a, b, c, current_stream)
    # if not IS_NCU:
    #     print('All close:', torch.allclose(ref, c))
    # if IS_DEBUG:
    #     print(ref)
    #     print(c)

    # if IS_DEBUG:
    #     n_incorrect = c.numel() - ((c - ref).abs() < 0.001).sum()
    #     print('n_incorrect :', n_incorrect)
    #     print('n_nonzero :', (c != 0).sum())

    # @torch.compile
    # def torch_gemm():
    #     return a @ b.t()
    
    # def cdsl_func(a, b):
    #     o = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device='cuda')
    #     compiled_gemm(a, b, o, current_stream)
    #     return o

    # if IS_SPEED:
    #     my_ms = do_bench(lambda: cdsl_func(a, b))
    #     other_ms = do_bench(torch_gemm)
    #     print(f'{my_ms=}, {other_ms=}')
    #     my_flops, other_flops = get_tflops(my_ms), get_tflops(other_ms)
    #     print(f'{my_flops=}, {other_flops=}')