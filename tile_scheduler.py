# Copied over from fused_kernels_cutedsl_lib/python/cutedsl_kernels/gemm/attempt1
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from functools import lru_cache

import cutlass
from cute_dsl_utils import ArgumentsBase, ParamsBase
from fast_math import FastDivmod
from cutlass import Int32, Boolean
from cutlass import cute, pipeline


"""
Simple tile scheduler, just rasterize stuff and no persistent schedule

Clusters are always (m, n). Cluster coords are also M, N but rasterization dictates how we map 1D --> 2D
"""

@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)

class RasterOrder(IntEnum):
    """Set M/N as fast dimension"""
    AlongM = 0
    AlongN = 1

@dataclass
class SimpleTileSchedulerArguments(ArgumentsBase):
    ntiles_mnl: cute.Shape
    raster_order: cutlass.Constexpr[RasterOrder]
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    is_persistent: cutlass.Constexpr[bool]

class SimpleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        dm_clusters_per_problem: FastDivmod
        num_groups_regular: Int32
        dm_group_size: FastDivmod
        dm_group_size_tail: FastDivmod
        dm_num_clusters_in_group: FastDivmod
        cluster_shape_mn: cutlass.Constexpr[cute.Shape]
        is_persistent: cutlass.Constexpr[bool]

        @staticmethod
        @cute.jit
        def create(args: SimpleTileSchedulerArguments, *, loc=None, ip=None):
            assert args.cluster_shape_mnk[2] == 1
            cluster_shape_mn = cutlass.const_expr(cute.select(args.cluster_shape_mnk, mode=[0, 1]))
            ntile_mn = cute.select(args.ntiles_mnl, mode=[0, 1])
            ncluster_mn = cute.ceil_div(ntile_mn, cluster_shape_mn)
            ncluster_mnl = ncluster_mn + (args.ntiles_mnl[2], )
            clusters_per_problem = cute.size(ncluster_mn)
            raster_order = args.raster_order
            ncluster_fast = ncluster_mn[0] if raster_order == RasterOrder.AlongM else ncluster_mn[1]
            ncluster_slow = ncluster_mn[1] if raster_order == RasterOrder.AlongM else ncluster_mn[0]
            
            # Each "group" are rows of clusters that take up (nSlow, groupSize) space, so we organize them side-by-side
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size # Number of regular(full) groups
            num_clusters_in_group = group_size * ncluster_slow
            return SimpleTileScheduler.Params(
                ncluster_mnl,
                raster_order,
                FastDivmod.create(clusters_per_problem),
                num_groups_regular,
                FastDivmod.create(group_size),
                FastDivmod.create(group_size_tail if group_size_tail > 0 else 1), # DO NOT DIVIDE BY 0
                FastDivmod.create(num_clusters_in_group),
                cluster_shape_mn,
                args.is_persistent,
            )
    
    # Int32 has an MLIR type so I guess we should use it
    def __init__(self, current_work_idx: Int32, batch_idx: Int32, num_tiles_executed: Int32, params: Params, *, loc=None, ip=None):
        self._cluster_id_in_problem = current_work_idx # disregard batch
        self._batch_idx = batch_idx
        self.num_tiles_executed = num_tiles_executed
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: SimpleTileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SimpleTileScheduler.Params.create(args, loc=loc, ip=ip)
    
    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SimpleTileScheduler":
        batch_idx = Int32(0) # just hardcode, not sure what to do for now

        # This is where we decide the linear layout of the clusters
        if cutlass.const_expr(not params.is_persistent):
            cidx, cidy, _ = cute.arch.cluster_idx()
            cdimx, _, _ = cute.arch.cluster_dim()
            cluster_id = cidx + cidy * cdimx
            current_work_linear_idx = Int32(cluster_id)
        else:
            _, _, bidz = cute.arch.block_idx() # see get_grid_shape for how persistent grid is managed
            current_work_linear_idx = Int32(bidz)
        return SimpleTileScheduler(Int32(current_work_linear_idx), batch_idx, Int32(0), params, loc=loc, ip=ip)


    @staticmethod
    def get_grid_shape(params: Params, max_active_clusters: Int32, *, loc=None, ip=None) -> tuple[Int32, Int32, Int32]:
        num_ctas_mnl = tuple(
            x * y for x, y in zip(params.ncluster_mnl, params.cluster_shape_mn)
        ) + (params.ncluster_mnl[2],)
        if cutlass.const_expr(not params.is_persistent):
            return num_ctas_mnl
        else:
            num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)
            num_ctas_in_cluster = cute.size(params.cluster_shape_mn, loc=loc, ip=ip)
            num_ctas_per_wave = max_active_clusters * num_ctas_in_cluster
            num_persistent_CTAs = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_CTAs // num_ctas_in_cluster
            return (*params.cluster_shape_mn, num_persistent_clusters)

    @cute.jit
    def _map_cta_coords(self, cluster_id_in_problem: Int32, *, loc=None, ip=None) -> tuple[Int32, Int32]:
        # Map the coords to a grouped grid shape, returning the M and N coordinate
        params = self.params
        group_id, id_in_group = params.dm_num_clusters_in_group.divmod(cluster_id_in_problem)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0) # cluster ID
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = params.dm_group_size.divmod(id_in_group)
        else: # tail
            cid_slow, cid_fast_in_group = params.dm_group_size_tail.divmod(id_in_group)
        if group_id % 2 == 1: # every second group, slow goes opposite way for a serpentine order
            ncluster_slow = (
                params.ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else params.ncluster_mnl[0]
            )
            cid_slow = ncluster_slow - 1 - cid_slow # reverse order
        cid_fast = group_id * params.dm_group_size.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        cluster_id_in_problem = self._cluster_id_in_problem

        # Cluster M and N in the 2D grid of clusters
        cid_m, cid_n = self._map_cta_coords(cluster_id_in_problem, loc=loc, ip=ip)

        # Get PID
        bidx_in_cluster = cute.arch.block_in_cluster_idx() # xyz
        pid_m = cid_m * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * params.cluster_shape_mn[1] + bidx_in_cluster[1]
        batch_idx = self._batch_idx # no permutation on batch idx
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        if cutlass.const_expr(not params.is_persistent):
            is_valid = self.num_tiles_executed == 0
        else:
            is_valid = self._cluster_id_in_problem < cute.size(params.ncluster_mnl)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def fetch_next_work(self, *, loc=None, ip=None):
        pass

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None):
        # If not persistent, this shouldn't matter.
        if cutlass.const_expr(self.params.is_persistent):
            num_persistent_clusters = cute.arch.grid_dim()[2]
            self._cluster_id_in_problem += Int32(num_persistent_clusters)

        # This needs to be done either way
        self.num_tiles_executed += Int32(1)
    
    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._cluster_id_in_problem,
            self._batch_idx,
            self.num_tiles_executed,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
            self._cluster_id_in_problem,
            self._batch_idx,
            self.num_tiles_executed,
            self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)