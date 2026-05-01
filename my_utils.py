# Copied over from fused_kernels_cutedsl_lib/python/cutedsl_kernels/gemm/attempt1
from typing import Type, Union, Optional
import cutlass
from cutlass import cute, const_expr
from cutlass.cute.nvgpu import warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass.utils import LayoutEnum


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    **kwargs,
):
    """Returns a callable to perform the G2S copy"""
    src_is_smem = cutlass.const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)

    s, g = cute.nvgpu.cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, cute.rank(smem_tensor) - 1),
        cute.group_modes(gmem_tensor, 0, cute.rank(gmem_tensor) - 1),
    )
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **kwargs2):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **kwargs2, **kwargs)
    return copy_tma, s, g


@dsl_user_op
def make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    tile: cute.Tile,
    stage: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    shape = cute.product_each(cute.shape(tile, loc=loc, ip=ip), loc=loc, ip=ip)
    major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(layout, dtype, major_mode_size),
        dtype,
    )
    order = (1, 0, 2) if const_expr(layout == LayoutEnum.COL_MAJOR) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, stage) if const_expr(stage is not None) else shape,
        order=order if const_expr(stage is not None) else order[:2],
    )
    return smem_layout_staged

# I ONLY use this for epi but in original quack codebase they use this for AB smem layout too(maybe)
make_smem_layout_epi = make_smem_layout