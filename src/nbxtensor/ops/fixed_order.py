"""Sums whose order no launch setting can change.

`tl.sum` reduces in the order of the launch's layout: the threads and warps a program spreads a block over set the
tree, so two launch settings of one kernel give two answers that differ in the last bits. A certified directory holds
one setting per memory class, and a key it does not certify is swept by timing at run time, so such a kernel gives
one release different bits on a 16 GB and a 32 GB card, or on two fresh installs. Measured on a V100, 2026-11-18: of
24 launch candidates of a [320, 512] softmax, 13 gave other bits than 1 row x 4 warps, and a served conversation's
fifth answer changed with them.

Here every step adds exactly two numbers, adjacent pairs, then adjacent pairs of those, down to one value; a + b is
b + a bit for bit, so no layout can change the result. The width summed over must be a power of two (every block
width these kernels use is one: `next_power_of_2` or a power-of-two candidate); masked lanes hold 0.

What these do NOT fix: a setting that changes WHICH partial sums are formed (a K chunk width, an attention tile
length) changes the association of the sum across chunks, whatever order each chunk is summed in."""
import triton
import triton.language as tl


@triton.jit
def row_sum_fixed_order(x, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    """Each row of x [ROWS, WIDTH] summed in one fixed pairwise order: a [ROWS] vector."""
    for i in tl.static_range(WIDTH.bit_length() - 1):
        x = tl.sum(tl.reshape(x, (ROWS, WIDTH >> (i + 1), 2)), axis=2)
    return tl.reshape(x, (ROWS,))


@triton.jit
def col_sum_fixed_order(x, HEIGHT: tl.constexpr, COLS: tl.constexpr):
    """Each column of x [HEIGHT, COLS] summed in one fixed pairwise order: a [COLS] vector."""
    for i in tl.static_range(HEIGHT.bit_length() - 1):
        x = tl.sum(tl.reshape(x, (HEIGHT >> (i + 1), 2, COLS)), axis=1)
    return tl.reshape(x, (COLS,))


@triton.jit
def row_dot_fixed_order(a, b, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    """Each row of a * b [ROWS, WIDTH] summed in one fixed order, the products included. A product summed by
    `row_sum_fixed_order(a * b)` is not enough: the compiler contracts a multiply and the add it feeds into one fused
    multiply-add, and which of the two products of a pair it fuses follows the layout (measured on a V100,
    2026-11-18: a gemv's rows per program and an RMS norm's warps still changed bits with the pairwise sum alone).
    Here the first level is written as one explicit fused multiply-add per pair, fma(a0, b0, a1 * b1), and every
    level after adds two rounded values, which nothing can contract. A width of 1 is the single product, rounded
    once: no pair to form, so no order to fix."""
    if WIDTH == 1:
        return tl.reshape(a * b, (ROWS,))
    else:
        a0, a1 = tl.split(tl.reshape(a, (ROWS, WIDTH // 2, 2)))
        b0, b1 = tl.split(tl.reshape(b, (ROWS, WIDTH // 2, 2)))
        return row_sum_fixed_order(tl.fma(a0, b0, a1 * b1), ROWS, WIDTH // 2)


@triton.jit
def col_dot_fixed_order(a, b, HEIGHT: tl.constexpr, COLS: tl.constexpr):
    """Each column of a * b [HEIGHT, COLS] summed in one fixed order, the products included (`row_dot_fixed_order`)."""
    a0, a1 = tl.split(tl.permute(tl.reshape(a, (HEIGHT // 2, 2, COLS)), (0, 2, 1)))
    b0, b1 = tl.split(tl.permute(tl.reshape(b, (HEIGHT // 2, 2, COLS)), (0, 2, 1)))
    return col_sum_fixed_order(tl.fma(a0, b0, a1 * b1), HEIGHT // 2, COLS)
