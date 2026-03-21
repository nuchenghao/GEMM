/* Cross-platform symbol name: macOS requires leading '_', Linux does not */
#if defined(MAC)
#define FUNC(x) _##x
#else
#define FUNC(x) x
#endif

    .text
    .global FUNC(peak_sme_fmopa_1_fp32_fp32_fp32)
    .align 4
FUNC(peak_sme_fmopa_1_fp32_fp32_fp32):
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b
loop_peak_sme_fmopa_1_fp32_fp32_fp32:
    sub x0, x0, #1

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za0.s, p0/m, p1/m, z2.s, z3.s
    fmopa za0.s, p0/m, p1/m, z4.s, z5.s
    fmopa za0.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za0.s, p0/m, p1/m, z10.s, z11.s
    fmopa za0.s, p0/m, p1/m, z12.s, z13.s
    fmopa za0.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za0.s, p0/m, p1/m, z18.s, z19.s
    fmopa za0.s, p0/m, p1/m, z20.s, z21.s
    fmopa za0.s, p0/m, p1/m, z22.s, z23.s

    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za0.s, p0/m, p1/m, z26.s, z27.s
    fmopa za0.s, p0/m, p1/m, z28.s, z29.s
    fmopa za0.s, p0/m, p1/m, z30.s, z31.s

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za0.s, p0/m, p1/m, z2.s, z3.s
    fmopa za0.s, p0/m, p1/m, z4.s, z5.s
    fmopa za0.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za0.s, p0/m, p1/m, z10.s, z11.s
    fmopa za0.s, p0/m, p1/m, z12.s, z13.s
    fmopa za0.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za0.s, p0/m, p1/m, z18.s, z19.s
    fmopa za0.s, p0/m, p1/m, z20.s, z21.s
    fmopa za0.s, p0/m, p1/m, z22.s, z23.s

    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za0.s, p0/m, p1/m, z26.s, z27.s
    fmopa za0.s, p0/m, p1/m, z28.s, z29.s
    fmopa za0.s, p0/m, p1/m, z30.s, z31.s

    cbnz x0, loop_peak_sme_fmopa_1_fp32_fp32_fp32

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    mov x0, 32*512

    ret
