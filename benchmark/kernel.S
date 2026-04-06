/* Cross-platform symbol name: macOS requires leading '_', Linux does not */
#if defined(__APPLE__)
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



    .global FUNC(peak_sme_fmopa_1_fp32_fp32_fp32_za0)
    .align 4
FUNC(peak_sme_fmopa_1_fp32_fp32_fp32_za0):
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b
loop_peak_sme_fmopa_1_fp32_fp32_fp32_za0:
    sub x0, x0, #1

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za0.s, p0/m, p1/m, z2.s, z3.s
    fmopa za0.s, p0/m, p1/m, z4.s, z5.s
    fmopa za0.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za0.s, p0/m, p1/m, z10.s, z11.s
    fmopa za0.s, p0/m, p1/m, z12.s, z13.s
    fmopa za0.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za0.s, p0/m, p1/m, z2.s, z3.s
    fmopa za0.s, p0/m, p1/m, z4.s, z5.s
    fmopa za0.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za0.s, p0/m, p1/m, z10.s, z11.s
    fmopa za0.s, p0/m, p1/m, z12.s, z13.s
    fmopa za0.s, p0/m, p1/m, z14.s, z15.s


    cbnz x0, loop_peak_sme_fmopa_1_fp32_fp32_fp32_za0

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    mov x0, 16 * 512

    ret





    .global FUNC(peak_sme_fmopa_1_fp32_fp32_fp32_za1)
    .align 4
FUNC(peak_sme_fmopa_1_fp32_fp32_fp32_za1):
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b
loop_peak_sme_fmopa_1_fp32_fp32_fp32_za1:
    sub x0, x0, #1

    fmopa za1.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za1.s, p0/m, p1/m, z4.s, z5.s
    fmopa za1.s, p0/m, p1/m, z6.s, z7.s

    fmopa za1.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za1.s, p0/m, p1/m, z12.s, z13.s
    fmopa za1.s, p0/m, p1/m, z14.s, z15.s

    fmopa za1.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za1.s, p0/m, p1/m, z4.s, z5.s
    fmopa za1.s, p0/m, p1/m, z6.s, z7.s

    fmopa za1.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za1.s, p0/m, p1/m, z12.s, z13.s
    fmopa za1.s, p0/m, p1/m, z14.s, z15.s


    cbnz x0, loop_peak_sme_fmopa_1_fp32_fp32_fp32_za1

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    mov x0, 16*512

    ret



    .global FUNC(peak_sme_fmopa_1_fp32_fp32_fp32_za2)
    .align 4
FUNC(peak_sme_fmopa_1_fp32_fp32_fp32_za2):
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b
loop_peak_sme_fmopa_1_fp32_fp32_fp32_za2:
    sub x0, x0, #1

    fmopa za2.s, p0/m, p1/m, z0.s, z1.s
    fmopa za2.s, p0/m, p1/m, z2.s, z3.s
    fmopa za2.s, p0/m, p1/m, z4.s, z5.s
    fmopa za2.s, p0/m, p1/m, z6.s, z7.s

    fmopa za2.s, p0/m, p1/m, z8.s, z9.s
    fmopa za2.s, p0/m, p1/m, z10.s, z11.s
    fmopa za2.s, p0/m, p1/m, z12.s, z13.s
    fmopa za2.s, p0/m, p1/m, z14.s, z15.s

    fmopa za2.s, p0/m, p1/m, z16.s, z17.s
    fmopa za2.s, p0/m, p1/m, z18.s, z19.s
    fmopa za2.s, p0/m, p1/m, z20.s, z21.s
    fmopa za2.s, p0/m, p1/m, z22.s, z23.s

    fmopa za2.s, p0/m, p1/m, z24.s, z25.s
    fmopa za2.s, p0/m, p1/m, z26.s, z27.s
    fmopa za2.s, p0/m, p1/m, z28.s, z29.s
    fmopa za2.s, p0/m, p1/m, z30.s, z31.s

    fmopa za2.s, p0/m, p1/m, z0.s, z1.s
    fmopa za2.s, p0/m, p1/m, z2.s, z3.s
    fmopa za2.s, p0/m, p1/m, z4.s, z5.s
    fmopa za2.s, p0/m, p1/m, z6.s, z7.s

    fmopa za2.s, p0/m, p1/m, z8.s, z9.s
    fmopa za2.s, p0/m, p1/m, z10.s, z11.s
    fmopa za2.s, p0/m, p1/m, z12.s, z13.s
    fmopa za2.s, p0/m, p1/m, z14.s, z15.s

    fmopa za2.s, p0/m, p1/m, z16.s, z17.s
    fmopa za2.s, p0/m, p1/m, z18.s, z19.s
    fmopa za2.s, p0/m, p1/m, z20.s, z21.s
    fmopa za2.s, p0/m, p1/m, z22.s, z23.s

    fmopa za2.s, p0/m, p1/m, z24.s, z25.s
    fmopa za2.s, p0/m, p1/m, z26.s, z27.s
    fmopa za2.s, p0/m, p1/m, z28.s, z29.s
    fmopa za2.s, p0/m, p1/m, z30.s, z31.s

    cbnz x0, loop_peak_sme_fmopa_1_fp32_fp32_fp32_za2

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    mov x0, 32*512

    ret



    .global FUNC(peak_sme_fmopa_1_fp32_fp32_fp32_za3)
    .align 4
FUNC(peak_sme_fmopa_1_fp32_fp32_fp32_za3):
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b
loop_peak_sme_fmopa_1_fp32_fp32_fp32_za3:
    sub x0, x0, #1

    fmopa za3.s, p0/m, p1/m, z0.s, z1.s
    fmopa za3.s, p0/m, p1/m, z2.s, z3.s
    fmopa za3.s, p0/m, p1/m, z4.s, z5.s
    fmopa za3.s, p0/m, p1/m, z6.s, z7.s

    fmopa za3.s, p0/m, p1/m, z8.s, z9.s
    fmopa za3.s, p0/m, p1/m, z10.s, z11.s
    fmopa za3.s, p0/m, p1/m, z12.s, z13.s
    fmopa za3.s, p0/m, p1/m, z14.s, z15.s

    fmopa za3.s, p0/m, p1/m, z16.s, z17.s
    fmopa za3.s, p0/m, p1/m, z18.s, z19.s
    fmopa za3.s, p0/m, p1/m, z20.s, z21.s
    fmopa za3.s, p0/m, p1/m, z22.s, z23.s

    fmopa za3.s, p0/m, p1/m, z24.s, z25.s
    fmopa za3.s, p0/m, p1/m, z26.s, z27.s
    fmopa za3.s, p0/m, p1/m, z28.s, z29.s
    fmopa za3.s, p0/m, p1/m, z30.s, z31.s

    fmopa za3.s, p0/m, p1/m, z0.s, z1.s
    fmopa za3.s, p0/m, p1/m, z2.s, z3.s
    fmopa za3.s, p0/m, p1/m, z4.s, z5.s
    fmopa za3.s, p0/m, p1/m, z6.s, z7.s

    fmopa za3.s, p0/m, p1/m, z8.s, z9.s
    fmopa za3.s, p0/m, p1/m, z10.s, z11.s
    fmopa za3.s, p0/m, p1/m, z12.s, z13.s
    fmopa za3.s, p0/m, p1/m, z14.s, z15.s

    fmopa za3.s, p0/m, p1/m, z16.s, z17.s
    fmopa za3.s, p0/m, p1/m, z18.s, z19.s
    fmopa za3.s, p0/m, p1/m, z20.s, z21.s
    fmopa za3.s, p0/m, p1/m, z22.s, z23.s

    fmopa za3.s, p0/m, p1/m, z24.s, z25.s
    fmopa za3.s, p0/m, p1/m, z26.s, z27.s
    fmopa za3.s, p0/m, p1/m, z28.s, z29.s
    fmopa za3.s, p0/m, p1/m, z30.s, z31.s

    cbnz x0, loop_peak_sme_fmopa_1_fp32_fp32_fp32_za3

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    mov x0, 32*512

    ret



    .global FUNC(peak_sme_fmopa_2_fp32_fp32_fp32)
    .align 4
FUNC(peak_sme_fmopa_2_fp32_fp32_fp32):
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b
loop_peak_sme_fmopa_2_fp32_fp32_fp32:
    sub x0, x0, #1

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za0.s, p0/m, p1/m, z4.s, z5.s
    fmopa za1.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za0.s, p0/m, p1/m, z12.s, z13.s
    fmopa za1.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za1.s, p0/m, p1/m, z18.s, z19.s
    fmopa za0.s, p0/m, p1/m, z20.s, z21.s
    fmopa za1.s, p0/m, p1/m, z22.s, z23.s

    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za1.s, p0/m, p1/m, z26.s, z27.s
    fmopa za0.s, p0/m, p1/m, z28.s, z29.s
    fmopa za1.s, p0/m, p1/m, z30.s, z31.s

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za0.s, p0/m, p1/m, z4.s, z5.s
    fmopa za1.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za0.s, p0/m, p1/m, z12.s, z13.s
    fmopa za1.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za1.s, p0/m, p1/m, z18.s, z19.s
    fmopa za0.s, p0/m, p1/m, z20.s, z21.s
    fmopa za1.s, p0/m, p1/m, z22.s, z23.s

    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za1.s, p0/m, p1/m, z26.s, z27.s
    fmopa za0.s, p0/m, p1/m, z28.s, z29.s
    fmopa za1.s, p0/m, p1/m, z30.s, z31.s

    cbnz x0, loop_peak_sme_fmopa_2_fp32_fp32_fp32

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    mov x0, 32*512

    ret





    .global FUNC(peak_sme_fmopa_3_fp32_fp32_fp32)
    .align 4
FUNC(peak_sme_fmopa_3_fp32_fp32_fp32):
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b
    loop_peak_sme_fmopa_3_fp32_fp32_fp32:
    sub x0, x0, #1

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za2.s, p0/m, p1/m, z4.s, z5.s
    fmopa za0.s, p0/m, p1/m, z6.s, z7.s

    fmopa za1.s, p0/m, p1/m, z8.s, z9.s
    fmopa za2.s, p0/m, p1/m, z10.s, z11.s
    fmopa za0.s, p0/m, p1/m, z12.s, z13.s
    fmopa za1.s, p0/m, p1/m, z14.s, z15.s

    fmopa za2.s, p0/m, p1/m, z16.s, z17.s
    fmopa za0.s, p0/m, p1/m, z18.s, z19.s
    fmopa za1.s, p0/m, p1/m, z20.s, z21.s
    fmopa za2.s, p0/m, p1/m, z22.s, z23.s

    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za1.s, p0/m, p1/m, z26.s, z27.s
    fmopa za2.s, p0/m, p1/m, z28.s, z29.s
    fmopa za0.s, p0/m, p1/m, z30.s, z31.s

    fmopa za1.s, p0/m, p1/m, z0.s, z1.s
    fmopa za2.s, p0/m, p1/m, z2.s, z3.s
    fmopa za0.s, p0/m, p1/m, z4.s, z5.s
    fmopa za1.s, p0/m, p1/m, z6.s, z7.s

    fmopa za2.s, p0/m, p1/m, z8.s, z9.s
    fmopa za0.s, p0/m, p1/m, z10.s, z11.s
    fmopa za1.s, p0/m, p1/m, z12.s, z13.s
    fmopa za2.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za1.s, p0/m, p1/m, z18.s, z19.s
    fmopa za2.s, p0/m, p1/m, z20.s, z21.s
    fmopa za0.s, p0/m, p1/m, z22.s, z23.s

    fmopa za1.s, p0/m, p1/m, z24.s, z25.s
    fmopa za2.s, p0/m, p1/m, z26.s, z27.s
    fmopa za0.s, p0/m, p1/m, z28.s, z29.s
    fmopa za1.s, p0/m, p1/m, z30.s, z31.s
    
    //add
    fmopa za2.s, p0/m, p1/m, z0.s, z1.s

    cbnz x0, loop_peak_sme_fmopa_3_fp32_fp32_fp32

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    mov x0, 33*512

    ret



    .global FUNC(peak_sme_fmopa_4_fp32_fp32_fp32)
    .align 4
FUNC(peak_sme_fmopa_4_fp32_fp32_fp32):
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b
loop_peak_sme_fmopa_4_fp32_fp32_fp32:
    sub x0, x0, #1

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za2.s, p0/m, p1/m, z4.s, z5.s
    fmopa za3.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za2.s, p0/m, p1/m, z12.s, z13.s
    fmopa za3.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za1.s, p0/m, p1/m, z18.s, z19.s
    fmopa za2.s, p0/m, p1/m, z20.s, z21.s
    fmopa za3.s, p0/m, p1/m, z22.s, z23.s

    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za1.s, p0/m, p1/m, z26.s, z27.s
    fmopa za2.s, p0/m, p1/m, z28.s, z29.s
    fmopa za3.s, p0/m, p1/m, z30.s, z31.s

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za2.s, p0/m, p1/m, z4.s, z5.s
    fmopa za3.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za2.s, p0/m, p1/m, z12.s, z13.s
    fmopa za3.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za1.s, p0/m, p1/m, z18.s, z19.s
    fmopa za2.s, p0/m, p1/m, z20.s, z21.s
    fmopa za3.s, p0/m, p1/m, z22.s, z23.s
    
    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za1.s, p0/m, p1/m, z26.s, z27.s
    fmopa za2.s, p0/m, p1/m, z28.s, z29.s
    fmopa za3.s, p0/m, p1/m, z30.s, z31.s

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za2.s, p0/m, p1/m, z4.s, z5.s
    fmopa za3.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za2.s, p0/m, p1/m, z12.s, z13.s
    fmopa za3.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za1.s, p0/m, p1/m, z18.s, z19.s
    fmopa za2.s, p0/m, p1/m, z20.s, z21.s
    fmopa za3.s, p0/m, p1/m, z22.s, z23.s

    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za1.s, p0/m, p1/m, z26.s, z27.s
    fmopa za2.s, p0/m, p1/m, z28.s, z29.s
    fmopa za3.s, p0/m, p1/m, z30.s, z31.s

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za2.s, p0/m, p1/m, z4.s, z5.s
    fmopa za3.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za2.s, p0/m, p1/m, z12.s, z13.s
    fmopa za3.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za1.s, p0/m, p1/m, z18.s, z19.s
    fmopa za2.s, p0/m, p1/m, z20.s, z21.s
    fmopa za3.s, p0/m, p1/m, z22.s, z23.s
    
    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za1.s, p0/m, p1/m, z26.s, z27.s
    fmopa za2.s, p0/m, p1/m, z28.s, z29.s
    fmopa za3.s, p0/m, p1/m, z30.s, z31.s

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za2.s, p0/m, p1/m, z4.s, z5.s
    fmopa za3.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za2.s, p0/m, p1/m, z12.s, z13.s
    fmopa za3.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za1.s, p0/m, p1/m, z18.s, z19.s
    fmopa za2.s, p0/m, p1/m, z20.s, z21.s
    fmopa za3.s, p0/m, p1/m, z22.s, z23.s

    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za1.s, p0/m, p1/m, z26.s, z27.s
    fmopa za2.s, p0/m, p1/m, z28.s, z29.s
    fmopa za3.s, p0/m, p1/m, z30.s, z31.s

    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z2.s, z3.s
    fmopa za2.s, p0/m, p1/m, z4.s, z5.s
    fmopa za3.s, p0/m, p1/m, z6.s, z7.s

    fmopa za0.s, p0/m, p1/m, z8.s, z9.s
    fmopa za1.s, p0/m, p1/m, z10.s, z11.s
    fmopa za2.s, p0/m, p1/m, z12.s, z13.s
    fmopa za3.s, p0/m, p1/m, z14.s, z15.s

    fmopa za0.s, p0/m, p1/m, z16.s, z17.s
    fmopa za1.s, p0/m, p1/m, z18.s, z19.s
    fmopa za2.s, p0/m, p1/m, z20.s, z21.s
    fmopa za3.s, p0/m, p1/m, z22.s, z23.s
    
    fmopa za0.s, p0/m, p1/m, z24.s, z25.s
    fmopa za1.s, p0/m, p1/m, z26.s, z27.s
    fmopa za2.s, p0/m, p1/m, z28.s, z29.s
    fmopa za3.s, p0/m, p1/m, z30.s, z31.s

    cbnz x0, loop_peak_sme_fmopa_4_fp32_fp32_fp32

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    mov x0, 32*512*3
    ret



    .global FUNC(load_data_sme_ldr_za)
    .align 4
FUNC(load_data_sme_ldr_za):
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    smstart

ldr_za_outer_loop:
    sub x0, x0, #1

    mov x4, x1
    mov x5, x2

    add x8, x2, #0
    add x9, x2, #16*4
    add x10, x2, #32*4
    add x11, x2, #48*4

ldr_za_inner_loop:
    sub x4, x4, #512
    mov x6, #8

    mov w12, #0
    mov w13, #1
    mov w14, #2
    mov w15, #3

ldr_za_full_tile_loop:
    
    sub x6, x6, #1

    ldr za[w12, #0], [x8]
    ldr za[w13, #0], [x9]
    ldr za[w14, #0], [x10]
    ldr za[w15, #0], [x11]

    add w12, w12, #4
    add w13, w13, #4
    add w14, w14, #4
    add w15, w15, #4 

    add x8, x8, #64*4
    add x9, x9, #64*4
    add x10, x10, #64*4
    add x11, x11, #64*4

    cbnz x6, ldr_za_full_tile_loop

    cbnz x4, ldr_za_inner_loop

    cbnz x0, ldr_za_outer_loop
    
    smstop


    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ret