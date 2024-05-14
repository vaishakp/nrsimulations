GPU notes

1. Nvidia A100 gives 9.7TFLOPS at FP64. 
2. AMD EPYC processors give around 5.7TFLOPS. Sonic 7352 gives 2.4 TFLOPS per node.
3. Tesla K40 gives 1.4TFLOPS
4. Thus at 64-bit arithmetic, using GPUs can at most give 2x speed improvement.
5. This improvement is only for high-end GPUs. NVIDIA e.g. does not support FP64 for many low-end GPUs
6. 32-bit arithmetic with no FMA gives significantly more errors.
7. The speedup using GPUs is easily attainable with compiler optimizations on CPUs 
8. For mixed precision or 32-bit precision, A100 gives around 19TFLOPS
