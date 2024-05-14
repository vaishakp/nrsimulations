GPU notes

1. Nvidia A100 gives 9.7TFLOPS at FP64. 
2. AMD EPYC processors give around 5.7TFLOPS.
3. Sonic 7352 gives 2.4 TFLOPS per node @3.2GHz.
4. Tesla K40 gives 1.4TFLOPS
5. Best in class GPGPU is AMD Mi300 instinct, which provides over 47TFLOPS and full 64bit support.
6. Some performance benchmarks are available at https://www.reddit.com/r/CUDA/comments/lkhcbv/is_there_a_list_of_gpus_ranked_by_fp64/
7. Thus at 64-bit arithmetic, using GPUs can at most give 2x speed improvement.
8. This improvement is only for high-end GPUs. NVIDIA e.g. does not support FP64 for many low-end GPUs
9. 32-bit arithmetic with no FMA gives significantly more errors.
10. The speedup using GPUs is easily attainable with compiler optimizations on CPUs 
11. For mixed precision or 32-bit precision, A100 gives around 19TFLOPS
