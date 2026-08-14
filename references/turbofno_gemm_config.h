// 从 TurboFNO/utils/TurboFNO.h 提取的 GEMM 配置参数
// 用于指导 SUPA kernel 的 tile 配置

#define THREADBLOCK_M 64
#define THREADBLOCK_N 64
#define THREADBLOCK_K 8
#define WARP_M 32
#define WARP_N 16
#define THREAD_M 4
#define THREAD_N 4
#define WARP_NUM_ROW (THREADBLOCK_M / WARP_M)
#define THREAD_NUM_ROW (WARP_M / THREAD_M)
#define THREAD_NUM (THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N))

#define LOAD_PER_THREAD_A (THREADBLOCK_M * THREADBLOCK_K / THREAD_NUM)
#define LOAD_PER_THREAD_B (THREADBLOCK_N * THREADBLOCK_K / THREAD_NUM)

int threadblock_bs = 4;  // batch size inside thread block
