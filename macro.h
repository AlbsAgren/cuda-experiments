
// Detect error in kernel launch (launches fail silently by default)
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUDA_TIME(call) do { \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    (call); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    float ms; cudaEventElapsedTime(&ms, start, stop); \
    printf("\nCuda call time elapsed: %.6f\n", ms); \
} while(0)

#define HOST_TIME(call) do { \
    auto t1 = std::chrono::high_resolution_clock::now(); \
    (call); \
    auto t2 = std::chrono::high_resolution_clock::now(); \
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); \
    std::chrono::duration<double, std::milli> ms_double = t2 - t1; \
    printf("\nHost call time elapsed: %.6f\n", ms_double.count()); \
} while(0)
