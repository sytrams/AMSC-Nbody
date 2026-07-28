#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARPS (BLOCK_SIZE / WARP_SIZE)

#define BITS 8
#define BUCKETS (1 << BITS)   // 256

___global__ void block_histogram(const uint64_t* __restrict__ keys, uint32_t* __restrict__ block_histo, int shift, int N)
{
    __shared__ uint32_t warp_histo[WARPS][BUCKETS];

    int tid  = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // zero warp histograms
    for (int i = lane; i < BUCKETS; i += WARP_SIZE)
        warp_histo[warp][i] = 0;

    __syncthreads();

    int gid = blockIdx.x * blockDim.x + tid;

    // grid-stride loop
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) 
    {
        uint32_t d = (keys[i] >> shift) & (BUCKETS - 1);

        // warp-local increment via ballot
        for (int b = 0; b < BUCKETS; b++) 
        {
            uint32_t mask = __ballot_sync(0xffffffff, d == b);
            if (lane == 0)
                warp_histo[warp][b] += __popc(mask);
        }
    }

    __syncthreads();

    // reduce warps → block histogram
    for (int b = tid; b < BUCKETS; b += BLOCK_SIZE) 
    {
        uint32_t sum = 0;
        for (int w = 0; w < WARPS; w++)
            sum += warp_histo[w][b];

        block_histo[blockIdx.x * BUCKETS + b] = sum;
    }
}
__global__ void scan_block_histograms(const uint32_t* __restrict__ block_histo, uint32_t* __restrict__ block_offsets, int numBlocks)
{
    int bucket = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ uint32_t temp[];

    // load column
    if (tid < numBlocks)
        temp[tid] = block_histo[tid * BUCKETS + bucket];
    else
        temp[tid] = 0;

    __syncthreads();

    // exclusive scan (Blelloch)
    for (int offset = 1; offset < numBlocks; offset <<= 1) 
    {
        uint32_t val = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (tid < numBlocks) 
    {
        uint32_t original = block_histo[tid * BUCKETS + bucket];
        block_offsets[tid * BUCKETS + bucket] = temp[tid] - original;
    }
}

__global__ void compute_global_offsets(const uint32_t* block_histo, uint32_t* global_offsets, int numBlocks)
{
    __shared__ uint32_t bucket_sum[BUCKETS];

    int b = threadIdx.x;

    uint32_t sum = 0;
    for (int i = 0; i < numBlocks; i++)
        sum += block_histo[i * BUCKETS + b];

    bucket_sum[b] = sum;
    __syncthreads();

    // exclusive scan across buckets
    for (int offset = 1; offset < BUCKETS; offset <<= 1) 
    {
        uint32_t val = (b >= offset) ? bucket_sum[b - offset] : 0;
        __syncthreads();
        bucket_sum[b] += val;
        __syncthreads();
    }

    global_offsets[b] = bucket_sum[b] - sum;
}

__device__ __forceinline__
uint32_t warp_prefix(uint32_t pred)
{
    uint32_t mask = __ballot_sync(0xffffffff, pred);
    int lane = threadIdx.x & 31;
    return __popc(mask & ((1u << lane) - 1));
}

__global__ void scatter(const uint64_t* __restrict__ keys_in, uint64_t* __restrict__ keys_out, const uint32_t* __restrict__ idx_in, uint32_t* __restrict__ idx_out, 
                        const uint32_t* __restrict__ block_offsets, const uint32_t* __restrict__ global_offsets, int shift, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid >= N) return;

    uint64_t key = keys_in[gid];
    uint32_t idx = idx_in[gid];

    uint32_t d = (key >> shift) & (BUCKETS - 1);

    // rank within warp for THIS bucket
    uint32_t rank = warp_prefix(((key >> shift) & (BUCKETS - 1)) == d);

    // count how many threads in warp match
    uint32_t warp_count =__popc(__ballot_sync(0xffffffff,((key >> shift) & (BUCKETS - 1)) == d));

    // shared memory to accumulate warp offsets
    __shared__ uint32_t warp_offsets[WARPS][BUCKETS];

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    if (lane == 0)
        warp_offsets[warp][d] = warp_count;

    __syncthreads();

    // prefix sum across warps (per bucket)
    uint32_t warp_base = 0;
    for (int w = 0; w < warp; w++)
        warp_base += warp_offsets[w][d];

    uint32_t block_base = block_offsets[blockIdx.x * BUCKETS + d];

    uint32_t global_base = global_offsets[d];

    uint32_t pos = global_base + block_base + warp_base + rank;

    keys_out[pos] = key;
    idx_out[pos]  = idx;
}

void radix_sort_gpu(uint64_t* d_keys, uint32_t* d_idx, int N)
{
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint64_t *d_keys_tmp;
    uint32_t *d_idx_tmp;

    cudaMalloc(&d_keys_tmp, N * sizeof(uint64_t));
    cudaMalloc(&d_idx_tmp, N * sizeof(uint32_t));

    uint32_t *d_block_histo, *d_block_offsets, *d_global_offsets;

    cudaMalloc(&d_block_histo, numBlocks * BUCKETS * sizeof(uint32_t));
    cudaMalloc(&d_block_offsets, numBlocks * BUCKETS * sizeof(uint32_t));
    cudaMalloc(&d_global_offsets, BUCKETS * sizeof(uint32_t));

    for (int shift = 0; shift < 64; shift += BITS) 
    {
        block_histogram<<<numBlocks, BLOCK_SIZE>>>(d_keys, d_block_histo, shift, N);

        scan_block_histograms<<<BUCKETS, numBlocks, numBlocks * sizeof(uint32_t)>>>(d_block_histo, d_block_offsets, numBlocks);

        compute_global_offsets<<<1, BUCKETS>>>(d_block_histo, d_global_offsets, numBlocks);

        scatter<<<numBlocks, BLOCK_SIZE>>>(d_keys, d_keys_tmp, d_idx, d_idx_tmp, d_block_offsets, d_global_offsets, shift, N);

        std::swap(d_keys, d_keys_tmp);
        std::swap(d_idx, d_idx_tmp);
    }

    cudaFree(d_keys_tmp);
    cudaFree(d_idx_tmp);
    cudaFree(d_block_histo);
    cudaFree(d_block_offsets);
    cudaFree(d_global_offsets);
}