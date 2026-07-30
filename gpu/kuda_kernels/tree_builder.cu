module;

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>


module tree.builder;
import tree.types;


__device__ int longestCommonPrefix(const std::uint32_t* keys, int i, int j, int N)
{
    if (j<0 || j>=N)
        return -1;
    
    if (i==j)
        return 64;  //defensive check

    const std::uint32_t keyA = keys[i];
    const std::uint32_t keyB = keys[j];

    if (keyA!=keyB)     //se le chiavi sono uniche e sappiamo che i e j osno diversi posso evitare la condizione per accelerare computazione
        return __clz(keyA ^ keyB);
    else
        return 32 + __clz(static_cast<std::uint32_t>(i ^ j));   //equal morton codes disambigued using indices -> equivalent to appending the index bits to the morton code
}


/*l'algoritmo di karras dimostra che il nodo deve espandersi nella direzione in cui il prefisso è maggiore:
    - se le chiavi alla tua destra condividono più bit con la tua,allora appartengono allo stesso sottoalbero
    - se invece il prefisso più lungo è a sinistra, allora il sottoalbero è da quella parte*/
__device__ int determineDirection(const std::uint32_t* keys, int i, int N)
{
    int deltaRight = longestCommonPrefix (keys, i, i+1, N);
    int deltaLeft = longestCommonPrefix (keys, i, i-1, N);

    return (deltaRight > deltaLeft) ? +1 : -1;
}

__device__ void determineRange(const std::uint32_t* keys, int i, int N, int& first, int& last)    //use reference to avoid temporary struct
{
    int direction = determineDirection(keys, i, N);
    int deltaMin = longestCommonPrefix(keys, i, i-direction, N);

    //exponential search
    int length = 2; //the immediate neigbour is known to belong in the range, so distance 2 is the first upper-bound candidate
    while (longestCommonPrefix(keys, i, i+length*direction, N)>deltaMin)
    {
        length *= 2; 
    }

    //binary search to find the exact point 
    int l = 0;
    for (int t = length/2; t>=1; t/=2)
    {
        if(longestCommonPrefix(keys, i, i+(l+t)*direction, N)>deltaMin)
            l += t;
    }

    int j = i + l*direction;

    first = (i < j) ? i : j;
    last = (i > j) ? i : j;
}

__device__ int findSplit(const std::uint32_t* keys, int first, int last, int N)
{
    if (first >= last)
        return first;

    const int nodePrefix = longestCommonPrefix(keys, first, last, N);   //keys are ordered, no need to check each one of them
    
    int split = first;  //initial position -> i assume the left node contains at least the first key
    int step = last - first;    //maximum possible distance

    do 
    {
        step = (step + 1) >> 1; //dimezzamento con arrotondamento verso l'alto -> equals to step = (step+1)/2
        const int candidate = split + step; //avanzo di step

        if (candidate < last && longestCommonPrefix(keys, first, candidate, N) > nodePrefix)    //check the limit
            split = candidate;  //if candidate shares with the first key a longer prefix than the whole node, then it's still in the left node -> we can move forward int the keys to find the split
    } while (step > 1);

    return split;
}

//CUDDDA kernel
__global__ void buildTreeKernel(Tree tree, const std::uint32_t* keys, int N)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= N-1)
        return;

    int first;
    int last;

    determineRange(keys, i, N, first, last);    

    const int split = findSplit(keys, first, last, N);
    const int leftChild = (split == first) ? split : N + split;
    const int rightChild = (split+1 == last) ? split+1 : N + split+1;
    tree.left[i] = leftChild;   
    tree.right[i] = rightChild;

    const int currentNode = N + i;

    tree.parent[leftChild] = currentNode;
    tree.parent[rightChild] = currentNode;
}

//memory allocation
void allocateTree(Tree& tree, int N)
{
    if (tree.left != nullptr || tree.right != nullptr || tree.parent != nullptr)
        throw std::logic_error("Tree memory is already allocated");

    if (N <= 0)
        throw std::invalid_argument("N must be positive");

    const std::size_t internalNodeCount = static_cast<std::size_t>(N - 1);

    const std::size_t totalNodeCount = static_cast<std::size_t>(N) * 2 - 1;

    cudaError_t error;

    if (N > 1)
    {
        error = cudaMalloc(reinterpret_cast<void**>(&tree.left), internalNodeCount * sizeof(int));

        if (error != cudaSuccess)
            throw std::runtime_error(std::string("cudaMalloc tree.left failed: ") + cudaGetErrorString(error));

        error = cudaMalloc(reinterpret_cast<void**>(&tree.right), internalNodeCount * sizeof(int));

        if (error != cudaSuccess)
        {
            cudaFree(tree.left);
            tree.left = nullptr;

            throw std::runtime_error(std::string("cudaMalloc tree.right failed: ") + cudaGetErrorString(error));
        }
    }

    error = cudaMalloc(reinterpret_cast<void**>(&tree.parent), totalNodeCount * sizeof(int));

    if (error != cudaSuccess)
    {
        cudaFree(tree.left);
        cudaFree(tree.right);

        tree.left = nullptr;
        tree.right = nullptr;

        throw std::runtime_error(std::string("cudaMalloc tree.parent failed: ") + cudaGetErrorString(error));
    }

    tree.nBodies = N;
    tree.nInternalNodes = N - 1;
}

//free memory
void freeTree(Tree& tree)
{
    cudaFree(tree.left);
    cudaFree(tree.right);
    cudaFree(tree.parent);

    cudaFree(tree.mass);
    cudaFree(tree.comX);
    cudaFree(tree.comY);
    cudaFree(tree.comZ);
    cudaFree(tree.size);

    tree.left = nullptr;
    tree.right = nullptr;
    tree.parent = nullptr;

    tree.mass = nullptr;
    tree.comX = nullptr;
    tree.comY = nullptr;
    tree.comZ = nullptr;
    tree.size = nullptr;

    tree.nBodies = 0;
    tree.nInternalNodes = 0;
}

//host function
void buildTree (Tree& tree, const std::uint32_t* d_keys, int N)
{
    if (N <= 0)
        throw std::invalid_argument("buildTree requires N >= 1");

    if (tree.nBodies != N)
        throw std::invalid_argument("buildTree N does not match allocated tree size");

    if (tree.parent == nullptr)
        throw std::logic_error("Tree memory has not been allocated");

    if (N > 1 && (tree.left == nullptr || tree.right == nullptr))
    {
        throw std::logic_error("Tree child arrays have not been allocated");
    }

    if (d_keys == nullptr)
        throw std::invalid_argument("d_keys must not be null");
    
    const std::size_t totalNodes = static_cast<std::size_t>(2 * N - 1);

    cudaError_t error = cudaMemset(tree.parent, 0xFF, totalNodes*sizeof(int));

    if (error!=cudaSuccess)
        throw std::runtime_error(std::string("cudaMemset(tree.parent): ") + cudaGetErrorString(error));

    if (N==1)
        return;
    
    constexpr int threadsPerBlock = 256;

    const int blocks = (N-1 + threadsPerBlock -1)/threadsPerBlock;
    buildTreeKernel<<<blocks, threadsPerBlock>>>(tree, d_keys, N);
    error = cudaGetLastError();

    if (error!=cudaSuccess)
        throw std::runtime_error(std::string("buildTreeKernel launch: ") + cudaGetErrorString(error));

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
        throw std::runtime_error(std::string("buildTreeKernel execution: ") + cudaGetErrorString(error));
}