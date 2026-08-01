#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "tree_builder.hpp"

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

//CUDA kernel
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

__global__ void initializeLeavesKernel(Tree tree, const std::uint32_t* sortedIndices, const float* particleMass, const float* positionX, const float* positionY, const float* positionZ, int N)
{
    const int leaf = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (leaf >= N)
        return;
    
    //leaf is the position in Morton order, particleIndex is the index in the original order
    const std::uint32_t particleIndex = sortedIndices[leaf];

    tree.mass[leaf] = particleMass[particleIndex];
    tree.comX[leaf] = positionX[particleIndex];
    tree.comY[leaf] = positionY[particleIndex];
    tree.comZ[leaf] = positionZ[particleIndex];
}

__global__ void computeCentersOfMassKernel(Tree tree, int N)
{
    const int leaf = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (leaf >= N)
        return;

    int completedNode = leaf;
    int parentNode = tree.parent[completedNode];

    while (parentNode != -1)    //parentNode is global, left, right e visitCount use the local indices
    {
        const int parentIndex = parentNode - N;

        //mass and CoM of the subtree are now visible before signalling completion
        __threadfence();

        const int oldCount = atomicAdd(&tree.visitCount[parentIndex],1);    //doesn't this slow down computation????
        
        //the first child can't compute the parent
        if (oldCount == 0)
            return;
        //the second child finds both subtrees ready
        const int leftChild = tree.left[parentIndex];
        const int rightChild = tree.right[parentIndex];
        const float leftMass = tree.mass[leftChild];
        const float rightMass = tree.mass[rightChild];
        const float totalMass = leftMass + rightMass;
        tree.mass[parentNode] = totalMass;

        if (totalMass > 0.0f)       
        {
            const float inverseMass = 1.0f / totalMass; //inverse of the mass to multiply directly fo all the dimensions

            tree.comX[parentNode] = (leftMass * tree.comX[leftChild] + rightMass * tree.comX[rightChild]) * inverseMass;

            tree.comY[parentNode] = (leftMass * tree.comY[leftChild] + rightMass * tree.comY[rightChild]) * inverseMass;

            tree.comZ[parentNode] = (leftMass * tree.comZ[leftChild] + rightMass * tree.comZ[rightChild]) * inverseMass;
        }
        else
        {
            tree.comX[parentNode] = 0.0f;
            tree.comY[parentNode] = 0.0f;
            tree.comZ[parentNode] = 0.0f;
        }

        completedNode = parentNode;
        parentNode = tree.parent[completedNode];
    }
}

//memory allocation
void allocateTree(Tree& tree, int N)
{
    if (tree.left != nullptr || tree.right != nullptr || tree.parent != nullptr || tree.visitCount != nullptr || tree.mass != nullptr || tree.comX != nullptr || tree.comY != nullptr || tree.comZ != nullptr || tree.size != nullptr)
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

        error = cudaMalloc(reinterpret_cast<void**>(&tree.visitCount), internalNodeCount * sizeof(int));

        if (error != cudaSuccess)
        {
            freeTree(tree);

            throw std::runtime_error(std::string("cudaMalloc tree.visitCount failed: ") + cudaGetErrorString(error));
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

    const std::size_t physicalBytes = totalNodeCount * sizeof(float);

    error = cudaMalloc(reinterpret_cast<void**>(&tree.mass), physicalBytes);

    if (error != cudaSuccess) 
    {
        freeTree(tree);
        throw std::runtime_error(std::string("cudaMalloc tree.mass failed: ") + cudaGetErrorString(error));
    }

    error = cudaMalloc(reinterpret_cast<void**>(&tree.comX), physicalBytes);

    if (error != cudaSuccess)
    {
        freeTree(tree);

        throw std::runtime_error(std::string("cudaMalloc tree.comX failed: ") + cudaGetErrorString(error));
    }

    error = cudaMalloc(reinterpret_cast<void**>(&tree.comY), physicalBytes);

    if (error != cudaSuccess)
    {
        freeTree(tree);

        throw std::runtime_error(std::string("cudaMalloc tree.comY failed: ") + cudaGetErrorString(error));
    }

    error = cudaMalloc(reinterpret_cast<void**>(&tree.comZ), physicalBytes);

    if (error != cudaSuccess)
    {
        freeTree(tree);

        throw std::runtime_error(std::string("cudaMalloc tree.comZ failed: ") + cudaGetErrorString(error));
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

    cudaFree(tree.visitCount);

    tree.left = nullptr;
    tree.right = nullptr;
    tree.parent = nullptr;

    tree.mass = nullptr;
    tree.comX = nullptr;
    tree.comY = nullptr;
    tree.comZ = nullptr;
    tree.size = nullptr;

    tree.visitCount = nullptr;

    tree.nBodies = 0;
    tree.nInternalNodes = 0;
}

//host function
void buildTree (Tree& tree, const std::uint32_t* d_sortedKeys, int N)
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

    if (d_sortedKeys == nullptr)
        throw std::invalid_argument("d_sortedKeys must not be null");
    
    const std::size_t totalNodes = static_cast<std::size_t>(2 * N - 1);

    cudaError_t error = cudaMemset(tree.parent, 0xFF, totalNodes*sizeof(int));

    if (error!=cudaSuccess)
        throw std::runtime_error(std::string("cudaMemset(tree.parent): ") + cudaGetErrorString(error));

    if (N==1)
        return;
    
    constexpr int threadsPerBlock = 256;

    const int blocks = (N-1 + threadsPerBlock -1)/threadsPerBlock;
    buildTreeKernel<<<blocks, threadsPerBlock>>>(tree, d_sortedKeys, N);
    error = cudaGetLastError();

    if (error!=cudaSuccess)
        throw std::runtime_error(std::string("buildTreeKernel launch: ") + cudaGetErrorString(error));

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
        throw std::runtime_error(std::string("buildTreeKernel execution: ") + cudaGetErrorString(error));
}

void computeCenterOfMass (Tree& tree, const std::uint32_t* d_sortedIndices, const float* d_mass, const float* d_positionX, const float* d_positionY, const float* d_positionZ, int N)
{
    if (N <= 0)
        throw std::invalid_argument("computeCentersOfMass requires N >= 1");

    if (tree.nBodies != N)
        throw std::invalid_argument("computeCentersOfMass N does not match tree size");

    if (d_sortedIndices == nullptr)
        throw std::invalid_argument("d_sortedIndices must not be null");

    if (d_mass == nullptr || d_positionX == nullptr || d_positionY == nullptr || d_positionZ == nullptr)
        throw std::invalid_argument("Particle data arrays must not be null");

    if (tree.parent == nullptr || tree.mass == nullptr || tree.comX == nullptr || tree.comY == nullptr || tree.comZ == nullptr)
        throw std::logic_error("Tree memory has not been fully allocated");

    if (N > 1 && (tree.left == nullptr || tree.right == nullptr || tree.visitCount == nullptr))
        throw std::logic_error("Tree internal-node memory has not been allocated");

    constexpr int threadsPerBlock = 256;
    const int blocks = (N +  threadsPerBlock - 1)/threadsPerBlock;

    //leaves recieve particle data floowing SortPairs permutation
    initializeLeavesKernel<<<blocks, threadsPerBlock>>>(tree, d_sortedIndices, d_mass, d_positionX, d_positionY, d_positionZ, N);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
        throw std::runtime_error(std::string("initializeLeavesKernel launch failed: ") + cudaGetErrorString(error));

    if (N == 1) //single  leaf is also the radix
    {
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
            throw std::runtime_error(std::string("initializeLeavesKernel execution failed: ") + cudaGetErrorString(error));
        return;
    }

    error  = cudaMemset(tree.visitCount, 0, static_cast<std::size_t>(N-1) * sizeof(int));

    if (error != cudaSuccess)
        throw std::runtime_error(std::string("cudaMemset tree.visitCount failed: ") + cudaGetErrorString(error));

    computeCentersOfMassKernel<<<blocks, threadsPerBlock>>>(tree, N);

    error = cudaGetLastError();

    if (error != cudaSuccess)
        throw std::runtime_error(std::string("computeCentersOfMassKernel launch failed: ") + cudaGetErrorString(error));

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
        throw std::runtime_error(std::string("computeCenterOfMassKernel execution failed: ") + cudaGetErrorString(error));
}