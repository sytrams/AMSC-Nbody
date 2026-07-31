# Ordinamento Morton con `cub::DeviceRadixSort::SortPairs`

Questo documento contiene:

1. i punti 2, 3 e 4 della risposta passo-passo sull'uso di `SortPairs`;
2. l'intera spiegazione su dove inserire il codice nei file del progetto.

---

# Parte 1 — Punti 2, 3 e 4

## 2. Inizializzare gli indici prima del radix sort

Prima di chiamare `cub::DeviceRadixSort::SortPairs`, devi avere:

```cpp
d_indicesIn = [0, 1, 2, ..., N - 1]
```

Puoi inizializzarli con un kernel CUDA:

```cpp
__global__ void initializeIndicesKernel(
    std::uint32_t* indices,
    int N
)
{
    const int i =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= N)
        return;

    indices[i] = static_cast<std::uint32_t>(i);
}
```

Chiamata host:

```cpp
constexpr int threadsPerBlock = 256;
const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

initializeIndicesKernel<<<blocks, threadsPerBlock>>>(
    d_indicesIn,
    N
);

cudaError_t error = cudaGetLastError();

if (error != cudaSuccess)
{
    throw std::runtime_error(
        std::string("initializeIndicesKernel launch failed: ")
        + cudaGetErrorString(error)
    );
}
```

Puoi anche inizializzare gli indici sul lato host e copiarli, ma il kernel evita una copia CPU-GPU inutile.

---

## 3. Allocare input e output per indici e chiavi

Per `SortPairs` servono due buffer per le chiavi e due per gli indici:

```cpp
std::uint32_t* d_keysIn = nullptr;
std::uint32_t* d_keysOut = nullptr;

std::uint32_t* d_indicesIn = nullptr;
std::uint32_t* d_indicesOut = nullptr;
```

Allocazione:

```cpp
const std::size_t keysBytes =
    static_cast<std::size_t>(N) * sizeof(std::uint32_t);

const std::size_t indicesBytes =
    static_cast<std::size_t>(N) * sizeof(std::uint32_t);

cudaError_t error;

error = cudaMalloc(
    reinterpret_cast<void**>(&d_keysIn),
    keysBytes
);

if (error != cudaSuccess)
{
    throw std::runtime_error(
        std::string("cudaMalloc d_keysIn failed: ")
        + cudaGetErrorString(error)
    );
}

error = cudaMalloc(
    reinterpret_cast<void**>(&d_keysOut),
    keysBytes
);

if (error != cudaSuccess)
{
    cudaFree(d_keysIn);

    throw std::runtime_error(
        std::string("cudaMalloc d_keysOut failed: ")
        + cudaGetErrorString(error)
    );
}

error = cudaMalloc(
    reinterpret_cast<void**>(&d_indicesIn),
    indicesBytes
);

if (error != cudaSuccess)
{
    cudaFree(d_keysIn);
    cudaFree(d_keysOut);

    throw std::runtime_error(
        std::string("cudaMalloc d_indicesIn failed: ")
        + cudaGetErrorString(error)
    );
}

error = cudaMalloc(
    reinterpret_cast<void**>(&d_indicesOut),
    indicesBytes
);

if (error != cudaSuccess)
{
    cudaFree(d_keysIn);
    cudaFree(d_keysOut);
    cudaFree(d_indicesIn);

    throw std::runtime_error(
        std::string("cudaMalloc d_indicesOut failed: ")
        + cudaGetErrorString(error)
    );
}
```

`d_keysIn` deve contenere le chiavi Morton non ordinate.

---

## 4. Ordinare le coppie chiave-indice

La chiamata CUB deve essere concettualmente questa:

```cpp
cub::DeviceRadixSort::SortPairs(
    d_tempStorage,
    tempStorageBytes,
    d_keysIn,
    d_keysOut,
    d_indicesIn,
    d_indicesOut,
    N
);
```

Prima devi chiedere a CUB la quantità di memoria temporanea necessaria:

```cpp
void* d_tempStorage = nullptr;
std::size_t tempStorageBytes = 0;

cudaError_t error =
    cub::DeviceRadixSort::SortPairs(
        d_tempStorage,
        tempStorageBytes,
        d_keysIn,
        d_keysOut,
        d_indicesIn,
        d_indicesOut,
        N
    );

if (error != cudaSuccess)
{
    throw std::runtime_error(
        std::string("CUB temporary storage query failed: ")
        + cudaGetErrorString(error)
    );
}
```

Alloca il buffer temporaneo:

```cpp
error = cudaMalloc(
    &d_tempStorage,
    tempStorageBytes
);

if (error != cudaSuccess)
{
    throw std::runtime_error(
        std::string("cudaMalloc radix-sort storage failed: ")
        + cudaGetErrorString(error)
    );
}
```

Esegui il sort:

```cpp
error =
    cub::DeviceRadixSort::SortPairs(
        d_tempStorage,
        tempStorageBytes,
        d_keysIn,
        d_keysOut,
        d_indicesIn,
        d_indicesOut,
        N
    );

if (error != cudaSuccess)
{
    cudaFree(d_tempStorage);

    throw std::runtime_error(
        std::string("cub::DeviceRadixSort::SortPairs failed: ")
        + cudaGetErrorString(error)
    );
}
```

Dopo il sort:

```cpp
cudaFree(d_tempStorage);
```

Il risultato è:

```text
d_keysOut[i]       = i-esima chiave Morton ordinata
d_indicesOut[i]    = indice originale della sua particella
```

Per chiarezza, puoi usare alias:

```cpp
const std::uint32_t* d_sortedKeys = d_keysOut;
const std::uint32_t* d_sortedIndices = d_indicesOut;
```

---

# Parte 2 — Dove inserire il codice nei file

I punti 2, 3 e 4 **non vanno inseriti in `tree_builder.cu`**. Quella parte del codice deve occuparsi dell'albero dopo che le chiavi sono già state ordinate.

Devono andare nel file in cui attualmente:

1. calcoli le chiavi Morton;
2. chiami `cub::DeviceRadixSort::SortPairs(...)`;
3. ottieni gli array ordinati;
4. chiami `buildTree(...)`.

Correzione terminologica: in questo caso sono **frammenti di codice o funzioni**, non script.

## Organizzazione corretta

Il flusso dovrebbe essere separato così:

```text
Calcolo chiavi Morton
        ↓
Inizializzazione indici [0, 1, ..., N-1]
        ↓
SortPairs(keys, indices)
        ↓
d_sortedKeys + d_sortedIndices
        ↓
buildTree(d_sortedKeys)
        ↓
computeCentersOfMass(d_sortedIndices, dati particelle)
```

I tuoi file attuali dovrebbero avere questi ruoli:

```text
tree_types.hpp
    definizione della struttura Tree

tree_builder.hpp
    dichiarazioni delle funzioni dell'albero

tree_builder.cu
    costruzione dell'albero e calcolo bottom-up

file che contiene SortPairs
    calcolo chiavi, allocazione buffer e ordinamento
```

---

## Punto 2: `initializeIndicesKernel`

Essendo un kernel CUDA dichiarato con `__global__`, deve essere scritto in un file `.cu`.

La posizione migliore è nello stesso file in cui hai già il codice per il Morton sort, ad esempio:

```text
morton_sort.cu
```

oppure:

```text
particle_sort.cu
```

oppure nel tuo attuale file `.cu` dove chiami `cub::DeviceRadixSort::SortPairs`.

Inseriscilo nella parte iniziale del file, dopo gli `#include` e prima delle funzioni host:

```cpp
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <stdexcept>
#include <string>

__global__ void initializeIndicesKernel(
    std::uint32_t* indices,
    int N
)
{
    const int i =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= N)
        return;

    indices[i] = static_cast<std::uint32_t>(i);
}
```

La struttura del file deve quindi essere simile a:

```cpp
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdint>

// Kernel 1
__global__ void computeMortonKeysKernel(...)
{
    // ...
}

// Kernel 2
__global__ void initializeIndicesKernel(
    std::uint32_t* indices,
    int N
)
{
    const int i =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= N)
        return;

    indices[i] = static_cast<std::uint32_t>(i);
}

// Funzioni host
void sortParticlesByMortonCode(...)
{
    // ...
}
```

Non metterlo in:

```text
tree_builder.hpp
tree_types.hpp
```

e neppure in un normale `.cpp` compilato con `g++`, perché contiene sintassi CUDA.

---

## Punto 3: allocazione degli array

Le allocazioni:

```cpp
d_keysIn
d_keysOut
d_indicesIn
d_indicesOut
```

devono essere eseguite nella funzione host che gestisce l'ordinamento.

Probabilmente hai già una funzione simile a:

```cpp
void sortMortonKeys(...)
```

oppure il codice si trova direttamente nel tuo `main`.

La soluzione migliore è creare una funzione dedicata:

```cpp
void sortMortonPairs(
    const std::uint32_t* d_unsortedKeys,
    std::uint32_t* d_sortedKeys,
    std::uint32_t* d_sortedIndices,
    int N
);
```

In questo caso:

- `d_unsortedKeys` contiene le chiavi prodotte dal kernel Morton;
- `d_sortedKeys` riceve le chiavi ordinate;
- `d_sortedIndices` riceve la permutazione;
- `d_indicesIn` è un buffer temporaneo interno.

Esempio completo:

```cpp
void sortMortonPairs(
    const std::uint32_t* d_unsortedKeys,
    std::uint32_t* d_sortedKeys,
    std::uint32_t* d_sortedIndices,
    int N
)
{
    if (N <= 0)
        throw std::invalid_argument("N must be positive");

    if (
        d_unsortedKeys == nullptr ||
        d_sortedKeys == nullptr ||
        d_sortedIndices == nullptr
    )
    {
        throw std::invalid_argument(
            "Sort input and output pointers must not be null"
        );
    }

    std::uint32_t* d_indicesIn = nullptr;

    const std::size_t bytes =
        static_cast<std::size_t>(N) * sizeof(std::uint32_t);

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&d_indicesIn),
        bytes
    );

    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("cudaMalloc d_indicesIn failed: ")
            + cudaGetErrorString(error)
        );
    }

    // I punti 2 e 4 continueranno qui.
}
```

Nota importante: non devi necessariamente allocare `d_keysIn` dentro questa funzione se hai già un array contenente le chiavi Morton non ordinate.

Per esempio, se hai già:

```cpp
std::uint32_t* d_mortonKeys;
```

quello è già il tuo `d_keysIn`.

Allo stesso modo, se allochi gli output in una struttura esterna, non devi riallocarli ogni volta.

---

## Punto 4: chiamata a `SortPairs`

La chiamata a `SortPairs` va nella stessa funzione host, subito dopo:

1. l'allocazione di `d_indicesIn`;
2. l'inizializzazione di `d_indicesIn`.

La funzione completa assume questa forma:

```cpp
void sortMortonPairs(
    const std::uint32_t* d_unsortedKeys,
    std::uint32_t* d_sortedKeys,
    std::uint32_t* d_sortedIndices,
    int N
)
{
    if (N <= 0)
        throw std::invalid_argument("N must be positive");

    if (
        d_unsortedKeys == nullptr ||
        d_sortedKeys == nullptr ||
        d_sortedIndices == nullptr
    )
    {
        throw std::invalid_argument(
            "Sort input and output pointers must not be null"
        );
    }

    const std::size_t bytes =
        static_cast<std::size_t>(N) * sizeof(std::uint32_t);

    std::uint32_t* d_indicesIn = nullptr;

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&d_indicesIn),
        bytes
    );

    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("cudaMalloc d_indicesIn failed: ")
            + cudaGetErrorString(error)
        );
    }

    constexpr int threadsPerBlock = 256;

    const int blocks =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    /*
     * Crea:
     *
     * d_indicesIn = [0, 1, 2, ..., N - 1]
     */
    initializeIndicesKernel<<<blocks, threadsPerBlock>>>(
        d_indicesIn,
        N
    );

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        cudaFree(d_indicesIn);

        throw std::runtime_error(
            std::string(
                "initializeIndicesKernel launch failed: "
            ) + cudaGetErrorString(error)
        );
    }

    /*
     * Prima chiamata: CUB restituisce la quantità
     * di memoria temporanea necessaria.
     */
    void* d_tempStorage = nullptr;
    std::size_t tempStorageBytes = 0;

    error = cub::DeviceRadixSort::SortPairs(
        d_tempStorage,
        tempStorageBytes,
        d_unsortedKeys,
        d_sortedKeys,
        d_indicesIn,
        d_sortedIndices,
        N
    );

    if (error != cudaSuccess)
    {
        cudaFree(d_indicesIn);

        throw std::runtime_error(
            std::string("SortPairs storage query failed: ")
            + cudaGetErrorString(error)
        );
    }

    error = cudaMalloc(
        &d_tempStorage,
        tempStorageBytes
    );

    if (error != cudaSuccess)
    {
        cudaFree(d_indicesIn);

        throw std::runtime_error(
            std::string(
                "cudaMalloc radix-sort storage failed: "
            ) + cudaGetErrorString(error)
        );
    }

    /*
     * Seconda chiamata: esegue realmente il sort.
     */
    error = cub::DeviceRadixSort::SortPairs(
        d_tempStorage,
        tempStorageBytes,
        d_unsortedKeys,
        d_sortedKeys,
        d_indicesIn,
        d_sortedIndices,
        N
    );

    if (error != cudaSuccess)
    {
        cudaFree(d_tempStorage);
        cudaFree(d_indicesIn);

        throw std::runtime_error(
            std::string("CUB SortPairs failed: ")
            + cudaGetErrorString(error)
        );
    }

    error = cudaDeviceSynchronize();

    cudaFree(d_tempStorage);
    cudaFree(d_indicesIn);

    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("CUB SortPairs execution failed: ")
            + cudaGetErrorString(error)
        );
    }
}
```

---

## Dove allocare `d_sortedKeys` e `d_sortedIndices`

Questi due array devono sopravvivere alla funzione di ordinamento, perché verranno usati successivamente da:

```cpp
buildTree(...)
```

e:

```cpp
computeCentersOfMass(...)
```

Quindi non devono essere allocati e liberati internamente a `sortMortonPairs`.

Devi allocarli nel codice che controlla l'intera pipeline, per esempio nel `main`, in una classe GPU o in una funzione più generale:

```cpp
std::uint32_t* d_unsortedKeys = nullptr;
std::uint32_t* d_sortedKeys = nullptr;
std::uint32_t* d_sortedIndices = nullptr;

const std::size_t bytes =
    static_cast<std::size_t>(N) * sizeof(std::uint32_t);

cudaMalloc(
    reinterpret_cast<void**>(&d_unsortedKeys),
    bytes
);

cudaMalloc(
    reinterpret_cast<void**>(&d_sortedKeys),
    bytes
);

cudaMalloc(
    reinterpret_cast<void**>(&d_sortedIndices),
    bytes
);
```

Poi:

```cpp
computeMortonKeys(
    d_unsortedKeys,
    d_positionX,
    d_positionY,
    d_positionZ,
    N
);

sortMortonPairs(
    d_unsortedKeys,
    d_sortedKeys,
    d_sortedIndices,
    N
);

allocateTree(tree, N);

buildTree(
    tree,
    d_sortedKeys,
    N
);

computeCentersOfMass(
    tree,
    d_sortedIndices,
    d_mass,
    d_positionX,
    d_positionY,
    d_positionZ,
    N
);
```

Infine, quando non servono più:

```cpp
cudaFree(d_unsortedKeys);
cudaFree(d_sortedKeys);
cudaFree(d_sortedIndices);

freeTree(tree);
```

---

# Struttura dei file consigliata

Dato che hai già separato l'albero, la struttura pulita sarebbe:

```text
src/
├── tree_builder.hpp
├── tree_types.hpp
├── morton_sort.hpp
│
└── gpu/
    └── cuda_kernels/
        ├── tree_builder.cu
        └── morton_sort.cu
```

## `morton_sort.hpp`

```cpp
#pragma once

#include <cstdint>

void sortMortonPairs(
    const std::uint32_t* d_unsortedKeys,
    std::uint32_t* d_sortedKeys,
    std::uint32_t* d_sortedIndices,
    int N
);
```

## `morton_sort.cu`

Qui vanno:

```cpp
initializeIndicesKernel(...)
sortMortonPairs(...)
cub::DeviceRadixSort::SortPairs(...)
```

## `tree_builder.cu`

Qui restano:

```cpp
longestCommonPrefix(...)
determineDirection(...)
determineRange(...)
findSplit(...)
buildTreeKernel(...)
allocateTree(...)
buildTree(...)
computeCentersOfMass(...)
freeTree(...)
```

---

# Nel tuo caso concreto

Nei tre file che hai allegato:

- `tree_builder(5).cu`: **non inserire i punti 2, 3 e 4**;
- `tree_builder(1).hpp`: **non inserire i punti 2, 3 e 4**;
- `tree_types(1).hpp`: **non inserire i punti 2, 3 e 4**.

Devi inserirli nel file `.cu` in cui hai già scritto la tua chiamata:

```cpp
cub::DeviceRadixSort::SortPairs(...)
```

Se al momento quella chiamata si trova nel `main.cu`, puoi metterli lì temporaneamente. La soluzione corretta a lungo termine è spostarli in un file dedicato come `morton_sort.cu`.
