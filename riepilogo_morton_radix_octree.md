# Riepilogo del lavoro svolto: Morton keys, binary radix tree e raggruppamento delle chiavi duplicate

## Punto di partenza

Siamo partiti da una pipeline già funzionante per costruire un **binary radix tree** sulla GPU:

```text
posizioni delle particelle
        ↓
normalizzazione nel bounding box
        ↓
chiavi di Morton
        ↓
CUB DeviceRadixSort::SortPairs
        ↓
chiavi ordinate + permutation array
        ↓
binary radix tree di Karras
        ↓
massa e centro di massa bottom-up
```

Il codice principale era distribuito in:

```text
src/tree_types.hpp
src/tree_builder.hpp
gpu/cuda_kernels/tree_builder.cu
test_tree/
```

L’ordinamento con:

```cpp
cub::DeviceRadixSort::SortPairs(...)
```

produce due array:

```text
sortedKeys
sortedIndices
```

Esempio:

```text
chiavi originali:      12   3   7   3
indici originali:       0   1   2   3

dopo SortPairs:

sortedKeys:             3   3   7  12
sortedIndices:          1   3   2   0
```

`sortedIndices` è fondamentale perché permette di sapere quale particella originale corrisponde a ogni posizione Morton ordinata.

---

## 1. Binary radix tree iniziale

Il primo albero costruito era un **binary radix tree**, non ancora un octree.

Per `N` particelle, il layout era:

```text
foglie:          [0, N)
nodi interni:    [N, 2N - 1)
nodi totali:     2N - 1
```

Ogni nodo interno possiede:

```cpp
left
right
parent
```

e viene costruito in parallelo da un thread CUDA.

La logica deriva dal prefisso comune delle chiavi Morton:

```cpp
longestCommonPrefix(keys, i, j, N)
```

Ogni thread:

1. determina la direzione del proprio intervallo;
2. trova l’intervallo di foglie `[first, last]`;
3. individua lo split;
4. collega il figlio sinistro;
5. collega il figlio destro;
6. imposta i parent.

Il radix tree è utile perché può essere costruito completamente in parallelo, senza inserimenti sequenziali dalla radice.

---

## 2. Ordinamento e associazione con le particelle

Un problema importante era mantenere corretta l’associazione tra chiavi ordinate e dati fisici.

Non bastava ordinare soltanto:

```cpp
d_keys
```

perché massa e posizione erano ancora memorizzate secondo l’ordine originale delle particelle.

Abbiamo quindi usato `SortPairs`:

```text
keys    → ordinati
indices → riordinati nello stesso modo
```

Durante l’inizializzazione delle foglie:

```cpp
const std::uint32_t particleIndex =
    sortedIndices[leaf];
```

e quindi:

```cpp
tree.mass[leaf] = particleMass[particleIndex];
tree.comX[leaf] = positionX[particleIndex];
tree.comY[leaf] = positionY[particleIndex];
tree.comZ[leaf] = positionZ[particleIndex];
```

In questo modo la foglia `leaf` rappresentava la particella collocata in quella posizione dell’ordinamento Morton.

---

## 3. Computazione bottom-up di massa e centro di massa

Dopo aver costruito la topologia, abbiamo implementato la riduzione bottom-up.

Ogni foglia inizializza:

```text
massa
posizione, usata come centro di massa
```

Poi ogni foglia risale verso la radice.

Per ogni nodo interno, i due figli devono terminare prima che il genitore possa essere calcolato. Abbiamo usato:

```cpp
atomicAdd(&tree.visitCount[parentIndex], 1);
```

Il primo figlio che arriva:

```text
incrementa il contatore
termina
```

Il secondo figlio:

```text
trova entrambi i figli pronti
calcola massa e CoM del genitore
continua verso l’alto
```

Le formule sono:

\[
M_p = M_l + M_r
\]

\[
\mathbf{x}_p =
\frac{M_l\mathbf{x}_l + M_r\mathbf{x}_r}
     {M_l + M_r}
\]

Abbiamo verificato:

- masse delle foglie;
- centri di massa;
- massa della radice;
- CoM della radice;
- nodi interni;
- sincronizzazione;
- assenza di race;
- assenza di memory error;
- assenza di leak.

---

## 4. Aggiunta dei metadati del radix tree

Il radix tree iniziale costruiva correttamente la topologia, ma scartava alcune informazioni calcolate durante la costruzione.

Abbiamo quindi aggiunto:

```cpp
int* rangeFirst;
int* rangeLast;
int* prefixLength;
```

Per ogni nodo interno:

```text
rangeFirst[i]   = prima foglia Morton del sottoalbero
rangeLast[i]    = ultima foglia Morton del sottoalbero
prefixLength[i] = numero di bit comuni tra le chiavi estreme
```

Esempio:

```text
nodo interno i
range = [3, 7]
```

significa che il nodo rappresenta tutte le foglie Morton:

```text
3, 4, 5, 6, 7
```

Il prefisso:

```cpp
prefixLength[i]
```

descrive invece quanto è specifica la regione Morton rappresentata dal nodo.

Questi dati erano necessari perché la futura conversione radix → octree si basa precisamente sui prefissi delle chiavi.

---

## 5. Perché le chiavi Morton duplicate erano inizialmente accettabili

Nel binary radix tree originale potevano esserci chiavi duplicate:

```text
7 7 7 7
```

Un radix tree richiede però che gli elementi siano distinguibili.

Abbiamo quindi usato la tecnica prevista nell’algoritmo di Karras: quando due chiavi sono uguali, si usa l’indice ordinato come suffisso virtuale.

Concettualmente:

```text
chiave Morton || indice
```

Esempio:

```text
7 || 0
7 || 1
7 || 2
7 || 3
```

Nel codice:

```cpp
if (keyA != keyB)
{
    return __clz(keyA ^ keyB);
}
else
{
    return 32 +
           __clz(
               static_cast<std::uint32_t>(i ^ j));
}
```

Questo permette di costruire un binary radix tree valido anche se le chiavi Morton sono identiche.

Quindi, per il **binary radix tree da solo**, il raggruppamento delle chiavi duplicate non era strettamente necessario.

Il tree builder iniziale era corretto anche con duplicati.

---

## 6. Perché il raggruppamento è diventato necessario per l’octree

Questo è il punto fondamentale.

I bit aggiunti tramite l’indice:

```text
Morton key || sorted index
```

servono soltanto a distinguere elementi uguali nella struttura radix.

Non rappresentano coordinate spaziali.

In un octree, invece, ogni gruppo di tre bit Morton rappresenta una scelta di ottante:

```text
bit X
bit Y
bit Z
```

Un prefisso di:

```text
3 bit   → livello octree 1
6 bit   → livello octree 2
9 bit   → livello octree 3
...
```

I bit dell’indice virtuale non possono essere interpretati come livelli dell’octree, perché non derivano dalla posizione della particella.

Se usassimo direttamente il radix tree con duplicati per costruire l’octree, potremmo interpretare erroneamente:

```text
bit dell’indice
```

come:

```text
suddivisioni spaziali aggiuntive
```

Il risultato sarebbe un octree geometricamente falso.

Per questo abbiamo raggruppato tutte le particelle con la stessa chiave Morton.

---

## 7. Cosa significa raggruppare le chiavi duplicate

Partiamo da:

```text
sortedKeys:
7 7 7 12 18 18 25
```

Abbiamo costruito:

```text
uniqueKeys:
7 12 18 25

firstParticle:
0 3 4 6

particleCount:
3 1 2 1
```

Il gruppo `0` rappresenta:

```text
chiave Morton = 7
prima posizione ordinata = 0
numero particelle = 3
```

Quindi contiene:

```cpp
sortedIndices[0]
sortedIndices[1]
sortedIndices[2]
```

Il gruppo `2` rappresenta:

```text
chiave Morton = 18
prima posizione ordinata = 4
numero particelle = 2
```

e contiene:

```cpp
sortedIndices[4]
sortedIndices[5]
```

La struttura introdotta è:

```cpp
struct MortonLeafGroups
{
    int nParticles;
    int nGroups;
    int capacity;

    std::uint32_t* uniqueKeys;
    int* firstParticle;
    int* particleCount;

    int* numberOfGroupsDevice;

    void* temporaryStorage;
    std::size_t temporaryStorageBytes;
};
```

---

## 8. Come abbiamo implementato il raggruppamento

Abbiamo usato due primitive CUB.

### Run-length encoding

```cpp
cub::DeviceRunLengthEncode::Encode(...)
```

Trasforma:

```text
7 7 7 12 18 18 25
```

in:

```text
uniqueKeys:
7 12 18 25

particleCount:
3 1 2 1
```

Funziona perché le chiavi sono già ordinate.

Se le chiavi non fossero ordinate:

```text
7 12 7
```

verrebbero considerate tre run separate. Per questo il grouping deve avvenire dopo `SortPairs`.

### Exclusive scan

```cpp
cub::DeviceScan::ExclusiveSum(...)
```

Trasforma:

```text
particleCount:
3 1 2 1
```

in:

```text
firstParticle:
0 3 4 6
```

La relazione è:

```cpp
firstParticle[g + 1] =
    firstParticle[g] +
    particleCount[g];
```

---

## 9. Perché abbiamo costruito un nuovo radix tree sulle chiavi uniche

Dopo il grouping, abbiamo:

```text
nParticles = numero totale di particelle
nGroups    = numero di chiavi Morton uniche
```

Esempio:

```text
nParticles = 7
nGroups    = 4
```

Il radix tree destinato alla conversione in octree deve quindi essere costruito su:

```cpp
groups.uniqueKeys
```

con:

```cpp
groups.nGroups
```

La chiamata corretta è:

```cpp
allocateTree(
    tree,
    groups.nGroups);

buildTree(
    tree,
    groups.uniqueKeys,
    groups.nGroups);
```

oppure tramite il wrapper:

```cpp
buildTreeFromMortonGroups(
    tree,
    groups);
```

Il nuovo albero avrà:

```text
foglie:          nGroups
nodi interni:    nGroups - 1
nodi totali:     2 * nGroups - 1
```

Nell’esempio:

```text
4 foglie
3 nodi interni
7 nodi totali
```

invece di:

```text
7 foglie
6 nodi interni
13 nodi totali
```

Questo non è solamente un risparmio di memoria: è soprattutto la rappresentazione spaziale corretta.

Ogni foglia radix ora rappresenta:

```text
una cella Morton occupata
```

non:

```text
una singola particella arbitrariamente distinta da un indice virtuale
```

---

## 10. Modifica del significato delle foglie

Prima:

```text
una foglia radix = una particella
```

Ora:

```text
una foglia radix =
una chiave Morton unica =
una cella occupata =
una o più particelle
```

Per questo abbiamo rinominato:

```cpp
tree.nBodies
```

in:

```cpp
tree.nLeaves
```

`nBodies` era ormai semanticamente falso.

Può essere:

```text
nParticles = 1000
nLeaves = 700
```

se alcune particelle condividono la stessa chiave Morton.

---

## 11. Centro di massa delle foglie raggruppate

Con una particella per foglia era sufficiente:

```cpp
tree.mass[leaf] = particleMass[particle];
tree.comX[leaf] = positionX[particle];
```

Con più particelle per foglia dobbiamo prima aggregare il gruppo.

Abbiamo introdotto:

```cpp
initializeGroupedLeavesKernel(...)
```

Ogni thread gestisce un gruppo Morton.

Per il gruppo `g`:

```cpp
const int first =
    groups.firstParticle[g];

const int count =
    groups.particleCount[g];
```

Poi itera sulle particelle del gruppo:

```cpp
for (int local = 0;
     local < count;
     ++local)
{
    const int sortedPosition =
        first + local;

    const std::uint32_t particleIndex =
        sortedIndices[sortedPosition];

    // accumulo massa e posizione pesata
}
```

La massa della foglia è:

\[
M_g = \sum_{i \in g}m_i
\]

Il centro di massa:

\[
\mathbf{x}_g =
\frac{\sum_{i \in g}m_i\mathbf{x}_i}
     {M_g}
\]

Dopo questa inizializzazione, il precedente algoritmo bottom-up può essere riutilizzato senza modifiche sostanziali.

Il nodo interno non deve sapere se il figlio rappresenta:

- una singola particella;
- tre particelle;
- cento particelle.

Vede soltanto:

```text
massa del figlio
centro di massa del figlio
```

---

## 12. Perché manteniamo ancora il vecchio calcolo CoM

Ora esistono concettualmente due modalità.

### Vecchia modalità

```cpp
computeCenterOfMass(...)
```

Assume:

```text
una foglia = una particella
```

Serve ancora per:

- mantenere i test originali;
- verificare che il vecchio comportamento non sia stato rotto;
- avere una reference semplice;
- controllare la riduzione bottom-up.

### Nuova modalità

```cpp
computeGroupedCenterOfMass(...)
```

Assume:

```text
una foglia = un gruppo Morton
```

Questa è la funzione coerente con la futura pipeline octree.

---

## 13. Test che abbiamo aggiunto

Abbiamo mantenuto i test originali:

```text
test_tree/
```

che verificano:

- topologia radix;
- parent e children;
- range;
- prefissi;
- massa;
- CoM;
- chiavi duplicate;
- casi casuali.

Abbiamo aggiunto:

```text
test_morton_groups/
```

che verifica:

- singola chiave;
- tutte chiavi uniche;
- tutte duplicate;
- gruppi misti;
- chiavi minime e massime;
- capacità insufficiente;
- continuità degli offset;
- somma dei conteggi.

Abbiamo poi aggiunto:

```text
test_radix_groups/
```

che verifica l’intera pipeline:

```text
sorted keys
→ grouping
→ unique keys
→ radix tree
→ grouped leaf CoM
→ root CoM
```

Casi testati:

```text
una particella
tutte chiavi duplicate
gruppi misti
tutte chiavi uniche
caso generato con N = 257
```

Abbiamo verificato anche:

```text
memcheck: 0 errors
leak check: 0 bytes
racecheck: 0 hazards
```

---

## La necessità del grouping, in una frase

Il grouping **non era necessario per costruire un binary radix tree generico**, perché Karras permette di distinguere le chiavi duplicate usando l’indice come suffisso virtuale.

Era però **necessario per convertire il radix tree in un octree spazialmente corretto**, perché soltanto i bit reali delle chiavi Morton rappresentano suddivisioni geometriche dello spazio.

---

## Perché serve ancora il binary radix tree sulle chiavi uniche

Si potrebbe chiedere: una volta ottenute le chiavi uniche, perché non costruire direttamente l’octree?

Perché il binary radix tree fornisce in parallelo:

```text
relazioni gerarchiche tra prefissi;
intervalli di chiavi;
lunghezze dei prefissi;
parentela tra regioni Morton.
```

Questa struttura intermedia permette di enumerare i nodi dello sparse octree senza inserire sequenzialmente ogni chiave partendo dalla radice.

La pipeline corretta è quindi:

```text
chiavi Morton ordinate
        ↓
raggruppamento duplicati
        ↓
chiavi Morton uniche
        ↓
binary radix tree sui prefissi unici
        ↓
conversione parallela radix → sparse octree
```

Il radix tree non sarà necessariamente usato nel traversal Barnes–Hut finale. Serve come struttura efficiente per costruire l’octree.

---

## Stato attuale

Abbiamo completato:

```text
1. Morton key generation
2. SortPairs
3. permutation array
4. binary radix tree
5. radix range metadata
6. radix prefix metadata
7. bottom-up mass and CoM
8. duplicate-key grouping
9. unique Morton keys
10. radix tree sulle unique keys
11. grouped leaf mass and CoM
12. test funzionali e sanitizer
```

Il prossimo passo è:

```text
binary radix tree sulle chiavi uniche
                    ↓
         sparse octree
```

Dovremo determinare, per ogni arco del radix tree:

```text
quanti livelli octree attraversa;
quanti nodi octree genera;
quale prefisso spaziale rappresenta ogni nodo;
quale nodo è il parent;
in quale ottante collegare il figlio.
```

Poi useremo una prefix sum per assegnare gli indici dei nodi octree in parallelo.
