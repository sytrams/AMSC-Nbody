# Test del CUDA tree builder

Il test usa direttamente:

- `src/tree_types.hpp`
- `src/tree_builder.hpp`
- `gpu/cuda_kernels/tree_builder.cu`

## Esecuzione

Dalla root del progetto:

```bash
./test/run_tree_builder_test.sh
```

Per eseguire anche Racecheck:

```bash
RUN_RACECHECK=1 ./test/run_tree_builder_test.sh
```

Il sistema deve avere `nvcc`, una GPU NVIDIA utilizzabile e il CUDA runtime. Se disponibile, lo script esegue automaticamente anche `compute-sanitizer --tool memcheck`.

## Proprietà verificate

- numero corretto di foglie, nodi interni e nodi totali;
- un solo nodo radice, con ID globale `N`;
- figli validi e distinti per ogni nodo interno;
- coerenza bidirezionale tra `left`/`right` e `parent`;
- esattamente un genitore per ogni nodo non radice;
- assenza di cicli e nodi scollegati;
- copertura di tutte le foglie da parte della radice;
- intervalli contigui nell'ordine Morton per ogni sottoalbero;
- topologia esatta per casi semplici noti;
- gestione di chiavi Morton duplicate;
- associazione foglia-particella tramite `sortedIndices`;
- massa e centro di massa di ogni nodo interno;
- valore finale `visitCount == 2` per ogni nodo interno.
