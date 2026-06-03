# Referencja Flux.jl (KM2)

Przykładowe rozwiązanie we **Flux.jl** — to ono definiuje architekturę sieci, algorytm uczenia
(SGD `Descent`, eta=1e-2) i hiperparametry (batch=10, 3 epoki), które nasza biblioteka `flex.jl`
odwzorowuje. Wymagane przez polecenie KM2 jako jedno z dwóch porównań (drugie: PyTorch w `../pytorch`).

Źródło: gist `bchaber/4736db461e8d60dd70bde57e51f5bc24` (podlinkowany w `materialy/polecenie.txt`).

## Uruchomienie

```bash
julia -e 'using Pkg; Pkg.add(["Flux","MLDatasets"])'   # raz
julia reference/flux/flux_reference.jl
```

Skrypt:
- trenuje tę samą sieć przez 3 epoki (`Descent`, eta=1e-2, batch=10, seed=1),
- mierzy czas/epokę, alokacje, GC, test accuracy (z rozgrzewką kompilacji jak w `bench/benchmark.jl`),
- dopisuje wiersz `flux-reference` do wspólnego `bench/results.csv`, więc porównanie z `flex.jl`
  jest w jednej tabeli.

## Uczciwa interpretacja porównania

- **Architektura, optymalizator i hiperparametry identyczne** — nie zmieniać.
- **Różnica metodyczna:** Flux przetwarza całe batche tensorów (`W×H×C×batch`) przez wektoryzowane
  jądra i zoptymalizowane BLAS/NNlib, podczas gdy `flex.jl` przetwarza pojedyncze obrazy z
  akumulacją gradientu (dozwolone przez FAQ polecenia). To naturalna przewaga Fluxa w czasie —
  w artykule warto to jawnie zaznaczyć przy omawianiu wyników.
- Porównuj na **tej samej maszynie** co pomiary `flex.jl`.
