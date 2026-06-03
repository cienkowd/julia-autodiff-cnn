# Porównanie z rozwiązaniami referencyjnymi (KM2)

Wyniki testów porównawczych wymaganych przez polecenie: nasza biblioteka `flex.jl` (wersja KM2,
po optymalizacjach) vs przykładowe rozwiązanie we **Flux.jl** vs dodatkowa referencja w **PyTorch**.
Surowe liczby: [`results.csv`](results.csv). Ta sama architektura, `Descent`/SGD eta=1e-2, batch=10,
3 epoki, seed=1, CPU, ta sama maszyna.

| Rozwiązanie            | Framework               | Czas/epokę [s] | Razem [s] | Alokacje [GiB] | Test acc [%] |
|------------------------|-------------------------|---------------:|----------:|---------------:|-------------:|
| **flex.jl (KM2)**      | Julia / własna biblioteka | 13.90        | 41.69     | 52.2           | 87.44        |
| Flux.jl (przykładowe)  | Julia / Flux.jl         | 7.63           | 22.88     | 25.6           | 86.94        |
| PyTorch (dodatkowe)    | Python / PyTorch        | 7.74           | 23.23     | n/d¹           | 87.53        |

¹ Pomiar pamięci wg polecenia dotyczy tylko rozwiązań w Julii.

## Interpretacja (do artykułu)

- **Dokładność:** wszystkie trzy rozwiązania w tym samym wąskim przedziale (86.9–87.5%), powyżej
  wymaganego progu 85% — potwierdza poprawność odwzorowania architektury, algorytmu i hiperparametrów.
- **Czas:** oba dojrzałe frameworki (Flux i PyTorch) są niemal identyczne (~7.6–7.7 s/epokę) i ~1.8×
  szybsze od naszej biblioteki (13.9 s). To spodziewane: oba przetwarzają całe **batche**
  (`W×H×C×batch`) przez wektoryzowane jądra (NNlib / ATen), podczas gdy `flex.jl` przetwarza
  **pojedyncze obrazy** z akumulacją gradientu (dozwolone przez FAQ). Mimo to nasza biblioteka po
  optymalizacjach jest w tym samym rzędzie wielkości i bije limit 15 min/3 epoki z ogromnym zapasem
  (42 s). Zgodność Fluxa i PyTorcha co do czasu i dokładności uwiarygadnia metodykę pomiaru.
- **Pamięć:** `flex.jl` alokuje ~2× więcej niż Flux — naturalny kierunek dalszej pracy to batching
  na poziomie sieci, wykraczający poza zakres KM2.
- **Wniosek:** różnica ~1.8× między single-image a batched to nie jest narzut „własnej biblioteki",
  tylko koszt metodyki single-image; po optymalizacjach (KM1 → KM2: 6.8× szybciej, −85% alokacji)
  graf interpretowany w Julii dorównuje rzędem wielkości frameworkom produkcyjnym.
