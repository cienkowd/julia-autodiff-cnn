# Rejestr optymalizacji (KM2)

Pojedyncze źródło prawdy dla artykułu IEEE. Każdy wiersz tabeli = jedna zmergowana
optymalizacja = jedna podsekcja artykułu. Punkt odniesienia: tag **`km1-baseline`**
(commit `e406b58`).

Liczby pochodzą z `bench/benchmark.jl --mode full`, uruchamianego na **tej samej maszynie
i z tym samym seedem**. Porównuj zawsze względem własnego pomiaru baseline (sprzęt się różni —
liczby z notebooka poniżej są poglądowe).

## Tabela zbiorcza

| #  | Optymalizacja        | Kategoria | PR  | Czas/epokę [s] | Alokacje [GiB] | Test acc [%] | Gradcheck |
|----|----------------------|-----------|-----|---------------:|---------------:|-------------:|-----------|
| 0  | baseline KM1         | —         | —   | 163.8\*        | 330.6\*        | 86.37        | OK        |

\* liczby z notebooka KM1 (inna maszyna). **Zmierz baseline u siebie** przed pierwszym
porównaniem: `git checkout km1-baseline && julia bench/benchmark.jl --mode full --label baseline`.

## Szczegóły

### 0. baseline KM1
Stan z tagu `km1-baseline`. Charakterystyka kodu bazowego (cele optymalizacji):
graf interpretowany w pętli z dynamicznym dispatchem, operacje alokujące (`y.data = ...`,
`grad += ...`), `Dict{Any,Any}` na akumulację gradientów, im2col liczone dwukrotnie
(forward i backward), nietypowane pola `GraphNode` (boxing). Wartości odniesienia w tabeli.

<!-- ───────────────────────────────────────────────────────────────────────
Szablon wpisu — KOPIUJ dla każdej optymalizacji i dopisz wiersz do tabeli wyżej:

### N. <nazwa>
- **Kategoria:** pamięć | złożoność | kod
- **PR:** #NN
- **Problem:** (plik:linia) co było nieefektywne
- **Zmiana:** co zrobiono
- **Wynik:** czas X→Y s/epokę, alokacje A→B GiB, acc utrzymane (C%)
- **Weryfikacja:** gradcheck OK; czas 3 epok = Z min
- **Do artykułu:** kompromisy / odniesienia (np. W05, W06)
──────────────────────────────────────────────────────────────────────── -->
