# Rejestr optymalizacji (KM2)

Pojedyncze źródło prawdy dla artykułu IEEE. Każdy wiersz tabeli = jedna zmergowana
optymalizacja = jedna podsekcja artykułu. Punkt odniesienia: tag **`km1-baseline`**
(commit `e406b58`).

Liczby pochodzą z `bench/benchmark.jl --mode full`, uruchamianego na **tej samej maszynie
i z tym samym seedem**. Porównuj zawsze względem własnego pomiaru baseline (sprzęt się różni —
liczby z notebooka poniżej są poglądowe).

## Tabela zbiorcza

Wszystkie liczby zmierzone na **tej samej maszynie, seed=1, 3 epoki** (`--mode full`).

| #  | Optymalizacja        | Kategoria | PR  | Czas/epokę [s] | Alokacje [GiB] | Test acc [%] | Gradcheck |
|----|----------------------|-----------|-----|---------------:|---------------:|-------------:|-----------|
| 0  | baseline KM1         | —         | —   | 94.93          | 341.1          | 87.44        | OK        |
| 1  | cache im2col w Conv  | pamięć    | #1  | 94.37          | 325.7          | 87.44        | OK        |
| 2  | operacje in-place    | pamięć    | #2  | 95.41          | 215.8          | 87.44        | OK        |
|    | **Δ #2 względem #1** |          |     | +1.1% (szum)   | **−109.9 GiB (−33.7%)** | =   | OK        |
|    | **Δ łącznie vs KM1** |          |     | ≈0             | **−125.3 GiB (−36.7%)** | =   | OK        |

Dla porównania, baseline z notebooka KM1 (inna maszyna): 163.8 s/epokę, 330.6 GiB, 86.37% —
poglądowy, niemiarodajny do porównań czasu.

## Szczegóły

### 0. baseline KM1
Stan z tagu `km1-baseline`, zmierzony na tej samej maszynie: 94.93 s/epokę, 341.1 GiB, 87.44%.
Charakterystyka kodu bazowego (cele optymalizacji): graf interpretowany w pętli z dynamicznym
dispatchem, operacje alokujące (`y.data = ...`, `grad += ...`), `Dict{Any,Any}` na akumulację
gradientów, im2col liczone dwukrotnie (forward i backward), nietypowane pola `GraphNode` (boxing).

### 1. Cache im2col w warstwie Conv
- **Kategoria:** złożoność obliczeniowa + pamięć
- **PR:** #1
- **Problem:** `adjoint!(::GraphNode{:conv,2})` (flex.jl) liczył ponownie `pad2d`,
  `im2col` oraz `permutedims(W.data)` — dokładnie te same wartości, które policzył
  już `primal!`. Wejście i wagi nie zmieniają się między forward a backward jednej
  próbki (aktualizacja wag jest dopiero w `optimize!`), więc to czysta redundancja:
  podwójny koszt obliczeń i alokacji w najcięższej warstwie sieci.
- **Zmiana:** `primal!` zapamiętuje `col` i `W_mat` w polu `meta` węzła (przez `Ref`,
  analogicznie do maski w Dropout); `adjoint!` je odczytuje zamiast liczyć od nowa.
- **Wynik (ta sama maszyna, seed=1, 3 epoki):**
  - czas/epokę: 94.93 → 94.37 s (−0.6%, w granicach szumu)
  - alokacje: 341.1 → 325.7 GiB (**−15.4 GiB, −4.5%**)
  - test acc: 87.44% → 87.44% (identyczne — bo math bez zmian)
- **Weryfikacja:** `--gradcheck-only` → **OK** (max błąd względny 2.9e-3, tol 2e-2).
  Accuracy i loss identyczne co do 4 miejsc po przecinku (87.44%, 0.3464) — najmocniejszy dowód,
  że optymalizacja nie zmienia obliczeń.
- **Wniosek:** to optymalizacja **pamięciowa**, nie czasowa — rekompilacja im2col alokowała,
  ale kosztowała mało CPU względem mnożeń macierzy, więc czas się nie ruszył. Główne źródło
  alokacji jest gdzie indziej (drobne `+=`/`*` w `adjoint!`) → cel kolejnych PR-ów.
- **Do artykułu:** wzorzec „nie licz w backwardzie tego, co masz z forwardu"; im2col wg W06.
  Koszt: bufor `col`/`W_mat` trzymany do backward (kompromis pamięć vs obliczenia — tu zysk po
  obu stronach, bo i tak alokowaliśmy te tablice dwukrotnie). Ważna lekcja metodologiczna:
  porównanie z notebookiem (inna maszyna) sugerowało −42% czasu — pomiar na tej samej maszynie
  pokazał, że to artefakt sprzętu, a realny zysk jest w pamięci.

### 2. Operacje in-place (eliminacja tymczasowych alokacji)
- **Kategoria:** gospodarka pamięcią
- **PR:** #2
- **Problem:** `primal!`/`adjoint!` w gorącej ścieżce sieci alokowały tablice na każdy
  przykład: `=`/`+=` rebindowały pola (`y.data = W*x`, `grad += ...`), a `maxpool` i `dropout`
  kopiowały wycinki/maski. Przy ~60k przykładów × 3 epoki to główne źródło ~325 GiB.
- **Zmiana (bez zmiany matematyki):**
  - `:mul` — `mul!(y.data, W, x)`; `W.grad .+= y.grad .* x'` (broadcast); `mul!(x.grad, W', y.grad, true, true)`
  - `:add`, `:flatten`, `:logit_ce`, conv-grad — `=`/`+=` → `.=`/`.+=` (akumulacja in-place)
  - `:logit_ce` primal — `sum(v->exp(v-m), ...)` zamiast tymczasowej tablicy `exp.(...)`
  - `:maxpool` — `@view` zamiast kopiowania patcha (forward i backward)
  - `:dropout` — prealokowany bufor maski + `rand!` (zamiast `rand(...)` co forward)
- **Wynik (ta sama maszyna, seed=1, 3 epoki):**
  - czas/epokę: 94.37 → 95.41 s (+1.1%, w granicach szumu)
  - alokacje: 325.7 → 215.8 GiB (**−109.9 GiB, −33.7%**); GC 4.82% → 3.28%
  - test acc: 87.44% (identyczne, gradcheck OK)
- **Weryfikacja:** `--gradcheck-only` → **OK** (2.9e-3); acc/loss identyczne co do 4 miejsc.
- **Wniosek (ważny do artykułu):** presja pamięci spadła o 1/3, ale **wall-clock bez zmian** —
  czas jest zdominowany przez obliczenia (pętle im2col + małe mnożenia BLAS na pojedynczych
  obrazach), nie przez GC (które i tak było tylko ~5%). Wniosek: dalsze przyspieszenie wymaga
  ataku na *obliczenia* (batching, redukcja narzutu dispatchu, lepszy im2col), nie na alokacje.
- **Do artykułu:** klasyczna gospodarka pamięcią w Julii (W06) — `mul!`/`@.`/`@view`/`rand!`
  zamiast alokujących operatorów. Dobra ilustracja, że redukcja alokacji ≠ redukcja czasu,
  gdy obciążenie nie jest GC-bound.

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
