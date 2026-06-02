# julia-flux

Biblioteka do automatycznego różniczkowania (akumulacja w tył, graf obliczeniowy)
w języku Julia oraz konwolucyjna sieć neuronowa ucząca się klasyfikacji FashionMNIST.

- `flex.jl` — biblioteka (autodiff + warstwy: Conv, MaxPool, Dropout, Dense, SoftMax/logit-CE).
- `AWID-2026-CNN-flex.ipynb` — KM1: pełny przykład uczenia (~86% test acc, 3 epoki).

## KM2 — workflow optymalizacji

Cel KM2: zoptymalizować bibliotekę (złożoność obliczeniowa, pamięć, efektywność kodu)
**bez zmiany** struktury sieci ani parametrów uczenia, i opisać optymalizacje w artykule IEEE.
Żeby pisanie artykułu było łatwe, każdą optymalizację dokumentujemy w ustalony sposób.

**Zasada: jedna optymalizacja = jeden PR = jedna podsekcja artykułu.**

1. **Baseline jest zamrożony** w tagu `km1-baseline` (commit `e406b58`). To stały punkt
   odniesienia dla wszystkich liczb „przed".
2. **Gałąź integracyjna `km2`.** Każda optymalizacja powstaje na własnym feature-branchu
   (`opt/<nazwa>`) i trafia do `km2` przez PR.
3. **Każdy PR** wypełnia szablon (`.github/PULL_REQUEST_TEMPLATE.md`): motywacja, opis zmiany,
   kategoria (`opt:memory` / `opt:compute` / `opt:codegen`), tabela przed/po, weryfikacja.
4. **Pomiar i weryfikacja** — ten sam harness dla każdej zmiany:
   ```bash
   julia bench/benchmark.jl --gradcheck-only            # poprawność gradientów (roznice skonczone)
   julia bench/benchmark.jl --mode quick --label nazwa  # szybki pomiar do PR (1 epoka / podzbiór)
   julia bench/benchmark.jl --mode full  --label nazwa  # finalny pomiar do artykułu (3 epoki)
   ```
   Wynik dopisuje się automatycznie do `bench/results.csv` (z hashem commita).
5. **Rejestr** `OPTIMIZATIONS.md` — jeden wiersz na zmergowaną optymalizację. To jedyne
   źródło prawdy przy pisaniu artykułu (tabele i wykresy budujemy z `results.csv`).
6. **Review we dwóch.** Drugi członek zespołu recenzuje i akceptuje PR — to wymusza
   wzajemne zrozumienie kodu (wymóg samodzielności z polecenia).
7. **Referencja** `reference/pytorch/` — ta sama sieć w PyTorch do porównania.

### Pierwsze kroki

```bash
# zmierz baseline na SWOJEJ maszynie (liczby z notebooka są z innego sprzętu)
git checkout km1-baseline
julia bench/benchmark.jl --mode full --label baseline

# zacznij pierwszą optymalizację
git checkout km2
git checkout -b opt/typed-graphnode
```
