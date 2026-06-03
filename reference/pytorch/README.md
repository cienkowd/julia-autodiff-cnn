# Referencja PyTorch (KM2)

Rozwiązanie referencyjne wymagane przez polecenie KM2 — ta sama sieć i te same parametry
co w bibliotece Julia, służy do porównania (czas, dokładność).

## Uruchomienie

macOS (Python z CommandLineTools nie ma `pip` na PATH — używaj `python3 -m pip`; najlepiej w venv):

```bash
cd reference/pytorch
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 train.py            # 3 epoki, batch=10, lr=1e-2, CPU; dopisze wiersz do ../results.csv
deactivate
```

## Założenia porównania

- **Ta sama struktura sieci** i te same hiperparametry (SGD, lr=1e-2, batch=10, 3 epoki) — nie zmieniać.
- **CPU**, bo biblioteka Julia działa na CPU — inaczej porównanie czasu jest nieuczciwe.
- Inicjalizacja He (`he_init`) odwzorowuje skalowanie `sqrt(2/fan_in)` z `flex.jl`.
- Równoważność batcha: wersja Julia przetwarza pojedyncze obrazy i aktualizuje wagi co
  `batchsize` przykładów ze średnim gradientem (`eta/s`), co odpowiada mini-batch SGD
  z `batch_size=10` i `CrossEntropyLoss` (redukcja `mean`) w PyTorch.

Wyniki wpisz do `../../OPTIMIZATIONS.md` lub osobnej tabeli porównawczej w artykule.
