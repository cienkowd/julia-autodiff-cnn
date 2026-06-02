## Optymalizacja: <krótka nazwa>

<!-- Zasada: jeden PR = jedna optymalizacja = jedna podsekcja artykułu.
     Wypełnij wszystkie sekcje — to one wprost trafią do artykułu IEEE. -->

### Kategoria
<!-- zaznacz jedną; ustaw też odpowiednią etykietę PR -->
- [ ] Złożoność obliczeniowa (`opt:compute`)
- [ ] Gospodarka pamięcią (`opt:memory`)
- [ ] Efektywność wygenerowanego kodu (`opt:codegen`)

### Motywacja / problem
<!-- Co w kodzie było nieefektywne i dlaczego. Wskaż miejsce (plik:linia).
     Jeśli masz profil/alokacje — wklej liczby. -->

### Co zmieniono
<!-- Zwięzły opis techniczny. -->

### Wyniki (`--mode full`, ten sam seed co baseline)
| Metryka            | Przed | Po | Zmiana |
|--------------------|------:|---:|-------:|
| Czas / epokę [s]   |       |    |        |
| Alokacje [GiB]     |       |    |        |
| GC [%]             |       |    |        |
| Test accuracy [%]  |       |    |        |

> `julia bench/benchmark.jl --mode full --label "<label>"`

### Weryfikacja poprawności
- [ ] `julia bench/benchmark.jl --gradcheck-only` → gradienty **OK**
- [ ] Test accuracy w granicach tolerancji (≥ 80%, cel ~85%)
- [ ] Czas 3 epok < 15 min
- [ ] Wiersz dopisany do `bench/results.csv`
- [ ] Wpis dodany do `OPTIMIZATIONS.md`

### Notatki do artykułu
<!-- Kompromisy, dlaczego akurat tak, odniesienia do wykładów (np. W05 codegen, W06 im2col). -->

### Review
<!-- Dwuosobowy zespół: drugi członek musi zrozumieć i zaakceptować zmianę
     (wymóg samodzielności z polecenia). -->
- [ ] Zrecenzowane i zrozumiane przez drugiego członka zespołu
