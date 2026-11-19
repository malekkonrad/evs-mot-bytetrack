# Podsumowanie artykułów — Multiple Object Tracking (MOT)

Dokument zawiera streszczenie czterech artykułów dotyczących śledzenia wielu obiektów oraz metryk ewaluacji. Przeznaczony jako materiał referencyjny do projektu polegającego na implementacji trackera MOT na predefiniowanym zbiorze detekcji, ewaluowanego metryką MOTA z użyciem narzędzia TrackEval.

---

## Kontekst projektu

- **Zadanie:** śledzenie wielu obiektów (MOT) w oparciu o **predefiniowany zbiór detekcji** (detektor jest dany — nie trzeba go implementować/trenować).
- **Wyzwania w MOT:**
  - nowy obiekt pojawia się na scenie (inicjalizacja),
  - false positive detektora,
  - false negative (obiekt jest, ale niewykryty),
  - okluzje i pomylenie tożsamości (identity switch).
- **Metryka główna:** MOTA = 1 − Σ(FN + FP + IDSW) / Σ g_t.
- **Narzędzie ewaluacji:** TrackEval (https://github.com/JonathonLuiten/TrackEval).



---

## 1. SORT — Simple Online and Realtime Tracking (Bewley et al., 2016)

**Filozofia:** minimalizm zgodny z brzytwą Ockhama. Tracking-by-detection bez modelu wyglądu, oparty wyłącznie na geometrii bounding boxów i ich ruchu.

### Komponenty
- **Filtr Kalmana** ze stałym modelem prędkości.
  - Wektor stanu: `x = [u, v, s, r, u̇, v̇, ṡ]`
    - `u, v` — współrzędne środka BB,
    - `s` — skala (pole),
    - `r` — aspect ratio (traktowane jako stałe — nie ma pochodnej),
    - reszta to pochodne czasowe.
  - Gdy detekcja zostaje przypisana do toru, BB aktualizuje stan przez KF.
  - Gdy detekcji brak, stan jest tylko predykowany (bez korekcji).
- **Asocjacja danych:**
  - Macierz kosztów: **IoU distance** między predykowanymi BB istniejących torów a nowymi detekcjami.
  - Rozwiązywanie przypisania: **algorytm węgierski** (Hungarian / Munkres).
  - Próg `IoU_min` — przypisania poniżej progu są odrzucane.
- **Tworzenie torów:** każda detekcja niedopasowana do żadnego toru (z IoU < `IoU_min`) inicjalizuje nowy tor; nowy tor przechodzi okres próbny (musi się powtórzyć w kolejnych klatkach), żeby uniknąć śledzenia FP.
- **Kasowanie torów:** tor usuwany po `T_Lost` klatkach bez detekcji. W oryginale `T_Lost = 1` — autorzy świadomie nie obsługują długich okluzji, bo model stałej prędkości i tak źle predykuje na dłuższych odcinkach.

### Wyniki i zalety
- 33.4 MOTA na MOT15.
- **260 Hz** na pojedynczym rdzeniu — najszybszy tracker swojej klasy.
- Demonstruje, że jakość detektora jest kluczowa: zmiana ACF → Faster R-CNN(VGG16) podnosi MOTA z 15.1 do 34.0.

### Wady
- Brak obsługi długich okluzji → wysoka liczba ID switchy.
- Brak modelu wyglądu (re-identyfikacji).
- Czuły na ruch kamery.

### Kod referencyjny
https://github.com/abewley/sort

---

## 2. ByteTrack — Multi-Object Tracking by Associating Every Detection Box (Zhang et al., 2022)

**Kluczowa obserwacja:** standardowe trackery odrzucają detekcje o score < τ (np. 0.5), ale niskoscoringowe detekcje często odpowiadają **realnym obiektom częściowo zasłoniętym**. Odrzucanie ich powoduje fragmentację torów i ID switche.

### Mechanizm BYTE — asocjacja dwustopniowa

Pseudokod (uproszczony):

```
Dla każdej klatki:
    1. Detekcje → podziel na D_high (score > τ) i D_low (score ≤ τ)
    2. Predykcja KF dla wszystkich istniejących torów T
    3. PIERWSZA ASOCJACJA:
       - macierz kosztów IoU (lub IoU + Re-ID) między D_high i T
       - Hungarian → dopasowane pary
       - pozostałości: D_remain, T_remain
    4. DRUGA ASOCJACJA:
       - macierz kosztów IoU między T_remain a D_low
       - tylko IoU (Re-ID jest zawodne dla niskoscoringowych detekcji)
       - Hungarian → dopasowanie zasłoniętych obiektów
       - niedopasowane tory → T_re_remain
    5. Niedopasowane D_low odrzuć jako tło
    6. Tory niedopasowane przez 30 klatek → usuń (track rebirth buffer)
    7. Niedopasowane D_remain z wysokim score → inicjalizuj nowe tory
```

### Detektor
- **YOLOX-X** (jednostopniowy, COCO-pretrained).

### Wyniki na MOT17 test
- **80.3 MOTA, 77.3 IDF1, 63.1 HOTA, 30 FPS** — 1. miejsce na leaderboardzie w momencie publikacji.

### Generalność
BYTE jest "wrapperem" — autorzy zastosowali go do 9 różnych SOTA-trackerów (FairMOT, CenterTrack, TransTrack, MOTR, JDE, CSTrack, TraDes, QDTrack, ChainedTracker) i w każdym przypadku poprawił MOTA i IDF1.
- Przykład: CenterTrack 66.1 → 67.4 MOTA, IDF1 64.2 → 74.0, IDS 528 → 144.

### Inne ważne ustalenia (ablacja)
- IoU jako Similarity#1 — lepsza MOTA i mniej IDS.
- Re-ID jako Similarity#1 — lepsze IDF1 (lepsza długoterminowa identyfikacja).
- W drugiej asocjacji **zawsze IoU** — Re-ID na niskoscoringowych detekcjach degraduje wyniki.
- BYTE jest robustny na próg detekcji (`τ` od 0.2 do 0.8) — druga asocjacja "łapie" obiekty niezależnie od progu.

### Kod referencyjny
https://github.com/ifzhang/ByteTrack

---

## 3. BoT-SORT — Robust Associations Multi-Pedestrian Tracking (Aharon et al., 2022)

**Filozofia:** rozszerzenie ByteTrack o trzy ulepszenia adresujące jego słabości. "Bag of tricks" dla MOT.

### Trzy główne ulepszenia

#### (a) Zmodyfikowany filtr Kalmana
- Stary stan (DeepSORT): `[x_c, y_c, a, h, ẋ, ẏ, ȧ, ḣ]` — gdzie `a` to aspect ratio.
- **Nowy stan (BoT-SORT):** `[x_c, y_c, w, h, ẋ, ẏ, ẇ, ḣ]` — bezpośrednie estymowanie szerokości i wysokości.
- Powód: estymacja aspect ratio jest mniej dokładna niż bezpośrednia estymacja `w` i `h`.
- Macierze szumu Q i R są zależne od czasu (skalowane przez aktualne `ŵ`, `ĥ`).
- Domyślne współczynniki: `σ_p = 0.05`, `σ_v = 0.00625`, `σ_m = 0.05` (dla 30 FPS).

#### (b) Camera Motion Compensation (CMC)
Rejestracja obrazu między sąsiednimi klatkami w celu skorygowania ruchu kamery (kluczowe dla scen z ruchomą kamerą i nawet dla MOT20 z drganiami statycznej kamery).

- **Algorytm:** Global Motion Compensation z OpenCV (Video Stabilization).
  - Wykrycie keypointów (Shi-Tomasi "Good Features to Track"),
  - Sparse optical flow (Lucas-Kanade) do trackingu cech,
  - Odrzucanie outlierów na podstawie translacji,
  - Estymacja **macierzy afinicznej A** przez RANSAC.
- **Korekcja stanu KF:** macierz A rozkłada się na część skalująco-rotacyjną M (2×2) i translacyjną T (2×1). Predykowany stan i kowariancja są transformowane:
  - `x'ₖ|ₖ₋₁ = M̃ · x̂ₖ|ₖ₋₁ + T̃`
  - `P'ₖ|ₖ₋₁ = M̃ · Pₖ|ₖ₋₁ · M̃ᵀ`

  gdzie `M̃` (8×8) i `T̃` (8×1) to rozszerzenia M i T do wymiaru wektora stanu.
- Wykorzystanie sparse registration pozwala ignorować dynamiczne obiekty (na podstawie detekcji) i lepiej estymować ruch tła.

#### (c) Fuzja IoU + Re-ID
- **Re-ID:** SBS-S50 z biblioteki FastReID, backbone ResNeSt50.
- **EMA cech wyglądu:** `eᵏᵢ = α · eᵏ⁻¹ᵢ + (1 − α) · fᵏᵢ`, z α = 0.9. Tylko detekcje o wysokim score aktualizują wektor cech (ochrona przed zaszumionymi cechami w okluzjach).
- **Nowa fuzja kosztów** (zamiast ważonej sumy `C = λA_a + (1-λ)A_m`):

  ```
  d̂_cos(i,j) = 0.5 · d_cos(i,j),  jeśli d_cos < θ_emb ORAZ d_iou < θ_iou
              = 1,                   w przeciwnym razie

  C(i,j) = min(d_iou(i,j), d̂_cos(i,j))
  ```

  - Progi: `θ_iou = 0.5`, `θ_emb = 0.25` (dobrane z histogramu pozytywnych vs negatywnych par).
  - Idea: minimum z dwóch kosztów + maskowanie odległych/niepodobnych par.

### Wyniki
- **MOT17 test:** 80.5 MOTA, 80.2 IDF1, 65.0 HOTA — 1. miejsce.
- **MOT20 test:** 77.8 MOTA, 77.5 IDF1, 63.3 HOTA — 1. miejsce.
- Wersja bez Re-ID (BoT-SORT) — 80.6 MOTA, 79.5 IDF1.
- FPS: BoT-SORT ~6.6, BoT-SORT-ReID ~4.5 (znacznie wolniejszy od ByteTrack ze względu na Re-ID i CMC).

### Ablacja (MOT17 val)
| Wersja                              | MOTA  | IDF1  | HOTA  |
|-------------------------------------|-------|-------|-------|
| Baseline (ByteTrack reimpl.)        | 77.66 | 79.77 | 67.88 |
| + zmodyfikowany KF                  | 77.67 | 79.89 | 68.12 |
| + CMC                               | 78.31 | 81.51 | 69.06 |
| + predykcja brakujących detekcji    | 78.39 | 81.53 | 69.11 |
| + Re-ID (BoT-SORT-ReID)             | 78.46 | 82.07 | 69.17 |

### Pomocnicze narzędzie diagnostyczne — cMOTA
Autorzy proponują **Current-MOTA** — MOTA jako funkcja klatki, `cMOTA(t) = MOTA liczone od t₀ do t`. Pozwala wizualnie wykryć moment, w którym tracker zaczyna zawodzić (np. w trakcie obrotu kamery — bez CMC krzywa cMOTA gwałtownie spada).

### Kod referencyjny
https://github.com/NirAharon/BOT-SORT

---

## 4. CLEAR MOT Metrics (Bernardin & Stiefelhagen, 2008)

Artykuł metodologiczny — definiuje metryki MOTP i MOTA, używane do dziś jako standard ewaluacji MOT (w tym przez MOTChallenge i TrackEval).

### Procedura ewaluacji (per klatka)

1. **Ustalenie korespondencji** między hipotezami trackera `{h₁, ..., h_m}` a obiektami z ground truth `{o₁, ..., o_n}`:
   - Najpierw zachowaj korespondencje z poprzedniej klatki, jeśli odległość < T (próg).
   - Dla pozostałych — rozwiąż przypisanie minimum-cost (Munkres / Hungarian) z ograniczeniem `dist < T`.
   - Jeśli nowe przypisanie zaprzecza poprzedniemu (`oᵢ` zmieniło hipotezę z `hⱼ` na `hₖ`) — zlicz to jako **mismatch (IDSW)**.
2. Pozostałe niedopasowane obiekty → **misses (FN)**.
3. Pozostałe niedopasowane hipotezy → **false positives (FP)**.
4. Dla każdej dopasowanej pary — zapisz błąd lokalizacji `dᵢ_t`.

### Metryki

#### Najważniejsze metryki to: MOTA, MOTP, IDSW (chyba najbardziej liczy się MOTA)


**MOTP (Multiple Object Tracking Precision):**
```
MOTP = Σ_(i,t) dᵢ_t / Σ_t cₜ
```
gdzie `cₜ` to liczba dopasowań w klatce t. Mierzy tylko **precyzję lokalizacji** dopasowanych par.

**MOTA (Multiple Object Tracking Accuracy):**
```
MOTA = 1 − Σ_t (mₜ + fpₜ + mmeₜ) / Σ_t gₜ
```
gdzie `mₜ` = liczba FN, `fpₜ` = liczba FP, `mmeₜ` = liczba IDSW, `gₜ` = liczba GT obiektów w klatce t.

### Ważne uwagi

- **Próg T** silnie wpływa na zachowanie metryki:
  - T → ∞ → korespondencje zawsze zachowane → ignorowanie błędów lokalizacji,
  - T → 0 → wszystko liczone jako miss/FP → metryki tracą sens.
  - Typowy wybór: 50 cm dla 3D, IoU = 0 (jakikolwiek overlap) lub IoU < 0.5 dla 2D.
- **IDSW liczone tylko w momencie zmiany przypisania**, nie w każdej klatce — to świadomy wybór projektowy zapobiegający niespójnym wynikom.
- **Sumowanie globalne, nie uśrednianie per-klatka:** najpierw zsumuj wszystkie błędy w całej sekwencji, potem podziel — w przeciwnym razie krótkie klatki o złych wynikach dominują nad długimi o dobrych.
- **MOTA może być ujemna** — jeśli FP + FN + IDSW > GT.

### Inne metryki w użyciu (kontekst)
- **IDF1** (Ristani et al., 2016) — ocenia ciągłość tożsamości.
- **HOTA** (Luiten et al., 2021) — łączy detekcję, asocjację i lokalizację w jednej liczbie. Bardziej zbalansowana niż MOTA.

---

## Porównanie metod — tabela syntetyczna

| Cecha                       | SORT          | ByteTrack         | BoT-SORT-ReID        |
|-----------------------------|---------------|-------------------|----------------------|
| Filtr Kalmana               | tak (`s, r`)  | tak (`a, h`)      | tak (`w, h`)         |
| Wykorzystanie low-score det.| nie           | **tak (BYTE)**    | tak (BYTE)           |
| Bufor lost tracks           | 1 klatka      | 30 klatek         | 30 klatek            |
| Re-ID                       | nie           | nie               | **tak (FastReID)**   |
| Camera Motion Compensation  | nie           | nie               | **tak (GMC + RANSAC)** |
| MOTA na MOT17 test          | ~33 (MOT15)   | 80.3              | 80.5                 |
| IDF1 na MOT17 test          | —             | 77.3              | **80.2**             |
| FPS                         | **260**       | 30                | 4.5                  |
| Złożoność implementacji     | niska         | niska/średnia     | wysoka               |

---

## Rekomendacja dla projektu (predefiniowane detekcje, ewaluacja MOTA, TrackEval)

Zalecane podejście **stopniowe** — buduj kolejne warstwy złożoności:

### Etap 1 — SORT (baseline)
- Filtr Kalmana ze stałą prędkością (wektor stanu jak w SORT).
- IoU jako miara podobieństwa.
- `scipy.optimize.linear_sum_assignment` jako Hungarian.
- Próg `IoU_min` ~ 0.3.
- Okres próbny dla nowych torów (3 klatki), `T_Lost = 1` lub niewielki bufor.
- **Cel:** działający tracker w kilkuset linijkach kodu, zrozumienie podstawowych mechanizmów.

### Etap 2 — ByteTrack-style (główne usprawnienie)
- Podział detekcji na high/low score (próg ~ 0.5 lub 0.6).
- Dwustopniowa asocjacja: pierwsza dla high-score, druga dla low-score na pozostałych torach.
- Zwiększenie bufora lost tracks do 30 klatek (track rebirth).
- **To najczęściej daje największy skok MOTA przy minimalnym koszcie implementacji.**

### Etap 3 — opcjonalne ulepszenia z BoT-SORT
- Zmiana wektora stanu KF na `[x, y, w, h, ...]` (bezpośrednie estymowanie wymiarów).
- CMC — jeśli sekwencje testowe zawierają ruch kamery (OpenCV: `cv2.estimateAffinePartial2D` + RANSAC).
- **Pomijanie Re-ID** — wymaga sieci neuronowej i trenowania, wykracza poza zakres "śledzenia na predefiniowanych detekcjach".


### Praktyczne wskazówki techniczne
- TrackEval oczekuje wyników w formacie MOTChallenge: `frame, id, x, y, w, h, conf, -1, -1, -1` (CSV per sekwencja).
- Dla filtra Kalmana wystarczy `filterpy.kalman.KalmanFilter` lub własna implementacja (8 wymiarów stanu, 4 wymiary obserwacji).
- IoU dla wektoryzowanej macierzy kosztów: warto zaimplementować jako operację numpy na całej macierzy `tracks × detections` jednocześnie (znacznie szybsze niż pętla).
- Próg IoU dla asocjacji ≠ próg IoU dla TP w ewaluacji (typowo 0.5 w MOTChallenge).

---

## Literatura — pełne odnośniki

1. Bewley, A., Ge, Z., Ott, L., Ramos, F., Upcroft, B. (2016). *Simple Online and Realtime Tracking*. ICIP.
2. Zhang, Y., Sun, P., Jiang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. arXiv:2110.06864.
3. Aharon, N., Orfaig, R., Bobrovsky, B.-Z. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*. arXiv:2206.14651.
4. Bernardin, K., Stiefelhagen, R. (2008). *Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics*. EURASIP J. on Image and Video Processing.