# MOT 44.02 — jak działa to rozwiązanie

Opis odpowiada konfiguracji, która dała `44.02` na sprawdzarce. Pliki
`results/test/MOT_0{1,6,7}.txt` zostały wygenerowane komendą:

```bash
uv run python tracker/track_all.py \
  data/evs_mot_public_dataset/evs_mot-test \
  results/test \
  --cmc \
  --predict-lost-frames 8 \
  --track-thresh 0.6 \
  --new-track-thresh 0.7 \
  --match-thresh 0.9 \
  --track-buffer 60 \
  --interpolate \
  --max-gap 60
```

Rdzeń to ByteTrack (Zhang et al. 2022) z filtrem Kalmana, plus trzy
rozszerzenia: Camera Motion Compensation, predict-fill dla niedawno straconych
torów oraz liniowa interpolacja luk w post-processingu.

## 1. Co to za model

To nie jest model uczony end-to-end, tylko klasyczny pipeline MOT:

1. wejściem są gotowe detekcje z `det/det.txt`,
2. tracker przewiduje ruch obiektów filtrem Kalmana,
3. w każdej klatce dopasowuje detekcje do istniejących torów (dwustopniowo),
4. nadaje spójne `track_id`,
5. uzupełnia wyjście predykcjami Kalmana dla świeżo zgubionych torów,
6. po przelocie całej sekwencji wypełnia luki wewnątrz traków interpolacją,
7. zapisuje wynik w formacie MOTChallenge.

## 2. Wejście

Plik detekcji ma format:

```text
frame, -1, x, y, w, h, score
```

[run_tracker.py](../tracker/run_tracker.py)
wczytuje go i grupuje po numerze klatki:

- `xywh` — bounding boxy,
- `scores` — confidence detektora.

Dla każdej klatki wywoływane jest:

```python
tracker.update(xywh, scores)
```

## 3. Reprezentacja toru — filtr Kalmana

Każdy tor ma stan 8D z
[kalman_filter.py](../tracker/kalman_filter.py):

```text
[cx, cy, w, h, vx, vy, vw, vh]
```

czyli środek bboxa, szerokość/wysokość i ich prędkości.

- `initiate()` — start nowego toru,
- `predict()` — krok predykcji w następnej klatce,
- `update()` — korekcja stanu nową obserwacją (przez wzmocnienie Kalmana).

Szumy procesu i pomiaru są skalowane wysokością `h` (DeepSORT-style), więc
duże, bliskie obiekty mają większą tolerancję ruchu niż małe, odległe.

W `Track.predict` tory w stanie `Lost` mają wygaszane prędkości w/h (= 0)
oraz tłumioną prędkość środka (`lost_velocity_damping`), żeby predykcje bez
obserwacji się nie rozjeżdżały.

## 4. Krok pojedynczej klatki

Logika siedzi w
[byte_tracker.py](../tracker/byte_tracker.py)
(`ByteTracker.update`).

### 4.1. CMC — kompensacja ruchu kamery (`--cmc`)

Przed asocjacją wołane jest `apply_cmc(affine)` z transformacją affine 2×3
estymowaną przez
[cmc.py](../tracker/cmc.py):

1. Shi-Tomasi: keypointy w tle (z maską na obszary detekcji, żeby nie łapać
   ruchomych obiektów),
2. Lucas-Kanade: sparse optical flow na te punkty między klatkami,
3. RANSAC: globalna affine (obrót + skala + translacja).

Korekcja stanu KF dla wszystkich torów (active + lost):

```text
mean = M8 @ mean + T8
cov  = M8 @ cov @ M8.T
```

Bez CMC predykcje Kalmana po ruchu kamery rozsynchronizowują się z detekcjami
i sztucznie rośnie liczba ID switch.

### 4.2. Podział detekcji na high/low

```text
high: score >= track_thresh           (0.6)
low:  low_thresh <= score < track_thresh  (0.1..0.6)
```

`high` służą do głównego dopasowania, `low` tylko do ratowania istniejących
torów (nigdy nie tworzą nowych).

### 4.3. Podział torów

- `tracked` — aktywne, potwierdzone (`is_activated == True`),
- `unconfirmed` — świeżo założone w poprzedniej klatce, nie potwierdzone,
- `lost` — chwilowo zgubione, jeszcze w buforze (≤ `track_buffer = 60`).

Wszystkie aktywne i stracone tory dostają `Kalman.predict()` zanim ruszy
asocjacja.

### 4.4. Asocjacja #1 — `dets_high` vs (`tracked` + `lost`)

Algorytm węgierski
([matching.py](../tracker/matching.py))
na koszcie:

```text
cost = 1 - IoU
```

Z aktywnym `fuse_score`: `cost = 1 - IoU * score` — silniejsza detekcja z
lepszym IoU dostaje priorytet. Próg odcięcia par: `match_thresh = 0.9`.

Efekt:

- dopasowany `tracked` → `update()` (KF koryguje stan),
- dopasowany `lost` → `re_activate()` (z zachowaniem `track_id`),
- pozostałe tory idą do następnej rundy.

### 4.5. Asocjacja #2 — `dets_low` vs niedopasowane `tracked`

Tylko tory `tracked`, które przeżyły rundę #1, dostają drugą szansę przez
słabe detekcje. `lost` tu nie wchodzą — to chroni przed ich resurrekcją z
przypadkowych słabych boxów. Bez `fuse_score` (score jest zbyt zaszumiony),
próg `0.5`.

Tory, które przeżyły obie rundy bez pary → `mark_lost()`.

### 4.6. Asocjacja #3 — pozostałe `dets_high` vs `unconfirmed`

Świeżo narodzone tory potrzebują drugiego dopasowania, żeby się
"potwierdzić". Próg pary: `0.7`. Niedopasowane unconfirmed → `mark_removed()`
(prawdopodobnie były fałszywym alarmem detektora).

### 4.7. Inicjalizacja nowych torów

Niedopasowane `dets_high` mogą założyć tor tylko jeśli
`score >= new_track_thresh = 0.7`. Wyższy próg → mniej fałszywych ID.

### 4.8. Sprzątanie

- Tory `Lost` powyżej `max_time_lost = track_buffer = 60` klatek →
  `Removed`.
- `_remove_duplicate_tracks` usuwa duplikaty między `tracked` i `lost` przy
  IoU > 0.85, zachowując tor o dłuższej historii.

## 5. Predict-fill straconych torów (`--predict-lost-frames 8`)

Na końcu `update()`, do listy wyjściowej dorzucane są tory, które są w
`lost`, ale zostały zgubione tylko `1..8` klatek temu — z bboxem z bieżącej
predykcji Kalmana.

Logika: detektor pokrywa GT tylko ~30–50% per klatka, więc krótkie luki to
zwykle "uciekająca" detekcja, a nie zniknięcie obiektu. Emisja predykcji
zwiększa recall przy minimalnym koszcie precision (KF jest dokładny na
horyzontach rzędu kilku klatek).

## 6. Cykl życia toru

```text
new -> tracked -> lost -> removed
```

- `new` — założony w bieżącej klatce, jeszcze `unconfirmed`,
- `tracked` — potwierdzony (po ponownym dopasowaniu w następnej klatce),
- `lost` — brak dopasowania w obu rundach (chwilowo, jeszcze w buforze),
- `removed` — przekroczył `max_time_lost` lub był duplikatem.

## 7. Post-processing — interpolacja luk (`--interpolate --max-gap 60`)

Po wygenerowaniu pliku MOT,
[interpolate.py](../tracker/interpolate.py)
przelatuje wynik per `track_id`:

- sortuje wpisy po `frame`,
- dla par sąsiednich obserwacji `(frame_a, frame_b)` z luką
  `1 < gap <= max_gap`, dodaje liniowo interpolowane bbox dla klatek
  `frame_a+1 .. frame_b-1`.

Działa **wyłącznie wewnątrz** traku (są obserwacje po obu stronach), więc
generuje praktycznie czyste TP — standardowy idiom z ByteTrack/StrongSORT.

To drugi mechanizm zwiększający recall (po predict-fill), ale działa na luki
o długości do 60 klatek, podczas gdy predict-fill emituje tylko do 8 klatek
od ostatniej obserwacji w żywym `lost` torze.

## 8. Dlaczego te konkretne progi

Walidacja na zbiorze treningowym pokazała, że:

- `track_thresh=0.6 / new_track_thresh=0.7 / match_thresh=0.9` dają mniej FP
  niż domyślne `0.5/0.6/0.8` — detektor jest "hojny", więc agresywne progi
  pomagają.
- `track_buffer=60` (zamiast domyślnych 30) razem z CMC pozwala odzyskiwać
  tory po dłuższych okluzjach bez gubienia ID.
- `predict-lost-frames=8` i `max-gap=60` to dwa komplementarne sposoby
  uzupełniania detekcji, krótko- i długoterminowo.

## 9. Wyjście

Standardowy MOTChallenge:

```text
frame,id,x,y,w,h,1,-1,-1,-1
```

Tracker nie segmentuje ani nie klasyfikuje — dla każdej klatki podaje
gdzie jest obiekt, jaki ma identyfikator i jak go utrzymać między klatkami.

## 10. Najkrótsze podsumowanie

`44.02` = **ByteTrack + CMC + predict-fill (≤8 kl.) + interpolacja
liniowa (≤60 kl.)** z agresywnymi progami detekcji (0.6/0.7/0.9) i
długim buforem traków (60 kl.). Detektor jest tu wąskim gardłem (pokrywa
30–50% GT per klatka), więc cały sens dodatków polega na **wyciąganiu
maksymalnego recall** z podtrzymywania torów przez krótko- i
średnioterminowe predykcje, bez wpuszczania FP.
