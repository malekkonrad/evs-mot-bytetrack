Zbiór danych
Do dyspozycji mają Państwo zbiór danych podzielony na sekwencje treningowe evs_mot-train oraz testowe evs_mot-test . Dla każdej sekwencji, w folderze det znajdują się detekcje, których należy użyć do inicjalizacji algorytmu śledzenia. W folderze img1 znajdują się poszczególne klatki sekwencji, natomiast w folderze gt znajduje się wzorcowy wynik śledzenia (dostępny tyllko dla zbioru treningowego).

W pliku det.txt, każda linijka oznacza jedną detekcję ludzkiej sylwetki i jest następującego formatu:

```
<frame>,-1,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<detection_confidence>
```

Natomiast plik gt.txt:

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<eval_flag>,<class>,<visibility>
```
Gdzie:

```<frame>``` - numer klatki sekwencji (pierwsza klatka ma numer 1)

```<id> ```- unikalne id identyfikujące śledzony obiekt (istota całego zadania śledzenia)

```<bb_left>,<bb_top>,<bb_width>,<bb_height>``` - współrzędne pikselowe prostokąta otaczającego pozycję obiektu, odpowiednio: lewy bok, górny bok, szerokość, wysokość (dopuszczalne wartości zmiennoprzecinkowe)

```<detection_confidence>``` - miara pewności detektora co do detekcji (w zakresie 0 - 1)

```<eval_flag> ```- flaga informująca, czy dany obiekt jest traktowany jako wzorcowy i powinien być brany pod uwagę w ewaluacji (0 - nie, 1 - tak)

```<class> ```- klasa obiektu, interesuje nas klasa 1, czyli sylwetki ludzi, którzy nie są wewnątrz budynków i nie są odbiciami (np w szybach budynków/samochodów).

```<visibility>``` - liczba w zakresie ```(0 - 1] ```informująca, jaka część obiektu jest widoczna w danej klatce

Zadanie
Dla każdej z sekwencji testowych ```MOT_01, MOT_06, MOT_07``` Państwa algorytm powinien wygenerować plik .txt, w którym każda linijka oznacza jedną instancję obiektu i powinna być w następującym formacie:


```<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,1,-1,-1,-1```

Uwaga, nie jest istotne jakie id przypiszemy do obiektu w trakcie inicjalizacji. Ważne jest, aby było one unikalne dla danego obiektu w obrębie sekwencji.

Format przesyłanego rozwiązania
Rozwiązanie to pojedynczy plik .zip, który powinien zawierać folder submission . W folderze submission powinien znajdować się folder data z wynikiem śledzenia dla każdej z sekwencji testowych oraz folder source z kodem źródłowym algorytmu użytego do generacji rzeczonego wyniku. Podsumowując, zawartość przesłanego pliku .zip powinna wyglądać następująco:

```
submission/
├──  data/
│     ├──  MOT_01.txt
│     ├──  MOT_06.txt
│     └──  MOT_07.txt
└──  source/
       └──  ...

```



```bash
uv run python eval_train.py --cmc --predict-lost-frames 8  --track-thresh 0.6 --new-track-thresh 0.7 --match-thresh 0.9  --track-buffer 60 --interpolate --max-gap 60```


```bash 

uv run python visualize_live.py --sequence MOT_05 --mot-file submission/_eval_tmp/MOT_05.txt --title "Predictions"```

```bash
uv run python visualize_live.py --sequence MOT_05 --mot-file evs_mot_public_dataset/evs_mot-train/MOT_05/gt/gt.txt --gt --title "Ground Truth"```


Klawisze (w aktywnym oknie cv2)
                                                                                         
  - q / Esc — wyjście                                       
  - space — pauza/play                                                                   
  - n lub → — krok do przodu (wymusza pauzę)                                             
  - p lub ← — krok wstecz                                                                
  - r — restart sekwencji                                                                
  - + / - — szybciej/wolniej                                                             
                                                                                         
  HUD pokazuje                                              
                                                                                         
  title | frame X/N | boxes K | fps | [PAUSED]   