# face-segmentation-70+

Repozytorium to służy do przetestowania sieci neuronowej stworzonej do segmentacji twarzy ze zdjęć.
W tym celu wykorzystano model dostępny [tutaj](https://github.com/ipazc/mtcnn)

Plik main.py posłużył do przetworzenia zdjęć dostępnych w folderze *dataset* i wykrycia w nich twarzy. Wyniki analizy zapisane zostały w folderze *result*, w którym na każdym zdjęciu umieszczone zostały prostokąty z wykrytą twarzą (kolor niebieski) oraz z ground truth (kolor zielony). Dodatkowo w rogu zdjęcia znajduje się wartość statystyki IoU.

W pliku utils znajdują się dodatkowe funkcję wspomagające analizę. Opis każdej z nich został umieszczony bezpośrednio w funkcji.
