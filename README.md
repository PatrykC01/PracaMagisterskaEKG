# Wykrywanie arytmii serca z wykorzystaniem głębokich sieci neuronowych

Repozytorium zawiera kod eksperymentów wykonanych w ramach pracy magisterskiej dotyczącej klasyfikacji uderzeń EKG ze zbioru MIT-BIH Arrhythmia Database. Badanie obejmuje wybór długości segmentu, porównanie reprezentacji sygnału, strojenie architektur, ocenę technik balansowania klas oraz końcową ocenę na wydzielonym zbiorze testowym DS2.

Klasyfikacja wykorzystuje cztery klasy AAMI:

- `N` – uderzenia prawidłowe i pokrewne,
- `S` – nadkomorowe uderzenia ektopowe,
- `V` – komorowe uderzenia ektopowe,
- `F` – uderzenia fuzyjne.

Klasa `Q` nie była uwzględniana w eksperymentach.

## Protokół badawczy

- Częstotliwość próbkowania: 360 Hz.
- Wybrane okno segmentacji: 65 próbek przed i 110 próbek po punkcie adnotacji, łącznie 175 próbek.
- Podział przeprowadzono na poziomie rekordów: żaden rekord nie występuje w więcej niż jednej części zbioru.
- Zastosowano standardowy podział DS1/DS2 używany w literaturze. Rekordy [201 i 202](https://physionet.org/physiobank/database/html/mitdbdir/records.htm) pochodzą od tego samego pacjenta i znajdują się odpowiednio w DS1 TRAIN oraz DS2 TEST. Z tego względu podział nie jest całkowicie rozłączny na poziomie pacjentów.
- `DS1 TRAIN` i `DS1 VAL` służą do strojenia, wyboru modeli i analiz pomocniczych.
- `DS2 TEST` pozostaje odseparowany do etapu końcowej oceny.
- Każdy model otrzymuje segment EKG oraz cztery cechy R-R.
- Główne metryki to Macro F1, Rare-Macro F1 oraz Min-Rare F1.

## Struktura repozytorium

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── 01_select_segmentation_window_ds1.py
│   ├── 02_generate_dataset_65x110.py
│   ├── 03a_tune_stft.py
│   ├── 03b_tune_cwt.py
│   ├── 03c_tune_dwt_swt.py
│   ├── 04_confirm_tf_hyperparameters.py
│   ├── 05_select_optimal_representation.py
│   ├── 06a_tune_architectures.py
│   ├── 06b_extend_hybrid_architectures.py
│   ├── 06c_repair_architecture_candidates.py
│   ├── 06d_confirm_architectures.py
│   ├── 07a_screen_balancing_methods.py
│   ├── 07b_confirm_balancing_methods.py
│   ├── 08_final_ds2_evaluation.py
│   ├── 09a_analyze_ds2_class_f_errors.py
│   └── 09b_screen_ds1_group_f_generalization.py
└── results/
    └── wyniki CSV, JSON i PNG wykorzystane w pracy
```

## Kolejność eksperymentów

| Skrypt | Zastosowanie |
|---|---|
| `01_select_segmentation_window_ds1.py` | Wybór długości segmentu wyłącznie na DS1. |
| `02_generate_dataset_65x110.py` | Przygotowanie końcowych zbiorów TRAIN, VAL i TEST dla okna 65+110. |
| `03a_tune_stft.py` | Screening hiperparametrów STFT. |
| `03b_tune_cwt.py` | Screening hiperparametrów CWT. |
| `03c_tune_dwt_swt.py` | Screening hiperparametrów DWT i SWT. |
| `04_confirm_tf_hyperparameters.py` | Potwierdzenie wybranych konfiguracji reprezentacji czasowo-częstotliwościowych. |
| `05_select_optimal_representation.py` | Porównanie RAW 1D z najlepszymi wariantami STFT, CWT, DWT i SWT. |
| `06a_tune_architectures.py` | Główny sweep sześciu rodzin architektur 1D i 2D. |
| `06b_extend_hybrid_architectures.py` | Rozszerzenie sweepu o modele CNN–BiLSTM oraz domknięcie badania CNN 2D. |
| `06c_repair_architecture_candidates.py` | Usunięcie duplikatów i ponowny wybór kandydatów z baz Optuny. |
| `06d_confirm_architectures.py` | Test potwierdzający 11 kandydatów architektur na nowych ziarnach losowości. |
| `07a_screen_balancing_methods.py` | Screening technik balansowania, w tym WGAN-GP. |
| `07b_confirm_balancing_methods.py` | Test potwierdzający wybranych technik balansowania. |
| `08_final_ds2_evaluation.py` | Jednorazowa ocena zamrożonych konfiguracji na DS2. |
| `09a_analyze_ds2_class_f_errors.py` | Diagnostyczna analiza zapisanych predykcji klasy F na DS2, bez ponownego treningu i wyboru modelu. |
| `09b_screen_ds1_group_f_generalization.py` | Diagnostyczny stress-test generalizacji klasy F między rekordami, wykonywany wyłącznie na DS1. |

Skrypty `01–07` nie wykorzystują DS2. Skrypt `08` przeprowadza końcową ocenę na DS2. Skrypt `09a` analizuje wyniki zapisane przez etap `08`, natomiast `09b` jest analizą diagnostyczną DS1 i nie służy do zmiany modelu po poznaniu wyników DS2.

## Dane

Oryginalne rekordy MIT-BIH Arrhythmia Database nie są przechowywane w repozytorium. Można je pobrać z serwisu [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).

Skrypt `02_generate_dataset_65x110.py` tworzy:

```text
mitbih_train.npz
mitbih_val.npz
mitbih_test.npz
dataset_manifest.json
```

Pliki NPZ zawierają tablice `X`, `RR`, `Y` i `RECORD`. Nie są publikowane w repozytorium ze względu na rozmiar oraz możliwość ich odtworzenia z danych źródłowych.

Ścieżkę do pobranych rekordów oraz katalog wyjściowy można ustawić zmiennymi środowiskowymi. Przykład dla PowerShell:

```powershell
$env:MITDB_PATH="C:\sciezka\do\mitdb"
$env:OUTPUT_DIR="datasetostrrfixed_65x110"
python src/02_generate_dataset_65x110.py
```

Pozostałe skrypty były przygotowane przede wszystkim do uruchamiania w środowisku Kaggle. W razie uruchamiania lokalnego należy dostosować stałe określające katalogi wejściowe i wyjściowe.

## Instalacja

Zalecany jest Python 3.10 lub nowszy oraz karta graficzna zgodna z CUDA dla etapów treningowych.

```bash
python -m venv .venv
```

Aktywacja środowiska w systemie Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Instalacja zależności:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

W przypadku lokalnego treningu na GPU wersję PyTorch należy dobrać do zainstalowanej wersji CUDA zgodnie z instrukcją projektu PyTorch. W środowisku Kaggle PyTorch i obsługa CUDA są dostępne w obrazie wykonawczym.

## Uruchamianie

Każdy etap jest samodzielnym skryptem uruchamianym zgodnie z jego numerem, na przykład:

```bash
python src/03a_tune_stft.py
```

Część późniejszych etapów wyszukuje wyniki wcześniejszych eksperymentów w katalogach `/kaggle/input` i `/kaggle/working`. Przy wznawianiu obliczeń należy udostępnić odpowiednie pliki CSV, JSON lub bazy Optuny jako Kaggle Input albo zmienić katalogi wyszukiwania w konfiguracji skryptu.

## Wyniki

Katalog `results/` zawiera niewielkie pliki wynikowe wykorzystane do opracowania tabel, porównań statystycznych, macierzy pomyłek i wykresów przedstawionych w pracy. Nie obejmuje dużych plików roboczych, takich jak:

- zbiory `*.npz`,
- bazy Optuny `*.db`,
- checkpointy modeli `*.pt`,
- syntetyczne pule WGAN-GP,
- pełne logi środowiska Kaggle.

Losowość eksperymentów kontrolowano za pomocą zapisanych w skryptach ziaren. Konfiguracje, podziały danych i parametry poszczególnych etapów są również zapisywane w plikach manifestu znajdujących się w katalogach wynikowych.
