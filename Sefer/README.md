# Sefer Streamlit App (GPU Docker)

Локальное Streamlit-приложение для проекта **«Сэфер»**.  
Этот вариант README и Dockerfile подготовлен для **GPU-запуска через Docker + NVIDIA Container Toolkit**.

## Что умеет приложение

- читает фото из локальной папки;
- использует готовые веса детекции `best.pt`;
- запускает OCR через PaddleOCR;
- формирует новые имена файлов по правилам проекта;
- выводит сводку по трём группам:
  - **Распознано**
  - **Распознано, но с сомнением**
  - **Не распознано**
- позволяет вручную исправить итоговые имена для сомнительных и нераспознанных файлов;
- сохраняет отчёты CSV / XLSX / JSON;
- по желанию копирует или перемещает файлы в `renamed_files`.

## Структура проекта

```text
.
├── app.py
├── sefer_pipeline.py
├── requirements.txt
├── Dockerfile.gpu
├── .dockerignore
├── README.md
└── best.pt
```

## Важно про `best.pt`

Приложение ожидает файл весов **рядом с `app.py`**:

```text
./best.pt
```

Вы можете:
- положить `best.pt` в корень проекта перед сборкой контейнера;
- или примонтировать файл в контейнер как `/app/best.pt`.

## Локальный запуск без Docker

```bash
conda activate sefer_yolo
pip install -r requirements.txt
python -m streamlit run app.py
```

Затем откройте:

```text
http://localhost:8501
```

## GPU-запуск через Docker

### 1. Требования на хосте

На машине должны быть:
- установленный **NVIDIA GPU driver**;
- установленный **Docker**;
- установленный и настроенный **NVIDIA Container Toolkit**.

Без NVIDIA Container Toolkit флаг `--gpus all` работать не будет.

### 2. Сборка образа

```bash
docker build -f Dockerfile.gpu -t sefer-streamlit-gpu .
```

### 3. Запуск контейнера с GPU

Пример с примонтированием:
- папки с фото
- папки для результатов
- файла `best.pt`

```bash
docker run --rm --gpus all -p 8501:8501 \
  -v "/absolute/path/to/images:/data/images" \
  -v "/absolute/path/to/output:/data/output" \
  -v "/absolute/path/to/best.pt:/app/best.pt" \
  sefer-streamlit-gpu
```

После запуска откройте:

```text
http://localhost:8501
```

И в интерфейсе укажите:
- **Папка с фото**: `/data/images`
- **Папка с результатами**: `/data/output`

## Проверка, что контейнер видит GPU

Быстрая проверка на хосте:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 nvidia-smi
```

Если выводится информация о GPU, контейнерный доступ к GPU настроен корректно.

## Что создаётся в папке результатов

В выбранной папке результатов приложение сохраняет:

```text
ocr_pipeline_output/
├── ocr_report.csv
├── ocr_report.xlsx
├── ocr_report.json
├── rename_log.csv
└── renamed_files/
```

Дополнительно, если включены debug-опции:

```text
crops/
prepared_crops/
visualizations/
```

## Публикация на GitHub

Если `best.pt` нельзя выкладывать в репозиторий:
- не добавляйте его в Git;
- добавьте `best.pt` в `.gitignore`;
- подключайте его через volume mount:
  `-v "/absolute/path/to/best.pt:/app/best.pt"`.

## Полезно знать

Этот Dockerfile ориентирован на ваш стек с CUDA 12.6.  
Если вы будете менять версии `torch`, `paddlepaddle` или базовый CUDA-образ, лучше проверять совместимость отдельно.
