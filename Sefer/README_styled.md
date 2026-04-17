# 🪦 Sefer OCR Dashboard  
> 🧠 *Локальное Streamlit-приложение для распознавания номеров на табличках и пакетного переименования фото*

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Container-Docker-blue?logo=docker)
![Python](https://img.shields.io/badge/Python-3.11-yellow?logo=python)
![GPU](https://img.shields.io/badge/GPU-NVIDIA-green)
![OCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)

---

## 🚀 Описание

**Sefer OCR Dashboard** — это локальное веб-приложение на **Streamlit** для проекта **«Сэфер»**,  
которое позволяет автоматически:

- находить табличку на фотографии с помощью уже обученного детектора YOLO;
- распознавать номер на табличке через PaddleOCR;
- формировать новое имя файла по правилам проекта;
- сортировать результаты по уверенности распознавания;
- вручную исправлять сомнительные и нераспознанные случаи прямо в интерфейсе;
- сохранять отчёты и копировать / перемещать итоговые файлы в `renamed_files`.

Приложение может запускаться:
- 💻 **локально без Docker**
- 🐳 **в Docker**
- ⚡ **в GPU-контейнере через NVIDIA**

---

## 📦 Содержимое проекта

| 📄 Файл | 🧠 Назначение |
|:--------|:--------------|
| `app.py` | Основной код Streamlit-приложения |
| `sefer_pipeline.py` | OCR-пайплайн, логика детекции, OCR, confidence и переименования |
| `requirements.txt` | Зависимости Python |
| `Dockerfile.gpu` | Сборка GPU-образа Docker |
| `.dockerignore` | Исключения из Docker build context |
| `README.md` | Документация и инструкция по запуску |
| `best.pt` | Предобученные веса детектора YOLO |

---

## 📁 Важно про `best.pt`

Файл **`best.pt` должен лежать рядом с `app.py`**.

Пример структуры:
```text
sefer_repo_final/
├── app.py
├── sefer_pipeline.py
├── requirements.txt
├── Dockerfile.gpu
├── .dockerignore
├── README.md
└── best.pt
```

---

## 🐳 Запуск в Docker (GPU)

### 1️⃣ Перейти в папку проекта
```bash
cd path/to/sefer_repo_final
```

### 2️⃣ Собрать Docker-образ
```bash
docker build -f Dockerfile.gpu -t sefer-streamlit-gpu .
```

### 3️⃣ Запустить контейнер
```bash
docker run --rm --gpus all -p 8501:8501 \
  -v "/absolute/path/to/images:/data/images" \
  -v "/absolute/path/to/output:/data/output" \
  -v "/absolute/path/to/best.pt:/app/best.pt" \
  sefer-streamlit-gpu
```

После запуска приложение будет доступно по адресу:

👉 [http://localhost:8501](http://localhost:8501)

---

## 🖥️ Запуск без Docker

Если Docker не используется:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Если команда `streamlit` не находится:

```bash
python -m streamlit run app.py
```

После запуска откройте:

👉 [http://localhost:8501](http://localhost:8501)

---

## ⚙️ Что нужно указать в приложении

После запуска в интерфейсе приложения задаются:

- **Папка с фото**
- **Папка с результатами**
- настройки confidence threshold
- режим файловых операций: `copy` или `move`
- префикс и логика итогового имени
- локальные пути моделей PaddleOCR (при необходимости)

---

## 📊 Что создаётся в папке результатов

После обработки приложение сохраняет:

```text
ocr_pipeline_output/
├── ocr_report.csv
├── ocr_report.xlsx
├── ocr_report.json
├── rename_log.csv
└── renamed_files/
```

При включённых debug-опциях дополнительно создаются:

```text
crops/
prepared_crops/
visualizations/
```

---

## 🧠 Логика работы

Приложение делит результат на три группы:

- ✅ **Распознано**
- ⚠️ **Распознано, но с сомнением**
- ❌ **Не распознано**

Все сомнительные и нераспознанные фото выводятся внизу интерфейса,  
где пользователь может вручную изменить итоговое имя файла перед сохранением.

---

## 🧱 Пример Dockerfile.gpu

```dockerfile
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY sefer_pipeline.py /app/sefer_pipeline.py
COPY README.md /app/README.md

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

---

## 🧠 Автор

**Каргальцев Никита**  
📧 Telegram: [https://t.me/nikitakargaltsev](https://t.me/nikitakargaltsev)

💻 GitHub: [https://github.com/KargaltsevNikita](https://github.com/KargaltsevNikita)
