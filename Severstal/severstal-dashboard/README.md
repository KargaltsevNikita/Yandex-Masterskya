# 🧬 Severstal Dashboard  
> 🧠 *Интерактивная кластеризация данных на Streamlit с использованием FAISS и Decision Tree*

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Container-Docker-blue?logo=docker)
![Python](https://img.shields.io/badge/Python-3.11-yellow?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Описание

**Severstal Dashboard** — это интерактивное веб-приложение для визуализации и анализа данных,  
позволяющее выполнять **кластеризацию с помощью FAISS**, анализировать корреляции,  
строить деревья решений и исследовать структуру данных.

Приложение полностью изолировано в **Docker**, что обеспечивает:
- 🧩 Полную переносимость  
- 🔒 Отсутствие конфликтов зависимостей  
- ⚡ Быструю установку без Python-окружений  

---

## 📦 Содержимое проекта

| 📄 Файл | 🧠 Назначение |
|:--------|:--------------|
| `app.py` | Основной код приложения Streamlit |
| `requirements.txt` | Список зависимостей Python |
| `Dockerfile` | Инструкции сборки Docker-образа |
| `.dockerignore` | Исключения из сборки |
| `README.md` | Документация и инструкция по запуску |
| `Research.ipynb` | Исследовательский налаиз на тестовых данных |
| EXPORT_YANDEX_v4.xlsx` | Тестовые данные |

---

## 🐳 Запуск в Docker

### 1️⃣ Клонировать репозиторий
```bash
git clone https://github.com/<твой_логин>/severstal-dashboard.git
cd severstal-dashboard
```

### 2️⃣ Собрать Docker-образ
```bash
docker build -t severstal-dashboard .
```

### 3️⃣ Запустить контейнер
```bash
docker run -p 8501:8501 severstal-dashboard
```

После запуска появится сообщение:
```
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
```

### 4️⃣ Открыть приложение
Перейдите в браузере 👉 [http://localhost:8501](http://localhost:8501)

---

## 🧰 Альтернатива: запуск без Docker
Если Docker не установлен:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🖼️ Пример интерфейса

*(Добавь свой скриншот, если хочешь)*  
```
📊 Выбор данных → 🔍 Анализ → 🤖 Кластеризация → 🌳 Интерпретация
```

<img src="https://user-images.githubusercontent.com/your-screenshot-id.png" width="800"/>

---

## 🧱 Пример Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y     build-essential     libopenblas-dev     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY app.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🧠 Автор

**Каргальцев Никита**  
📧 Telegram: [https://t.me/nikitakargaltsev](https://t.me/nikitakargaltsev)

💻 GitHub: [https://github.com/<твой_логин>](https://github.com/KargaltsevNikita)
