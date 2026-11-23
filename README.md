# 🤖 RAG-ассистент с ChromaDB, кешированием и оценкой RAGAS

Проект реализует полный цикл работы RAG-ассистента (Retrieval-Augmented Generation) на базе локального векторного хранилища **ChromaDB**, SQLite-кеширования ответов, оценки качества через **RAGAS** и подключения к LLM API (**Gemini / OpenAI API** и **GigaChat API**).

В качестве базы знаний используется документация IT-компании **ООО «ХАЙТЕК МЕДИА»** (`hi-techmedia.ru` — веб-разработка, внедрение CRM Битрикс24, клиентский FAQ и регламент SLA технической поддержки).

---

## 📋 Описание работы системы

Процесс обработки запроса в RAG-пайплайне (`rag_pipeline.py`):

1. **Пользователь задает вопрос** через консоль или программный интерфейс.
2. **Проверка кеша (`cache.py`)** — если вопрос уже задавался, ответ мгновенно возвращается из SQLite-базы (`~0.0003 сек.`).
3. **Поиск релевантных документов (`vector_store.py`)** — семантический поиск в ChromaDB (`top_k=3`) с поддержкой фильтрации по метаданным разделов (`services`, `faq`, `sla`).
4. **Формирование контекста** — сборка найденных фрагментов текста с метаданными первоисточников и историей диалога.
5. **Генерация ответа** — запрос к LLM (Gemini / OpenAI или GigaChat) строго на основе извлечённого контекста.
6. **Сохранение в кеш** — запись ответа и использованного контекста в SQLite по SHA-256 хешу запроса.
7. **Оценка качества (`evaluate_ragas.py`)** — автоматический расчёт метрик **Faithfulness**, **Answer Relevancy** и **Context Precision**.

---

## 🎯 Особенности

- ✅ **Локальное векторное хранилище ChromaDB** (`cosine` similarity)
- ✅ **Очистка текста и семантический чанкинг** с перекрытием (`overlap=90`) и мета-заголовками
- ✅ **SQLite-кеширование ответов** для мгновенного отклика при повторных запросах
- ✅ **Фильтрация по метаданным источников** (`1_hitech_services.txt`, `2_hitech_faq.txt`, `3_hitech_regulations.txt`)
- ✅ **Краткосрочная память диалога** в рамках текущей сессии
- ✅ **Два варианта подключения к LLM**: `assistant_api` (Gemini / OpenAI API) и `assistant_giga` (GigaChat API от Сбера)
- ✅ **Оценка качества через RAGAS** на контрольной выборке вопросов

---

## 📁 Структура проекта

```text
.
├── assistant_api/                   # RAG-ассистент (режим Gemini / OpenAI API)
│   ├── data/
│   │   ├── docs.txt                 # Сводная база знаний (9 смысловых чанков)
│   │   ├── 1_hitech_services.txt    # Услуги, портфолио и реквизиты компании
│   │   ├── 2_hitech_faq.txt         # FAQ: сроки, гарантия 12 мес., оплата, ЭДО
│   │   ├── 3_hitech_regulations.txt # Регламент SLA техподдержки и бэкапов
│   │   └── raw_dirty_texts.txt      # Исходные тексты до очистки
│   ├── app.py                       # Консольное приложение (CLI)
│   ├── rag_pipeline.py              # Основной RAG-пайплайн
│   ├── vector_store.py              # Загрузка, чанкинг и поиск в ChromaDB
│   ├── cache.py                     # SQLite-кеширование ответов
│   └── evaluate_ragas.py            # Оценка качества системы через RAGAS
│
├── assistant_giga/                  # RAG-ассистент (режим GigaChat API)
│   ├── data/                        # Документы базы знаний
│   ├── app.py                       # Консольное приложение для GigaChat
│   ├── gigachat_client.py           # Клиент авторизации OAuth и запросов к GigaChat
│   ├── rag_pipeline.py              # RAG-пайплайн для GigaChat
│   ├── vector_store.py              # Векторное хранилище ChromaDB
│   └── cache.py                     # SQLite-кеш ответов
│
├── .env.example                     # Пример конфигурации переменных окружения
├── requirements.txt                 # Зависимости Python
└── README.md                        # Документация проекта
```

---

## 📦 Установка и настройка

### 1. Создание виртуального окружения и установка зависимостей

```bash
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# или venv\Scripts\activate (Windows)

pip install -r requirements.txt
```

### 2. Настройка `.env`

Скопируйте `.env.example` в `.env` и укажите API-ключ:

```bash
cp .env.example .env
```

Содержимое `.env`:
```env
OPENAI_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here

GIGACHAT_RQUID=your_gigachat_rquid_here
GIGACHAT_AUTH_KEY=your_gigachat_auth_key_here
```

---

## 🚀 Использование

### 1. Интерактивный консольный ассистент (API Mode)

```bash
cd assistant_api
python app.py
```

Доступные команды в консоли:
- `stats` — просмотр статистики ChromaDB, SQLite-кеша и памяти
- `clear` — очистка SQLite-кеша
- `forget` — сброс краткосрочной памяти диалога
- `filter:services` / `filter:faq` / `filter:sla` / `filter:off` — фильтрация поиска по категориям документов
- `exit` / `quit` — выход

### 2. Автоматический тест RAG-пайплайна и кеша

```bash
cd assistant_api
python rag_pipeline.py
```

### 3. Оценка качества системы через RAGAS

```bash
cd assistant_api
python evaluate_ragas.py
```

Скрипт выполняет 5 контрольных запросов и рассчитывает метрики:
- **Faithfulness** (`0.9600`) — верность ответа контексту без галлюцинаций
- **Answer Relevancy** (`0.9591`) — релевантность ответа заданному вопросу
- **Context Precision** (`0.9760`) — точность извлечённого контекста
- **Сводный балл**: `0.9650`

### 4. Запуск версии с GigaChat API

```bash
cd assistant_giga
python app.py
```
