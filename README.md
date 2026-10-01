# RAG Support Assistant

Локальна RAG-система для внутрішньої підтримки співробітників. Корпоративні документи (відскановані папери, звичайні PDF і Word) автоматично індексуються у векторну базу, а на запитання працівників у простому вебчаті відповідає LLM **лише** на основі знайдених фрагментів, з посиланням на джерело. Якщо відповіді немає, система не вигадує, а ввічливо пропонує до 5 тем документів, де варто пошукати.

> A local Retrieval-Augmented Generation service: scanned, digital PDF and Word documents → text extraction / OCR → Hugging Face embeddings → ChromaDB → grounded answers with citations via OpenAI in a minimal web chat; document ingestion is orchestrated by n8n.

## Архітектура

```mermaid
flowchart LR
    subgraph Ingestion
        S[Сканер / мережева папка] -->|PDF, DOCX, TIFF, PNG| IN[(data/inbox)]
        IN -. Local File Trigger .-> N1[n8n: Ingest workflow]
        N1 -->|POST /ingest| API
    end

    subgraph rag-api [rag-api · Python / FastAPI]
        API[FastAPI] --> EX[Текстовий шар PDF / Word]
        EX -->|немає тексту| OCR[Tesseract OCR]
        EX --> TP[Тема документа]
        OCR --> TP
        TP --> CH[LangChain splitter]
        CH --> EMB[HF embeddings]
        API --> QA[QA service]
    end

    EMB --> DB[(ChromaDB)]
    QA -->|semantic search| DB
    QA -->|prompt + context| OAI[OpenAI API]

    U[Співробітник] -->|вебчат → POST /ask| API
```

## Стек

| Шар | Технологія | Навіщо |
|---|---|---|
| Оркестрація | n8n 2.x | моніторинг папки, запуск індексації без коду |
| API + UI | Python 3.12, FastAPI, vanilla HTML/JS | HTTP-інтерфейс для n8n і вебчат без збирання фронтенду |
| Видобування тексту | pypdf, python-docx | точний текст з цифрових PDF і Word, без OCR |
| OCR | Tesseract (`ukr+eng`), Poppler | друкований текст зі сканів |
| RAG | LangChain | розбиття тексту, промпти, інтеграції |
| Embeddings | Hugging Face `paraphrase-multilingual-MiniLM-L12-v2` | локально, безкоштовно, підтримує українську |
| Vector DB | ChromaDB 1.5 | локальне зберігання векторів і метаданих |
| LLM | OpenAI API | генерація відповіді та назв тем документів |
| Інфраструктура | Docker Compose | запуск усього однією командою |

## Швидкий старт

```bash
cp .env.example .env          # заповнити секрети
docker compose up -d --build  # перший запуск: ~5–10 хв (збирання образу + модель)
curl http://localhost:8100/health
```

Вебчат: http://localhost:8100 (при першому відкритті вкажіть `API_KEY` з `.env`).

Порти на хості (8100 API/вебчат, 8101 ChromaDB, 5690 n8n) задаються в `.env` змінними `API_HOST_PORT`, `CHROMA_HOST_PORT`, `N8N_HOST_PORT`, якщо вони зайняті іншими проєктами.

Далі налаштуйте workflow індексації в n8n (http://localhost:5690) за інструкцією [docs/03-n8n-workflows.md](docs/03-n8n-workflows.md).

## API

| Метод | Шлях | Опис |
|---|---|---|
| GET | `/` | вебчат |
| GET | `/health` | стан сервісу та зв'язку з ChromaDB |
| POST | `/ingest` | `{"filename": "x.pdf"}`: обробити файл з `data/inbox` |
| POST | `/ingest/pending` | обробити все, що лишилось в `inbox` (страховка) |
| POST | `/ask` | `{"question": "..."}`: відповідь з джерелами або `no_answer` з підказками тем |

Усі ендпоінти, крім `/` і `/health`, вимагають заголовок `X-API-Key`. Інтерактивна документація доступна за адресою http://localhost:8100/docs.

## Документація

| Файл | Про що |
|---|---|
| [docs/01-architecture.md](docs/01-architecture.md) | архітектура, рішення та їх обґрунтування, захист від втрати даних |
| [docs/02-python-service.md](docs/02-python-service.md) | розбір Python-коду модуль за модулем |
| [docs/03-n8n-workflows.md](docs/03-n8n-workflows.md) | workflow n8n нода за нодою |
| [docs/04-docker.md](docs/04-docker.md) | Docker Compose та Dockerfile пояснено |

## Структура

```
rag-support-assistant/
├── app/                      # Python-сервіс (образ rag-api)
│   ├── support_rag/
│   │   ├── api.py            # FastAPI: ендпоінти, auth, request-id
│   │   ├── config.py         # усі налаштування з env + валідація
│   │   ├── logging_setup.py  # логування з correlation id
│   │   ├── file_store.py     # inbox → processing → processed | failed
│   │   ├── extraction.py     # PDF (текстовий шар або OCR), Word, зображення
│   │   ├── ocr.py            # Tesseract + Poppler
│   │   ├── topics.py         # коротка назва теми документа через LLM
│   │   ├── chunking.py       # LangChain text splitter + метадані
│   │   ├── vector_store.py   # ChromaDB + HF embeddings
│   │   ├── ingestion.py      # пайплайн індексації
│   │   ├── qa.py             # пошук → рішення → LLM / підказки тем
│   │   └── static/index.html # вебчат
│   ├── tests/                # pytest, без важких залежностей
│   ├── Dockerfile
│   └── requirements.txt
├── data/                     # runtime-дані (у git лише порожні папки)
├── docs/                     # технічна документація
├── n8n/workflows/            # експортовані workflow (JSON)
├── docker-compose.yml
└── .env.example
```

## Тести

```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```
