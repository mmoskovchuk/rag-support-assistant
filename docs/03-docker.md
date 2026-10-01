# 03. Docker: що і навіщо

## Загальна картина

```
┌──────────── docker network: rag-support-assistant_default ────────────┐
│                                                                       │
│   n8n:5678  ──HTTP──▶  rag-api:8000  ──HTTP──▶  chromadb:8000         │
└───────────────────────────────────────────────────────────────────────┘
   127.0.0.1:5690          127.0.0.1:8100           127.0.0.1:8101
       (UI)                (API, /docs)            (debug)
```

Усередині мережі Docker сервіси звертаються один до одного **за іменем сервісу** (`http://chromadb:8000`). `localhost` усередині контейнера означає сам цей контейнер.

## `docker-compose.yml` по блоках

### `chromadb`
- `image: chromadb/chroma:1.5.9`: версія зафіксована і **має збігатися** з `chromadb==1.5.9` у `requirements.txt`. Клієнт і сервер різних версій можуть не порозумітися.
- `volumes: chroma_data:/data`: іменований том. Вектори переживуть `docker compose down` і пересоздання контейнера. Видалити їх можна лише явно через `docker compose down -v`.

### `rag-api`
- `build: ./app`: образ збирається з нашого `Dockerfile`.
- `env_file: .env`: усі змінні з `.env` потрапляють у контейнер, а Pydantic їх прочитає.
- `./data:/data`: **bind mount**, тобто папка хоста видна в контейнері. Ви кладете файли в `data/inbox` на своїй машині, а сервіс їх бачить.
- `hf_cache:/models`: кеш Hugging Face. Без нього модель (`bge-m3`, ~2.2 GB) завантажувалась би при кожному перестворенні контейнера.
- `depends_on`: compose запускає ChromaDB першим. Але «запущено» не означає «готово приймати запити», тому в коді ще є ретраї підключення.

### `n8n`
- `N8N_ENCRYPTION_KEY`: ключ шифрування credentials. **Не губіть його:** без нього збережені credentials не розшифруються.
- `NODES_EXCLUDE='["n8n-nodes-base.executeCommand"]'`: у n8n 2.x за замовчуванням вимкнено і Execute Command, і Local File Trigger. Ми перевизначаємо список, тож вимкненою лишається лише Execute Command.
- `N8N_RESTRICT_FILE_ACCESS_TO=/data/inbox`: файлові ноди n8n бачать лише цю папку.
- `./data/inbox:/data/inbox:ro`: **read-only**. n8n лише спостерігає; переміщувати файли може тільки API. Принцип найменших привілеїв.

### Порти `127.0.0.1:XXXX:YYYY`
Ліве число є портом на вашій машині, праве є портом усередині контейнера. Ліві числа беруться з `.env` (`API_HOST_PORT=8100`, `CHROMA_HOST_PORT=8101`, `N8N_HOST_PORT=5690`), тож якщо порт зайнятий іншим проєктом, достатньо змінити `.env`. Праві числа фіксовані, тому адреси всередині мережі (`http://rag-api:8000`) не змінюються.

Без `127.0.0.1` Docker відкрив би порт на **всіх** мережевих інтерфейсах, тобто n8n і API були б доступні з локальної мережі або інтернету. Для продакшну ставлять reverse proxy (Caddy/Traefik/Nginx) з HTTPS.

### `restart: unless-stopped`
Контейнер підніметься сам після збою чи перезавантаження сервера.

## `app/Dockerfile` по кроках

1. **`python:3.12-slim`**: мінімальний Debian з Python. Проєкт вимагає Python 3.10+, і 3.12 є стабільним вибором для ML-бібліотек.
2. **apt-пакети:** `tesseract-ocr` + мовні дані `ukr`/`eng`, `poppler-utils` (рендер PDF), `curl` (для healthcheck). `rm -rf /var/lib/apt/lists/*` зменшує образ.
3. **CPU-версія torch окремо.** Звичайний `pip install torch` тягне CUDA (кілька ГБ), а нам GPU не потрібен.
4. **Спочатку `requirements.txt`, потім код.** Docker кешує шари: якщо змінився лише код, залежності не перевстановлюються, і пересборка займає секунди замість хвилин.
5. **`USER appuser`**: якщо хтось зламає сервіс, він не отримає root у контейнері.
6. **`HEALTHCHECK`**: Docker сам перевіряє `/health`; `docker compose ps` показує `healthy`/`unhealthy`. `start-period=180s` дає час на завантаження моделі.

## Корисні команди

```bash
docker compose up -d --build        # зібрати і запустити у фоні
docker compose ps                   # стан і health
docker compose logs -f rag-api      # логи в реальному часі
docker compose restart rag-api      # перезапуск після зміни .env
docker compose up -d --build rag-api  # пересобрати після зміни коду
docker compose exec rag-api bash    # зайти в контейнер
docker compose down                 # зупинити (дані зберігаються)
docker compose down -v              # зупинити і ВИДАЛИТИ томи (вектори, n8n!)
```

## Типові проблеми

| Симптом | Причина / рішення |
|---|---|
| `Permission denied` на `/data/...` | UID вашого користувача не 1000. Перевірте `id -u`; або змініть UID у Dockerfile, або виконайте `sudo chown -R 1000:1000 data/` |
| `rag-api` довго `starting` | Перший старт завантажує модель. Дивіться `docker compose logs -f rag-api` |
| Local File Trigger не бачить файлів | Увімкніть **Use Polling** в опціях ноди |
| `ValidationError ... openai_api_key` | Не заповнено `.env` |
| `port is already allocated` при старті | Порт на хості зайнятий іншим проєктом. Перевірте `ss -tlnp \| grep <порт>` і змініть `*_HOST_PORT` у `.env` |
| n8n: `Connection refused` до `localhost:8000` | Використовуйте `http://rag-api:8000`, бо всередині контейнера `localhost` означає сам n8n |
