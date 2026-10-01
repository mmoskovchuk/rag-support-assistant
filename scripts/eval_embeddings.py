"""Compare embedding models and chunk sizes on the indexed demo documents.

Run inside the API container (models are cached in the hf_cache volume):
    docker compose exec -T rag-api python - < scripts/eval_embeddings.py
"""

import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from support_rag.archive import parse_archive_text
from support_rag.chunking import build_splitter

ANSWERABLE = [
    ("Скільки днів щорічної відпустки?", "vacation-policy.docx"),
    ("Скільки днів відпустки, якщо я працюю в компанії 4 роки?", "vacation-policy.docx"),
    ("За скільки днів треба подати заяву на відпустку?", "vacation-policy.docx"),
    ("Які добові за кордоном?", "business-trips.pdf"),
    ("Який ліміт на готель у відрядженні по Україні?", "business-trips.pdf"),
    ("Коли треба здати авансовий звіт?", "business-trips.pdf"),
    ("Як часто треба змінювати пароль?", "information-security.pdf"),
    ("Куди повідомити, якщо я загубив ноутбук?", "information-security.pdf"),
    ("Скільки днів на тиждень можна працювати з дому?", "remote-work-order-scan.jpg"),
    ("How much is the internet compensation for remote work?", "remote-work-order-scan.jpg"),
    ("Скільки триває випробувальний строк?", "onboarding-guide.docx"),
    ("Хто такий наставник і чим він допомагає?", "onboarding-guide.docx"),
]
UNANSWERABLE = [
    "Чи можна привести собаку в офіс?",
    "Яка зарплата у junior розробника?",
    "Де знаходиться найближча їдальня?",
    "Чи оплачує компанія спортзал?",
    "Як отримати корпоративну машину?",
    "What is the company's stock price?",
]

MODELS = [
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    ("BAAI/bge-m3", "", ""),
]


def load_chunks(chunk_size, overlap):
    splitter = build_splitter(chunk_size, overlap)
    chunks = []
    for txt in sorted(Path("/data/processed").glob("*/*.txt")):
        archived = parse_archive_text(txt.read_text(encoding="utf-8"), txt.stem)
        for page in archived.pages:
            chunks += [(archived.source, c) for c in splitter.split_text(page.text)]
    return chunks


def evaluate(model_name, q_prefix, d_prefix, chunks):
    t0 = time.time()
    model = SentenceTransformer(model_name, device="cpu")
    load = time.time() - t0
    t0 = time.time()
    doc_vecs = model.encode([d_prefix + c for _, c in chunks], normalize_embeddings=True)
    index_time = time.time() - t0

    def top(question):
        qv = model.encode(q_prefix + question, normalize_embeddings=True)
        scores = doc_vecs @ qv
        best = int(scores.argmax())
        return chunks[best][0], float(scores[best])

    t0 = time.time()
    pos = [(top(q), src) for q, src in ANSWERABLE]
    neg = [top(q)[1] for q in UNANSWERABLE]
    query_ms = (time.time() - t0) / (len(ANSWERABLE) + len(UNANSWERABLE)) * 1000

    hit1 = sum(found == src for (found, _), src in pos)
    pos_scores = [s for (_, s), _ in pos]
    # Best threshold: answerable must reach it with the right document, unanswerable must stay below.
    best_acc, best_t = 0, 0
    for t in sorted(set(pos_scores + neg)):
        acc = sum(s >= t and f == src for (f, s), src in pos) + sum(s < t for s in neg)
        if acc > best_acc:
            best_acc, best_t = acc, t
    margin = min(pos_scores) - max(neg)
    return dict(
        hit1=hit1, acc=best_acc, t=best_t, margin=margin, pos_min=min(pos_scores),
        neg_max=max(neg), load=load, index=index_time, query_ms=query_ms,
        dim=doc_vecs.shape[1],
    )


total = len(ANSWERABLE) + len(UNANSWERABLE)
for size, overlap in [(1000, 150), (500, 80)]:
    chunks = load_chunks(size, overlap)
    print(f"\n=== CHUNK_SIZE={size}: {len(chunks)} chunks ===")
    print(f"{'model':58} top1    acc@best_t      margin  pos_min neg_max  dim  query")
    for name, qp, dp in MODELS:
        r = evaluate(name, qp, dp, chunks)
        print(
            f"{name:58} {r['hit1']:2}/{len(ANSWERABLE)}  {r['acc']:2}/{total} @{r['t']:.3f}  "
            f"{r['margin']:+.3f}  {r['pos_min']:.3f}   {r['neg_max']:.3f}  {r['dim']:4} {r['query_ms']:4.0f}ms",
            flush=True,
        )
