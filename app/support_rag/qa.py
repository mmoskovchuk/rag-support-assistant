"""Question answering: retrieve context, decide whether it is good enough, generate."""

import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import Settings
from .extraction import NO_PAGE

logger = logging.getLogger(__name__)

NO_ANSWER = "NO_ANSWER"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an internal support assistant for company employees.\n"
            "Answer ONLY with facts from the context below. Do not use outside knowledge.\n"
            f"If the context does not contain the answer, reply with exactly {NO_ANSWER} and nothing else.\n"
            "Answer in the same language as the question. Be concise and precise.\n"
            "After each fact, cite its source exactly as the label in square brackets above the fragment.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

# Fixed wording instead of an LLM call: the reply is predictable, free and instant.
NO_ANSWER_TEXT = {
    "uk": {
        "intro": "На жаль, у базі знань не знайшлося відповіді на ваше запитання.",
        "topics": "Можливо, потрібна інформація є в документах на такі теми:",
        "hint": "Спробуйте уточнити або переформулювати запитання з огляду на одну з цих тем.",
        "empty": "Наразі база знань ще не містить жодного документа.",
    },
    "en": {
        "intro": "Unfortunately, the knowledge base does not contain an answer to your question.",
        "topics": "The information you need may be in documents on these topics:",
        "hint": "Try refining or rephrasing your question with one of these topics in mind.",
        "empty": "The knowledge base does not contain any documents yet.",
    },
}


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[Document, float]]: ...


@dataclass
class Source:
    source: str
    topic: str | None
    page: int
    relevance: float


@dataclass
class Answer:
    status: Literal["answered", "no_answer"]
    answer: str
    reason: str | None = None
    # For "no_answer" these are the closest chunks found: useful for tuning the threshold.
    sources: list[Source] = field(default_factory=list)
    # For "no_answer": topics of the closest documents, most relevant first.
    topics: list[str] = field(default_factory=list)


def source_label(metadata: dict) -> str:
    page = metadata.get("page", NO_PAGE)
    name = metadata.get("source")
    return f"[{name}]" if page == NO_PAGE else f"[{name}, p. {page}]"


def format_context(hits: list[tuple[Document, float]]) -> str:
    return "\n\n".join(f"{source_label(doc.metadata)}\n{doc.page_content}" for doc, _ in hits)


def detect_language(text: str) -> str:
    # The corpus is Ukrainian/English, so "any Cyrillic letter" is a sufficient detector.
    return "uk" if re.search(r"[а-яіїєґ]", text, re.IGNORECASE) else "en"


def distinct_topics(hits: list[tuple[Document, float]], limit: int) -> list[str]:
    """Unique topics in order of relevance (hits are already sorted best first)."""
    topics: list[str] = []
    for doc, _ in hits:
        topic = doc.metadata.get("topic")
        if topic and topic not in topics:
            topics.append(topic)
        if len(topics) == limit:
            break
    return topics


def no_answer_message(language: str, topics: list[str]) -> str:
    text = NO_ANSWER_TEXT[language]
    if not topics:
        return f"{text['intro']} {text['empty']}"
    bullet_list = "\n".join(f"• {topic}" for topic in topics)
    return f"{text['intro']}\n\n{text['topics']}\n{bullet_list}\n\n{text['hint']}"


class QAService:
    def __init__(self, settings: Settings, retriever: Retriever, llm: BaseChatModel) -> None:
        self._s = settings
        self._retriever = retriever
        self._chain = PROMPT | llm | StrOutputParser()

    def ask(self, question: str) -> Answer:
        # Retrieval and LLM errors (ChromaDB or OpenAI down) propagate: that is an outage,
        # not a "no answer", and the API reports it as 503.
        # One wider search serves both purposes: the best top_k chunks become the context,
        # the rest of the pool supplies a variety of topics if there is no answer.
        pool = self._retriever.search(question, k=max(self._s.top_k, self._s.suggestion_pool_k))
        hits = pool[: self._s.top_k]
        relevant = [(doc, score) for doc, score in hits if score >= self._s.relevance_threshold]
        best = max((score for _, score in hits), default=0.0)
        logger.info("Retrieved %d hit(s), %d above threshold, best=%.3f", len(hits), len(relevant), best)

        if not relevant:
            return self._no_answer(question, "no_relevant_context", hits, pool)

        text = self._chain.invoke({"context": format_context(relevant), "question": question}).strip()

        if not text or NO_ANSWER in text:
            return self._no_answer(question, "llm_insufficient_context", hits, pool)

        return Answer(status="answered", answer=text, sources=self._sources(relevant))

    def _no_answer(self, question: str, reason: str, hits: list, pool: list) -> Answer:
        topics = distinct_topics(pool, self._s.max_suggested_topics)
        logger.info("No answer in the knowledge base (%s), suggesting %d topic(s)", reason, len(topics))
        return Answer(
            status="no_answer",
            answer=no_answer_message(detect_language(question), topics),
            reason=reason,
            sources=self._sources(hits),
            topics=topics,
        )

    @staticmethod
    def _sources(hits: list[tuple[Document, float]]) -> list[Source]:
        return [
            Source(
                source=str(doc.metadata.get("source")),
                topic=doc.metadata.get("topic"),
                page=int(doc.metadata.get("page", NO_PAGE)),
                relevance=round(score, 3),
            )
            for doc, score in hits
        ]
