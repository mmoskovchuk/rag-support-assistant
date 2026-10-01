"""Short human-readable topic for every document, e.g. "Annual leave policy".

Topics are generated once at ingestion time and stored in chunk metadata.
When the knowledge base has no answer, the closest topics are suggested to
the employee instead of a bare "not found".
"""

import logging
import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

TOPIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You name internal company documents for a knowledge-base catalogue.\n"
            "Read the beginning of the document and reply with its topic: 3 to 8 words, "
            "in the language of the document, sentence case, no quotes, no trailing period, nothing else.",
        ),
        ("human", "{text}"),
    ]
)


def topic_from_filename(filename: str) -> str:
    """Fallback topic when the LLM is unavailable: "vacation_policy-2026.pdf" -> "Vacation policy 2026"."""
    words = re.sub(r"[_\-.]+", " ", Path(filename).stem).strip()
    return words[:1].upper() + words[1:] if words else filename


class TopicNamer:
    def __init__(self, llm: BaseChatModel, max_chars: int) -> None:
        self._chain = TOPIC_PROMPT | llm | StrOutputParser()
        self._max_chars = max_chars

    def name(self, text: str, filename: str) -> str:
        try:
            # The beginning (title, preamble) describes a document well and keeps the call cheap.
            topic = self._chain.invoke({"text": text[: self._max_chars]})
        except Exception:
            # A missing topic must never block indexing: the document itself matters more.
            logger.warning("Topic generation failed for %s, using the file name", filename, exc_info=True)
            return topic_from_filename(filename)

        topic = topic.strip(" \n\"'«»„“”.")
        if not topic or len(topic) > 120:
            return topic_from_filename(filename)
        return topic
