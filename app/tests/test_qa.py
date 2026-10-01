from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from support_rag.qa import NO_ANSWER, QAService, detect_language, distinct_topics, no_answer_message


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k):
        return self.hits[:k]


def doc(source="hr.pdf", topic="Annual leave", page=1):
    return Document(page_content="Vacation is 24 days.", metadata={"source": source, "topic": topic, "page": page})


def make_service(settings, hits, llm_answer):
    llm = FakeListChatModel(responses=[llm_answer])
    return QAService(settings, FakeRetriever(hits), llm)


def test_answers_when_context_is_relevant(settings):
    qa = make_service(settings, [(doc(), 0.8)], "24 days [hr.pdf, p. 1]")
    result = qa.ask("How long is vacation?")
    assert result.status == "answered"
    assert result.sources[0].source == "hr.pdf"
    assert result.topics == []


def test_no_answer_below_threshold_suggests_topics(settings):
    hits = [(doc("a.pdf", "Parking rules"), 0.3), (doc("b.pdf", "Business trips"), 0.2)]
    qa = make_service(settings, hits, "should not be called")
    result = qa.ask("Хто директор компанії?")
    assert result.status == "no_answer"
    assert result.reason == "no_relevant_context"
    assert result.topics == ["Parking rules", "Business trips"]
    assert result.answer.startswith("На жаль")
    assert "• Parking rules" in result.answer


def test_no_answer_when_llm_says_no_answer(settings):
    qa = make_service(settings, [(doc(), 0.9)], NO_ANSWER)
    result = qa.ask("What is the parking policy?")
    assert result.status == "no_answer"
    assert result.reason == "llm_insufficient_context"
    assert result.answer.startswith("Unfortunately")


def test_topics_come_from_the_wider_pool_not_only_top_k(settings):
    # top_k chunks all belong to one document; other topics are further down the pool.
    hits = [(doc("a.pdf", "Parking rules"), 0.3)] * settings.top_k + [(doc("b.pdf", "Business trips"), 0.2)]
    result = make_service(settings, hits, "unused").ask("Who is the CEO?")
    assert result.topics == ["Parking rules", "Business trips"]
    assert len(result.sources) == settings.top_k


def test_distinct_topics_are_unique_ordered_and_limited():
    hits = [(doc(topic=t), 0.5) for t in ["A", "B", "A", "C", None, "D", "E", "F"]]
    assert distinct_topics(hits, limit=5) == ["A", "B", "C", "D", "E"]


def test_empty_knowledge_base_message():
    assert "жодного документа" in no_answer_message("uk", [])


def test_detect_language():
    assert detect_language("Скільки днів відпустки?") == "uk"
    assert detect_language("How many vacation days?") == "en"
