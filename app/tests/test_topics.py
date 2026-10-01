from langchain_core.language_models import FakeListChatModel

from support_rag.topics import TopicNamer, topic_from_filename


class BrokenLLM(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise ConnectionError("OpenAI is down")


def test_topic_is_cleaned_from_quotes_and_period():
    namer = TopicNamer(FakeListChatModel(responses=["«Порядок надання відпустки»."]), max_chars=100)
    assert namer.name("text", "scan_001.pdf") == "Порядок надання відпустки"


def test_falls_back_to_file_name_when_llm_fails():
    namer = TopicNamer(BrokenLLM(responses=["unused"]), max_chars=100)
    assert namer.name("text", "vacation_policy-2026.pdf") == "Vacation policy 2026"


def test_topic_from_filename():
    assert topic_from_filename("business-trips.docx") == "Business trips"
