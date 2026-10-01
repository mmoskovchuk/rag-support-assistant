from langchain_core.messages import AIMessage

from support_rag.llm_usage import Pricing, record_usage


def test_cost_is_computed_from_reported_tokens():
    message = AIMessage(
        content="24 days",
        usage_metadata={"input_tokens": 700, "output_tokens": 40, "total_tokens": 740},
    )
    usage = record_usage(message, Pricing(input_per_1m=0.40, output_per_1m=1.60), purpose="answer")
    assert (usage.input_tokens, usage.output_tokens) == (700, 40)
    assert usage.cost_usd == 0.000344  # 700 * 0.40 / 1M + 40 * 1.60 / 1M


def test_missing_usage_metadata_returns_none():
    assert record_usage(AIMessage(content="x"), Pricing(0.4, 1.6), purpose="answer") is None
