import pytest

from support_rag.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(
        api_key="test",
        openai_api_key="test",
        data_dir=tmp_path,
        file_stable_seconds=0.1,
        file_stable_timeout=5,
        relevance_threshold=0.5,
        _env_file=None,  # tests must not depend on a developer's local .env
    )
    for d in (s.inbox_dir, s.processing_dir, s.processed_dir, s.failed_dir):
        d.mkdir(parents=True)
    return s
