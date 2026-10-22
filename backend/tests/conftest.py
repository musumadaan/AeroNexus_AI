import os
import sys
from pathlib import Path

os.environ.setdefault("AERONEXUS_STATE_DB", ":memory:")  # tests never touch the durable state file
# Tests are offline and deterministic even if backend/.env selects Groq: never spend free-tier quota in CI.
os.environ["AERONEXUS_LLM_PROVIDER"] = "mock"
os.environ["AERONEXUS_JUDGE_PROVIDER"] = "mock"
for k in ("AERONEXUS_SMALL_MODEL", "AERONEXUS_LARGE_MODEL", "AERONEXUS_JUDGE_MODEL"):
    os.environ.pop(k, None)

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def fresh_state():
    """Each test gets fresh singletons (approval queue, caches) so tests are order-independent."""
    import aeronexus.llm as llm
    from aeronexus.observability import tracer
    from aeronexus.services import reset_services

    reset_services()
    llm._router = None
    llm._judge = None
    tracer.reset()
    yield
