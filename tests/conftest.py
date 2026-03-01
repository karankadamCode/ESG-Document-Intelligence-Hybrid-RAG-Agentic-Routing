import os
import pytest
from dotenv import load_dotenv
from pathlib import Path
import sys
# Add project root (parent of tests/) to sys.path so `import main` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main


@pytest.fixture(scope="session")
def env():
    load_dotenv()
    return True


@pytest.fixture(scope="session")
def vs_and_bm25(env):
    vs = main.load_vectorstore(main.VECTORSTORE_PATH, main.EMBEDDING_MODEL)
    bm25_corpus = main.load_all_chunks_for_bm25(vs)
    return vs, bm25_corpus


@pytest.fixture(scope="session")
def default_cfg():
    return main.RetrievalConfig()


@pytest.fixture(scope="session")
def llm(env):
    # Only needed for LLM tests; keep it session-scoped
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY missing; skipping LLM tests.")
    return main.init_llm(main.DEFAULT_LLM_MODEL)