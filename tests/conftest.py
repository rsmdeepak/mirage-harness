import sys
from pathlib import Path

import pytest

# Make the project root importable so `import mirage` works under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirage import default_judge_bank  # noqa: E402


@pytest.fixture
def bank():
    return default_judge_bank()
