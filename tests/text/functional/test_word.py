import random
from typing import Dict, Any

import pytest

from augly_jp import text as txtaugs


@pytest.fixture
def get_data() -> Dict[str, Any]:
    random.seed(2021)
    return dict(text="あらゆる現実をすべて自分のほうへねじ曲げたのだ")


def test_replace_synonym_words(get_data):
    augmented_synonym_word = txtaugs.replace_synonym_words(get_data["text"])
    assert augmented_synonym_word == "あらゆる現実をすべて自分自身のほうへねじ曲げたのだ"


if __name__ == '__main__':
    pytest.main([__file__])
