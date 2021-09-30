import random
from typing import Dict, Any

import pytest
from augly.text import Compose

from augly_jp import text as txtaugs


# TODO: commonize the function to get data for test
@pytest.fixture
def get_data() -> Dict[str, Any]:
    random.seed(2021)
    return dict(
        text="あらゆる現実をすべて自分のほうへねじ曲げたのだ",
        metadata=None,
    )


def test_sentence_augmenters(get_data):
    aug = Compose([
        txtaugs.ReplaceBackTranslationSentences(),
    ])
    augmented = aug(get_data["text"], metadata=get_data['metadata'])
    assert augmented == ["そして、ほかの人たちをそれぞれの道に安置しておられた。"]


if __name__ == '__main__':
    pytest.main([__file__])
