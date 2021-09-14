import pytest
import random
from augly.text import Compose, OneOf
from augly_jp import text as txtaugs
from typing import Dict, Any


# TODO: commonize the function to get data for test
@pytest.fixture
def get_data() -> Dict[str, Any]:
    random.seed(2021)
    return dict(
        text="あらゆる現実をすべて自分のほうへねじ曲げたのだ",
        # meta example: https://github.com/facebookresearch/AugLy/blob/64a33f0a99/augly/utils/expected_output/text_tests/expected_metadata.json
        metadata=[
            dict(),
            dict(),
            dict(),
        ]
    )


def test_compose_word_augmenters(get_data):
    aug = Compose([
        OneOf([
            txtaugs.ReplaceSynonymWords(),
            txtaugs.ReplaceWordEmbsWords(),
        ]),
        txtaugs.ReplaceFillMaskWords(),
    ])
    augmented = aug(get_data["text"], metadata=get_data['metadata'])
    assert augmented == ["あらゆる物事をすべて自分のほうでねじ曲げたいのだ"]


if __name__ == '__main__':
    pytest.main([__file__])
