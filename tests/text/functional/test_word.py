import random
from typing import Dict, Any

import pytest

from augly_jp import text as txtaugs


@pytest.fixture
def get_data() -> Dict[str, Any]:
    seed = 2021
    random.seed(seed)
    return dict(text="あらゆる現実をすべて自分のほうへねじ曲げたのだ", seed=seed)


def test_replace_synonym_words(get_data):
    augmented = txtaugs.replace_synonym_words(get_data["text"], aug_p=0.8)
    assert augmented == "あらゆる現実をすべて自身のほうへねじ曲げたのだ"


def test_replace_wordembs_words(get_data):
    augmented = txtaugs.replace_wordembs_words(get_data["text"], aug_p=0.8)
    assert augmented == "あらゆる現実をすべて関心のほうへねじ曲げたのだ"


def test_replace_fillmask_words(get_data):
    augmented = txtaugs.replace_fillmask_words(get_data["text"], aug_p=0.8, seed=get_data["seed"])
    assert augmented == "つまり現実を、未来な未来まで変えたいんだ"

    texts = [
        'コテージのサウナに入った。\\n一人で独占して、湖にもチャレンジ。',
        '焼肉ライクはじめて行ってみたけど美味しくてコスパ良くて気に入った！！また行こ',
        '急遽、午後らお休みもらって支払い諸々、手続き諸々。とりあえず、今日の目的は達成。'
    ]
    augmented_seq = txtaugs.replace_fillmask_words(texts, aug_p=0.8, seed=get_data["seed"])
    assert augmented_seq == [
        "ホテルのサウナに通ってる。温泉が湧き、釣り大会が出来る.",
        "焼肉店まで行ってみた!美味しくもして気がしないのに",
        "急遽もなんでやら金もっての出発となりましょう、と、今日の約束はない、",
    ]


if __name__ == '__main__':
    pytest.main([__file__])
