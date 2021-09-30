import random
from typing import Dict, Any

import pytest

from augly_jp import text as txtaugs


@pytest.fixture
def get_data() -> Dict[str, Any]:
    seed = 2021
    random.seed(seed)
    return dict(sentences=[
        "あらゆる現実をすべて自分のほうへねじ曲げたのだ",
        "コテージのサウナに入った。\\n一人で独占して、湖にもチャレンジ。",
        "焼肉ライクはじめて行ってみたけど美味しくてコスパ良くて気に入った！！また行こ",
        "急遽、午後らお休みもらって支払い諸々、手続き諸々。とりあえず、今日の目的は達成。",
    ], seed=seed)


def test_replace_backtranslation_sentences(get_data):
    augmented = txtaugs.replace_backtranslation_sentences(get_data["sentences"], aug_p=0.8, seed=get_data["seed"])
    assert augmented == [
        'そして、ほかの人たちをそれぞれの道に安置しておられた。',
        'わたしは汗を手にとって、かおりを脱ぎ、それを吹いているのに寄港した。そして、良いと思われる湖の池に行き、',
        'わたしは、まずそこにおり、みごとな除いたことがある。わたしのように良い事をせよ。',
        'あなたがたが、安息を去ろうとして、休みながらに行くとき、そのあいさつや調査を払いなさい。きょう、あなたがたの招かれている。',
    ]


if __name__ == '__main__':
    pytest.main([__file__])
