from typing import Any, Dict, List, Optional

from augly.text import utils as txtutils
from augly_jp.text import augmenters as a
from augly_jp.text.augmenters.utils import Texts


def replace_backtranslation_sentences(
    sentences: List[Texts],
    aug_p: float = 0.3,
    aug_min: int = 1,
    aug_max: int = 1000,
    n: int = 1,
    model: str = "Helsinki-NLP",
    num_thread: int = 1,
    metadata: Optional[List[Dict[str, Any]]] = None,
    seed: int = None,
) -> List[Texts]:
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    bt_aug = a.BackTranslationAugmenterS(aug_min, aug_max, aug_p, model, seed)
    aug_texts = bt_aug.augment(sentences, n, num_thread)

    txtutils.get_metadata(
        metadata=metadata, function_name="replace_backtranslation_sentences", aug_texts=aug_texts, **func_kwargs
    )

    return aug_texts
