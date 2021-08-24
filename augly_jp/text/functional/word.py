from typing import Any, Dict, List, Optional

from augly.text import utils as txtutils
from augly_jp.text import augmenters as a
from augly_jp.text.augmenters.utils import Texts


def replace_synonym_words(
    texts: Texts,
    aug_p: float = 0.3,
    aug_min: int = 1,
    aug_max: int = 1000,
    n: int = 1,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    synonym_aug = a.SynonymAugmenter(aug_min, aug_max, aug_p)
    aug_texts = synonym_aug.augment(texts, n)

    txtutils.get_metadata(metadata=metadata, function_name="replace_synonym_words", aug_texts=aug_texts, **func_kwargs)

    return aug_texts


def replace_wordembedding_words(
    texts: Texts,
    aug_p: float = 0.3,
    aug_min: int = 1,
    aug_max: int = 1000,
    n: int = 1,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    we_aug = a.WordEmbeddingAugmenter(aug_min, aug_max, aug_p)
    aug_texts = we_aug.augment(texts, n)

    txtutils.get_metadata(
        metadata=metadata, function_name="replace_wordembedding_words", aug_texts=aug_texts, **func_kwargs
    )

    return aug_texts
