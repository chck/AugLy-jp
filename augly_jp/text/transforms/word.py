from typing import Any, Dict, List, Optional, Union

from augly.text.transforms import BaseTransform

from augly_jp.text import functional as F


class ReplaceSynonymWords(BaseTransform):
    def __init__(
        self, aug_p: float = 0.3, aug_min: int = 1, aug_max: int = 1000, n: int = 1, p: float = 1.0, num_thread: int = 1
    ):
        super().__init__(p)
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.n = n
        self.num_thread = num_thread

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **aug_kwargs: Any,
    ) -> Union[str, List[str]]:
        return F.replace_synonym_words(texts, metadata=metadata, **aug_kwargs)


class ReplaceWordEmbsWords(BaseTransform):
    def __init__(
        self, aug_p: float = 0.3, aug_min: int = 1, aug_max: int = 1000, n: int = 1, p: float = 1.0, num_thread: int = 1
    ):
        super().__init__(p)
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.n = n
        self.num_thread = num_thread

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **aug_kwargs: Any,
    ) -> Union[str, List[str]]:
        return F.replace_wordembs_words(texts, metadata=metadata, **aug_kwargs)


class ReplaceFillMaskWords(BaseTransform):
    def __init__(
        self,
        aug_p: float = 0.3,
        aug_min: int = 1,
        aug_max: int = 1000,
        n: int = 1,
        p: float = 1.0,
        num_thread: int = 1,
        model: str = "cl-tohoku/bert-base-japanese-v2",
        seed: Optional[int] = None,
    ):
        super().__init__(p)
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.n = n
        self.num_thread = num_thread
        self.model = model
        self.seed = seed

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **aug_kwargs: Any,
    ) -> Union[str, List[str]]:
        return F.replace_fillmask_words(texts, metadata=metadata, **aug_kwargs)
