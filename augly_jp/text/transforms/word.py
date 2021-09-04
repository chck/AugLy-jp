from typing import Any, Dict, List, Optional, Union

from augly.text.transforms import BaseTransform
from augly_jp.text import functional as F


class ReplaceSynonymWords(BaseTransform):
    def __init__(self, aug_p: float = 0.3, aug_min: int = 1, aug_max: int = 1000, n: int = 1, p: float = 1.0):
        super().__init__(p)
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.n = n

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        return F.replace_synonym_words(
            texts, aug_p=self.aug_p, aug_min=self.aug_min, aug_max=self.aug_max, n=self.n, metadata=metadata
        )


class ReplaceWordEmbsWords(BaseTransform):
    def __init__(self, aug_p: float = 0.3, aug_min: int = 1, aug_max: int = 1000, n: int = 1, p: float = 1.0):
        super().__init__(p)
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.n = n

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        return F.replace_wordembs_words(
            texts, aug_p=self.aug_p, aug_min=self.aug_min, aug_max=self.aug_max, n=self.n, metadata=metadata
        )


class ReplaceFillMaskWords(BaseTransform):
    def __init__(
        self,
        aug_p: float = 0.3,
        aug_min: int = 1,
        aug_max: int = 1000,
        n: int = 1,
        p: float = 1.0,
        model: str = "cl-tohoku/bert-base-japanese-v2",
    ):
        super().__init__(p)
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.n = n
        self.model = model

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        return F.replace_fillmask_words(
            texts,
            aug_p=self.aug_p,
            aug_min=self.aug_min,
            aug_max=self.aug_max,
            n=self.n,
            model=self.model,
            metadata=metadata,
        )
