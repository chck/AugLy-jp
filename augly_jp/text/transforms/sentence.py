from typing import Any, Dict, List, Optional, Union

from augly.text.transforms import BaseTransform
from augly_jp.text import functional as F


class ReplaceBackTranslationSentences(BaseTransform):
    def __init__(
        self,
        aug_p: float = 0.3,
        aug_min: int = 1,
        aug_max: int = 1000,
        n: int = 1,
        p: float = 1.0,
        num_thread: int = 1,
        model: str = "Helsinki-NLP",
        seed: int = None,
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
        sentences: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        return F.replace_backtranslation_sentences(
            sentences,
            aug_p=self.aug_p,
            aug_min=self.aug_min,
            aug_max=self.aug_max,
            n=self.n,
            num_thread=self.num_thread,
            model=self.model,
            metadata=metadata,
            seed=self.seed,
        )
