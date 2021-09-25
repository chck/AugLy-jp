import torch
from nlpaug.augmenter.sentence import Augmenter
from nlpaug.util import Action, Method
from transformers import set_seed

from augly_jp.text.augmenters.utils import Texts, init_backtranslation_model, MtTransformers, detokenize, replace_punctuation


class BackTranslationAugmenterS(Augmenter):
    def __init__(
        self,
        aug_min: int,
        aug_max: int,
        aug_p: float,
        model: str = "Helsinki-NLP",
        seed: int = None,
    ) -> None:
        super().__init__(
            name="BackTranslationAugmenterS",
            action=Action.SUBSTITUTE,
            method=Method.WORD,
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
        )
        from transformers import logging as tlog

        # https://github.com/huggingface/transformers/issues/5421#issuecomment-698778663
        # https://github.com/huggingface/transformers/issues/3050
        tlog.set_verbosity_error()
        if model != "Helsinki-NLP":
            raise NotImplementedError("BackTranslationAugmenter's model supports only `Helsinki-NLP`.")
        if seed:
            set_seed(seed)
        """you can choose Japanese translation model from:
        https://huggingface.co/models?pipeline_tag=translation&language=ja
        """

        self.model = self.get_model(
            from_model_name="Helsinki-NLP/opus-mt-ja-en", to_model_name="Helsinki-NLP/opus-mt-en-jap"
        )

    @classmethod
    def clean(cls, data: Texts) -> Texts:
        if isinstance(data, list):
            return [d.strip() for d in data]
        return data.strip()

    def substitute(self, data: Texts) -> str:
        if not data:
            return data
        return replace_punctuation(detokenize(self.model.predict(data)).replace(" ", ""))

    @classmethod
    def get_model(
        cls,
        from_model_name: str,
        to_model_name: str,
        device: torch.device = None,
        force_reload: bool = False,
        batch_size: int = 32,
        max_length: int = None,
    ) -> MtTransformers:
        return init_backtranslation_model(from_model_name, to_model_name, device, force_reload, batch_size, max_length)
