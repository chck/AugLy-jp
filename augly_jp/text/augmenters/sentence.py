from augly_jp.text.augmenters.utils import Texts, replace_punctuation
from nlpaug.augmenter.sentence import Augmenter
from nlpaug.util import Action, Method
from transformers import pipeline, set_seed


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
        self.from_model = pipeline(task="translation", model="Helsinki-NLP/opus-mt-ja-en")
        self.to_model = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-jap")

    @classmethod
    def clean(cls, data: Texts) -> Texts:
        if isinstance(data, list):
            return [d.strip() for d in data]
        return data.strip()

    def apply_back_translation(self, sentence: str) -> str:
        translated = self.from_model(sentence)[0]["translation_text"]
        return self.to_model(translated)[0]["translation_text"]

    def substitute(self, data: Texts) -> str:
        if not data:
            return data
        return replace_punctuation(self.apply_back_translation(data).replace(" ", ""))
