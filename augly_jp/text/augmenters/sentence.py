from nlpaug.augmenter.sentence import Augmenter
from nlpaug.util import Action, Method


class BackTranslationAugmenter(Augmenter):
    def __init__(self):
        super().__init__(
            name="BackTranslationAugmenter",
            method=Method.SENTENCE,
            action=Action.,
            aug_min=None,
            aug_max=None,
            aug_p=None,
        )
