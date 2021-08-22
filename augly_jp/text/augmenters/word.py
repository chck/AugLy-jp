import random
from typing import List

from augly.text.augmenters.utils import get_aug_idxes
from chikkarpy import Chikkar
from chikkarpy.dictionarylib import Dictionary
from nlpaug.augmenter.word import Augmenter
from nlpaug.util import Action, Method

from augly_jp.text.augmenters.utils import tokenize, detokenize, Texts


class SynonymAugmenter(Augmenter):
    def __init__(self, aug_min: int, aug_max: int, aug_p: float, synonym_dic_path: str = None):
        super().__init__(name="SynonymAugmenter", action=Action.SUBSTITUTE, method=Method.WORD, aug_min=aug_min,
                         aug_max=aug_max, aug_p=aug_p)
        self.synonyms = Chikkar()
        self.synonyms.add_dictionary(Dictionary(synonym_dic_path))

    @classmethod
    def clean(cls, data: Texts) -> Texts:
        if isinstance(data, list):
            return [d.strip() for d in data]

        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset: List[str], data: Texts) -> bool:
        return data in dataset

    def apply_synonym(self, word: str) -> str:
        # TODO: includes VERB's one, now supports NOUN only.
        maybe_synonym = self.synonyms.find(word)
        return random.choice(maybe_synonym) if maybe_synonym else word

    def substitute(self, data: Texts) -> str:
        tokens = tokenize(data, with_pos=True)
        tokens_augmentable = []
        idx = 1  # not 0 start
        for token in tokens:
            if token['pos'] in {'NOUN'}:
                token["aug_word_idx"] = idx
                tokens_augmentable.append(token)
                idx += 1
            else:
                token["aug_word_idx"] = None
        results = []
        aug_word_cnt = self._generate_aug_cnt(len(tokens_augmentable), self.aug_min, self.aug_max, self.aug_p)
        aug_word_idxes = set(get_aug_idxes(self, tokens_augmentable, list(range(len(tokens_augmentable))), aug_word_cnt, Method.WORD))
        for row in tokens:
            if row["aug_word_idx"] not in aug_word_idxes:
                results.append(row['token'])
                continue
            results.append(self.apply_synonym(row['token']))
        return detokenize(results)
