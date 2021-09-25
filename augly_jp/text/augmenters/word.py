import logging
import os
import random
import warnings
from pathlib import Path
from typing import List

from augly.text.augmenters.utils import get_aug_idxes
from augly_jp.text.augmenters.utils import Texts, detokenize, get_model, tokenize, tokenize_unidic
from chikkarpy import Chikkar
from chikkarpy.dictionarylib import Dictionary

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from gensim.models import KeyedVectors

from nlpaug.augmenter.word import Augmenter
from nlpaug.util import Action, Method
from transformers import pipeline, set_seed


class SynonymAugmenter(Augmenter):
    def __init__(self, aug_min: int, aug_max: int, aug_p: float, synonym_dic_path: str = None) -> None:
        super().__init__(
            name="SynonymAugmenter",
            action=Action.SUBSTITUTE,
            method=Method.WORD,
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
        )
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
        idx = 0
        for token in tokens:
            if token["pos"] in {"NOUN"}:
                token["aug_word_idx"] = idx
                tokens_augmentable.append(token)
                idx += 1
            else:
                token["aug_word_idx"] = None
        results = []
        aug_word_cnt = self._generate_aug_cnt(len(tokens_augmentable), self.aug_min, self.aug_max, self.aug_p)
        aug_word_idxes = set(
            get_aug_idxes(self, tokens_augmentable, list(range(len(tokens_augmentable))), aug_word_cnt, Method.WORD)
        )
        for row in tokens:
            if row["aug_word_idx"] not in aug_word_idxes:
                results.append(row["token"])
                continue
            results.append(self.apply_synonym(row["token"]))
        return detokenize(results)


class WordEmbsAugmenter(Augmenter):
    def __init__(self, aug_min: int, aug_max: int, aug_p: float):
        super().__init__(
            name="WordEmbsAugmenter",
            action=Action.SUBSTITUTE,
            method=Method.WORD,
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
        )
        word_vector_path = get_model(
            origin="https://storage.googleapis.com/ailab-users/chck/models/pretrained/text/entityvector/entityvector.tar.gz"
        )
        if os.path.isdir(word_vector_path):
            maybe_word_vector_path = list(Path(word_vector_path).glob("*.bin"))
            if maybe_word_vector_path:
                word_vector_path = str(maybe_word_vector_path[0])
            else:
                raise FileNotFoundError(f"{word_vector_path} do NOT have the binary of word vector.")
        self.model = KeyedVectors.load(word_vector_path, mmap="r")
        # # no more training, only querying mode. use (much) less RAM.
        # # ref. https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec.most_similar
        # self.model.init_sims(replace=True)

    @classmethod
    def clean(cls, data: Texts) -> Texts:
        if isinstance(data, list):
            return [d.strip() for d in data]

        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset: List[str], data: Texts) -> bool:
        return data in dataset

    def apply_wordembs(self, word: str, similarity_min: float = 0.0) -> str:
        assert 0.0 <= similarity_min <= 1.0
        try:
            similar_words = [
                key for key, similarity in self.model.most_similar(word, topn=10) if similarity >= similarity_min
            ]
        except KeyError:  # not in vocabulary
            similar_words = []
        return random.choice(similar_words) if similar_words else word

    def substitute(self, data: Texts) -> str:
        _pos = {"NOUN", "SYM"}
        # lemmatization works if augmentable targets have VERB
        tokens = tokenize(data, with_pos=True, lemmatize="VERB" in _pos)
        tokens_augmentable = []
        idx = 0
        for token in tokens:
            if token["pos"] in _pos:
                token["aug_word_idx"] = idx
                tokens_augmentable.append(token)
                idx += 1
            else:
                token["aug_word_idx"] = None
        results = []
        aug_word_cnt = self._generate_aug_cnt(len(tokens_augmentable), self.aug_min, self.aug_max, self.aug_p)
        aug_word_idxes = set(
            get_aug_idxes(self, tokens_augmentable, list(range(len(tokens_augmentable))), aug_word_cnt, Method.WORD)
        )
        for row in tokens:
            if row["aug_word_idx"] not in aug_word_idxes:
                results.append(row["token"])
                continue
            results.append(self.apply_wordembs(row["token"]))
        return detokenize(results)


class FillMaskAugmenter(Augmenter):
    def __init__(
        self,
        aug_min: int,
        aug_max: int,
        aug_p: float,
        model: str = "cl-tohoku/bert-base-japanese-v2",
        seed: int = None,
    ) -> None:
        super().__init__(
            name="FillMaskAugmenter",
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
        if seed:
            set_seed(seed)
        """you can choose Japanese fill-mask model from:
        https://huggingface.co/models?pipeline_tag=fill-mask&sort=downloads&search=japanese
        """
        self.model = pipeline(task="fill-mask", model=model, top_k=5)
        self.mask_token = self.model.tokenizer.mask_token

    @classmethod
    def clean(cls, data: Texts) -> Texts:
        if isinstance(data, list):
            return [d.strip() for d in data]
        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset: List[str], data: Texts) -> bool:
        return data in dataset

    def apply_fill_mask(self, tokens: List[str]) -> str:
        assert self.mask_token in tokens
        candidates = self.model(detokenize(tokens))
        return random.choice(candidates)["sequence"].split(" ")

    def substitute(self, data: Texts) -> str:
        # Default model (cl-tohoku) expect uni-dic tokenizer
        tokens = tokenize_unidic(data)
        aug_word_cnt = self._generate_aug_cnt(len(tokens), self.aug_min, self.aug_max, self.aug_p)
        aug_word_idxes = set(get_aug_idxes(self, tokens, list(range(len(tokens))), aug_word_cnt, Method.WORD))
        for idx in aug_word_idxes:
            try:
                tokens[idx] = self.mask_token
                tokens = self.apply_fill_mask(tokens)
            except IndexError as e:
                # The higher aug_p, the more frequent IndexError occurs.
                # Because apply_fill_mask has a risk of changing token length.
                logging.warning(e)
        return detokenize(tokens)
