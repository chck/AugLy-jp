import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from urllib.request import urlretrieve

import spacy
from fugashi import Tagger  # ty: ignore[unresolved-import]
from nlpaug.util import Method
from spacy.tokens import Doc
from tenacity import retry, retry_if_exception_message, retry_if_exception_type, stop_after_attempt
from tqdm.std import tqdm

log = logging.getLogger(__name__)

nlp = spacy.load("ja_ginza_electra")
tagger = Tagger()
Texts = Union[str, List[str]]
POS = {
    # ref: https://universaldependencies.org/docs/u/pos/
    # https://yu-nix.com/blog/2021/3/3/spacy-pos-list/
    "ADJ",  # adjective
    "ADP",  # adposition
    "ADV",  # adverb
    "AUX",  # auxiliary verb
    "CONJ",  # coordinating conjunction
    "CCONJ",  # NOTE: this may duplicate CONJ, but i dont know why but this pos tag exists.
    "DET",  # determiner
    "INTJ",  # interjection
    "NOUN",  # noun
    "NUM",  # numeral
    "PART",  # particle
    "PRON",  # pronoun
    "PROPN",  # proper noun
    "PUNCT",  # punctuation
    "SCONJ",  # subordinating conjunction
    "SYM",  # symbol
    "VERB",  # verb
    "X",  # other
}


def calculate_aug_count(size: int, aug_min: int, aug_max: int, aug_p: float) -> int:
    """Preserve nlpaug 1.1.3's floor-based augmentation count."""
    count = int(aug_p * size)
    if count < aug_min:
        return aug_min
    if count > aug_max:
        return aug_max
    return count


def select_aug_indices(
    augmenter: Any,
    tokens: Sequence[Any],
    filtered_indices: List[int],
    aug_count: int,
    mode: str,
    min_char: Union[int, None] = None,
) -> List[int]:
    """Preserve the token selection contract used by AugLy 0.1.7."""
    if mode not in Method.getall():
        raise ValueError("mode must be a value defined by nlpaug.util.Method")

    priority_indices = []
    priority_words = getattr(augmenter, "priority_words", None)
    if mode == Method.WORD and priority_words is not None:
        priority_words_set = set(priority_words)
        for index, token in enumerate(tokens):
            if token in priority_words_set and (min_char is None or len(token) >= min_char):
                priority_indices.append(index)

    indices = [
        index
        for index in filtered_indices
        if index not in priority_indices and (min_char is None or len(tokens[index]) >= min_char)
    ]
    if not priority_indices and not indices:
        return []
    if len(priority_indices) <= aug_count:
        aug_indices = priority_indices
        remaining_count = min(aug_count - len(priority_indices), len(indices))
        aug_indices += augmenter.sample(indices, remaining_count)
        return aug_indices
    return augmenter.sample(priority_indices, aug_count)


def normalize_augmented_texts(source: Texts, augmented: List[str], n: int) -> Texts:
    """Keep the scalar return contract from nlpaug 1.1.3."""
    if isinstance(source, str) and n == 1:
        return augmented[0] if augmented else source
    return augmented


@retry(
    retry=(
        retry_if_exception_type(AttributeError) & retry_if_exception_message("EOS is not connected to BOS")
        | retry_if_exception_type(ValueError)
    ),
    stop=stop_after_attempt(5),
)
def tokenize(text: str, lemmatize: bool = False, with_pos: bool = False) -> List[Union[str, Dict[str, Any]]]:
    doc: Doc = nlp(text)
    tokens, pos = [], []
    for sentences in doc.sents:
        for token in sentences:
            if token.pos_ in POS:
                tokens.append(token.text if not lemmatize else token.lemma_)
                pos.append(token.pos_)
    return tokens if not with_pos else [dict(token=token, pos=_pos) for token, _pos in zip(tokens, pos)]


def tokenize_unidic(text: str, lemmatize: bool = False) -> List[str]:
    """TODO: merge one class all in tokenizer such as ginza and fugashi"""
    tokens = tagger(text)
    results = []
    for token in tokens:
        results.append(token.feature.orth if not lemmatize else token.feature.lemma)
    return results


def detokenize(tokens: List[str]) -> str:
    return "".join([token for token in tokens if token])


def replace_punctuation(text_en: str) -> str:
    text_en = text_en.replace(",", "、")
    text_en = text_en.replace(".", "。")
    return text_en


def get_model(fname: Union[str, None] = None, origin: Union[str, None] = None) -> str:
    """inspired: gensim.downloader.load() and tf.keras.utils.get_file
    TODO: support the file type except tar.gz
    """
    if origin is None:
        raise ValueError('Please specify the "origin" argument (URL of the file to download).')

    data_dir = os.path.join(os.path.expanduser("~"), "gensim-data")
    os.makedirs(data_dir, exist_ok=True)
    if not fname:
        # The 2 `.with_suffix()` are because of `.tar.gz` as pathlib considers it as 2 suffixes.
        fname = Path(origin).with_suffix("").with_suffix("").name
        untar_fpath = os.path.join(data_dir, fname)
        fpath = f"{untar_fpath}.tar.gz"
    else:
        fpath = os.path.join(data_dir, fname)
        untar_fpath = fpath

    if not os.path.exists(fpath):
        log.info(f"Downloading data from {origin}")

        with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=fname) as t:
            urlretrieve(origin, filename=fpath, reporthook=t.update_to)

        _extract_archive(fpath, data_dir)

    return untar_fpath


def _extract_archive(file_path: str, path: str = ".") -> bool:
    """TODO: support the file type except tar.gz"""
    open_fn = tarfile.open
    is_match_fn = tarfile.is_tarfile

    if is_match_fn(file_path):
        with open_fn(file_path) as archive:
            try:
                archive.extractall(path)
            except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                raise
        return True
    return False


class TqdmUpTo(tqdm):
    """ref: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: Union[int, None] = None) -> Union[bool, None]:
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize
