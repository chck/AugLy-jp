import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib.request import urlretrieve

import spacy
from fugashi import Tagger
from spacy.tokens import Doc
from tenacity import retry, retry_if_exception_message, retry_if_exception_type, stop_after_attempt
from tqdm.auto import tqdm

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


@retry(
    retry=(
        (
            retry_if_exception_type(AttributeError) & retry_if_exception_message("EOS is not connected to BOS")
            | retry_if_exception_type(ValueError)
        )
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


def get_model(fname: str = None, origin: str = None) -> str:
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


def _extract_archive(file_path: str, path=".") -> bool:
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

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> Union[bool, None]:
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
