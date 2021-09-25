import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib.request import urlretrieve

import spacy
import torch
from fugashi import Tagger
from nlpaug.model.lang_models import LanguageModels
from spacy.tokens import Doc
from tenacity import retry, retry_if_exception_message, retry_if_exception_type, stop_after_attempt
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizer

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


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class MtTransformers(LanguageModels):
    """TODO: remove this class. The details are as follows:
    Import https://github.com/makcedward/nlpaug/blob/5480074c61/nlpaug/model/lang_models/machine_translation_transformers.py
    Is is not necessary when nlpaug bump version ^1.1.5 in Augly.
    """

    def __init__(
        self,
        src_model_name: str,
        tgt_model_name: str,
        device: torch.device,
        silence: bool = True,
        batch_size: int = 32,
        max_length: int = None,
    ) -> None:
        super().__init__(device=device, silence=silence)
        self.src_model_name = src_model_name
        self.tgt_model_name = tgt_model_name
        self.src_model = AutoModelForSeq2SeqLM.from_pretrained(self.src_model_name)
        self.src_model.eval()
        self.src_model.to(device)
        self.tgt_model = AutoModelForSeq2SeqLM.from_pretrained(self.tgt_model_name)
        self.tgt_model.eval()
        self.tgt_model.to(device)
        self.src_tokenizer = AutoTokenizer.from_pretrained(self.src_model_name)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(self.tgt_model_name)
        self.batch_size = batch_size
        self.max_length = max_length

    def predict(self, texts: Texts, target_word: str = None, n: int = 1) -> List[str]:
        src_translated_texts = self.translate_one_step_batched(texts, self.src_tokenizer, self.src_model)
        tgt_translated_texts = self.translate_one_step_batched(src_translated_texts, self.tgt_tokenizer, self.tgt_model)
        return tgt_translated_texts

    def translate_one_step_batched(
        self, texts: Union[Texts, List[Texts]], tokenizer: PreTrainedTokenizer, model: AutoModel
    ) -> List[str]:
        tokenized_texts = tokenizer(texts, padding=True, return_tensors="pt")
        tokenized_ds = DataLoader(TensorDataset(*tokenized_texts.values()), batch_size=self.batch_size, shuffle=False)
        all_translated_ids = []
        with torch.no_grad():
            for batch in tokenized_ds:
                input_ids, attention_mask = tuple(text.to(self.device) for text in batch)
                translated_ids_batch = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, max_length=self.max_length
                )
                all_translated_ids.append(translated_ids_batch.detach().cpu().numpy())

        all_translated_texts = []
        for translated_ids_batch in all_translated_ids:
            translated_texts = tokenizer.batch_decode(translated_ids_batch, skip_special_tokens=True)
            all_translated_texts += translated_texts

        return all_translated_texts


BACK_TRANSLATION_MODELS = {}


def init_backtranslation_model(
    from_model_name: str,
    to_model_name: str,
    device: torch.device = None,
    force_reload: bool = False,
    batch_size: int = 32,
    max_length: int = None,
) -> MtTransformers:
    """ref. https://github.com/makcedward/nlpaug/blob/master/nlpaug/augmenter/word/back_translation.py"""
    global BACK_TRANSLATION_MODELS
    device = device or get_torch_device()

    model_name = "_".join([from_model_name, to_model_name, str(device)])
    if model_name in BACK_TRANSLATION_MODELS and not force_reload:
        BACK_TRANSLATION_MODELS[model_name].batch_size = batch_size
        BACK_TRANSLATION_MODELS[model_name].max_length = max_length
        return BACK_TRANSLATION_MODELS[model_name]

    model = MtTransformers(
        src_model_name=from_model_name,
        tgt_model_name=to_model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    BACK_TRANSLATION_MODELS[model_name] = model
    return model
