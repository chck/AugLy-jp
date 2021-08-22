from typing import List, Union, Dict, Any

import spacy
from spacy.tokens import Doc

nlp = spacy.load("ja_ginza")
Texts = Union[str, List[str]]
POS = {  # ref: https://universaldependencies.org/docs/u/pos/
    "ADJ",  # adjective
    "ADP",  # adposition
    "ADV",  # adverb
    "AUX",  # auxiliary verb
    "CONJ",  # coordinating conjunction
    "CCONJ",  # TODO: i dont know why but this pos tag exists.
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


def tokenize(text: str, lemmatize: bool = False, with_pos: bool = False) -> List[Union[str, Dict[str, Any]]]:
    doc: Doc = nlp(text)
    tokens, pos = [], []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in POS:
                tokens.append(token.text if not lemmatize else token.lemma_)
                pos.append(token.pos_)
    return tokens if not with_pos else [dict(token=token, pos=_pos) for token, _pos in zip(tokens, pos)]


def detokenize(tokens: List[str]) -> str:
    return "".join(tokens)
