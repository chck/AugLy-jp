from pathlib import Path

import numpy as np
import pytest
from chikkarpy import config as chikkarpy_config
from chikkarpy.command_line import build_dictionary
from gensim.models import KeyedVectors

from augly_jp.text.augmenters import word as word_augmenters


@pytest.fixture(autouse=True)
def local_word_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synonym_csv_path = tmp_path / "synonyms.csv"
    synonym_csv_path.write_text(
        "000001,1,0,1,0,0,0,(),自分,,\n"
        "000001,1,0,2,0,0,0,(),自身,,\n",
        encoding="utf-8",
    )
    synonym_dictionary_path = tmp_path / "system_synonym.dic"
    build_dictionary(str(synonym_csv_path), str(synonym_dictionary_path), "test dictionary")
    monkeypatch.setattr(chikkarpy_config, "DEFAULT_RESOURCEDIR", str(tmp_path))

    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(
        ["自分", "関心"],
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    word_vector_path = tmp_path / "word-vectors.bin"
    vectors.save(str(word_vector_path))
    monkeypatch.setattr(word_augmenters, "get_model", lambda origin: str(word_vector_path))
