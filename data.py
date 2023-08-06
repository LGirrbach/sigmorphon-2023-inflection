import os
import re
import torch
import regex
import pandas as pd

from typing import List
from typing import Tuple
from typing import Optional
from containers import Batch
from functools import partial
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from torchtext.vocab import build_vocab_from_iterator

Sequence = List[str]
SequencePair = Tuple[Sequence, Sequence]
Dataset = List[SequencePair]

kanji_regex = regex.compile(r"\p{IsHan}", regex.UNICODE)


def dekanjify(source: Sequence) -> Sequence:
    dekanjified_source = []

    for character in source:
        if regex.match(kanji_regex, character) is not None:
            dekanjified_source.append("K")
        else:
            dekanjified_source.append(character)

    return dekanjified_source


def _batch_collate(
    batch,
    source_tokenizer: Vocab,
    target_tokenizer: Vocab,
    sos_token: str = "[SOS]",
    eos_token: str = "[EOS]",
) -> Batch:
    sources, targets = zip(*batch)

    # Encode sources
    sources = [[sos_token] + source + [eos_token] for source in sources]
    sources = [source_tokenizer(source) for source in sources]

    source_tensor = [torch.tensor(source).long() for source in sources]
    source_tensor = pad_sequence(source_tensor, batch_first=True, padding_value=0)
    source_length = torch.tensor([len(source) for source in sources]).long()

    # Encode targets
    targets = [[sos_token] + target + [eos_token] for target in targets]
    targets = [target_tokenizer(target) for target in targets]

    target_tensor = [torch.tensor(target).long() for target in targets]
    target_tensor = pad_sequence(target_tensor, batch_first=True, padding_value=0)
    target_length = torch.tensor([len(target) for target in targets]).long()

    return Batch(
        source=source_tensor,
        target=target_tensor,
        source_length=source_length,
        target_length=target_length,
    )


class Seq2SeqDataModule(LightningDataModule):
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

    def __init__(
        self,
        from_files: bool = True,
        train_path: Optional[str] = None,
        dev_path: Optional[str] = None,
        test_path: Optional[str] = None,
        train_data: Optional[Dataset] = None,
        dev_data: Optional[Dataset] = None,
        test_data: Optional[Dataset] = None,
        batch_size: int = 32,
    ):
        super().__init__()

        self.from_files = from_files
        self.train_file_path = train_path
        self.dev_file_path = dev_path
        self.test_file_path = test_path
        self.batch_size = batch_size

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self._check_arguments()

        self.source_alphabet = None
        self.target_alphabet = None

        self.source_alphabet_size = None
        self.target_alphabet_size = None

        self.source_tokenizer = None
        self.target_tokenizer = None
        self._batch_collate = None

    def _check_arguments(self):
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        for path in [self.train_file_path, self.dev_file_path, self.test_file_path]:
            if path is not None:
                assert os.path.exists(path), f"Path {path} does not exist"

        for dataset in [self.train_data, self.dev_data, self.test_data]:
            if dataset is not None:
                assert isinstance(dataset, list)
                for datapoint in dataset:
                    assert isinstance(datapoint, tuple)
                    source, target = datapoint
                    assert isinstance(source, list) and all(
                        isinstance(symbol, str) for symbol in source
                    )
                    assert isinstance(target, list) and all(
                        isinstance(symbol, str) for symbol in target
                    )

    @classmethod
    def from_files(
        cls,
        train_path: str,
        dev_path: str,
        test_path: Optional[str],
        batch_size: int = 32,
    ):
        return cls(
            from_files=True,
            train_path=train_path,
            dev_path=dev_path,
            test_path=test_path,
            batch_size=batch_size,
        )

    @classmethod
    def from_data(
        cls,
        train_data: Dataset,
        dev_data: Dataset = None,
        test_data: Optional[Dataset] = None,
        batch_size: int = 32,
    ):
        return cls(
            from_files=False,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            batch_size=batch_size,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.from_files:
                self.train_data = self.load_file(self.train_file_path)
                self.dev_data = self.load_file(self.dev_file_path)

            self.source_alphabet = list(
                sorted(set.union(*(set(source) for source, _ in self.train_data)))
            )
            self.target_alphabet = list(
                sorted(set.union(*(set(target) for _, target in self.train_data)))
            )

            self.source_alphabet_size = len(self.source_alphabet) + 4
            self.target_alphabet_size = len(self.target_alphabet) + 4

            self.source_tokenizer = build_vocab_from_iterator(
                [[symbol] for symbol in self.source_alphabet],
                specials=self.special_tokens,
            )
            self.target_tokenizer = build_vocab_from_iterator(
                [[symbol] for symbol in self.target_alphabet],
                specials=self.special_tokens,
            )
            self.source_tokenizer.set_default_index(1)
            self.target_tokenizer.set_default_index(1)

            self._batch_collate = partial(
                _batch_collate,
                source_tokenizer=self.source_tokenizer,
                target_tokenizer=self.target_tokenizer,
            )

        if stage == "test" or stage is None:
            if self.from_files:
                self.test_data = self.load_file(self.test_file_path)

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._batch_collate,
            num_workers=6,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._batch_collate,
            num_workers=6,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._batch_collate,
            num_workers=6,
        )

    @staticmethod
    def load_file(path: str):
        raise NotImplementedError


class G2PDataModule(Seq2SeqDataModule):
    @staticmethod
    def load_file(path: str):
        with open(path) as df:
            source_target_pairs = [
                line.strip().split("\t") for line in df if line.strip()
            ]

        source_target_pairs = [
            (list(source), target.split(" ")) for source, target in source_target_pairs
        ]
        return source_target_pairs


class InflectionDataModule(Seq2SeqDataModule):
    @staticmethod
    def load_file(path: str):
        if "covered" in path:
            data = pd.read_csv(path, sep="\t", names=["lemma", "tags"])
            lemmas = data["lemma"].tolist()
            tags = data["tags"].tolist()
            forms = ["" for _ in lemmas]
        else:
            data = pd.read_csv(path, sep="\t", names=["lemma", "tags", "form"])
            lemmas = data["lemma"].tolist()
            tags = data["tags"].tolist()
            forms = data["form"].tolist()

        tags = [list(re.sub(r"[;,()]", " ", tag).split()) for tag in tags]
        sources = [tag + dekanjify(list(source)) for tag, source in zip(tags, lemmas)]
        targets = [dekanjify(list(target)) for target in forms]

        source_target_pairs = list(zip(sources, targets))

        return source_target_pairs
