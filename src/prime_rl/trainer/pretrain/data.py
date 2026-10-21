import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.pretrain import MetadataConfig, PretrainDataConfig, TokenSourceConfig
from prime_rl.trainer.sft.data import StatefulIterableDataset
from prime_rl.utils.logger import get_logger


class TokenFiles:
    """Token files read as one stream: ``files`` in order, each little-endian with the dtype recorded on the first
    line of its datatrove ``.metadata`` (or ``dtype``)."""

    def __init__(self, files: list[Path], dtype: np.dtype | None = None):
        self.files = files
        self.dtypes = [dtype or token_dtype(file) for file in files]
        sizes = [file.stat().st_size // dtype.itemsize for file, dtype in zip(files, self.dtypes)]
        self.starts = np.cumsum([0, *sizes])
        self.fds: tuple[int, dict[int, int]] = (os.getpid(), {})

    def read(self, start: int, length: int) -> np.ndarray:
        """Tokens ``[start, start + length)`` of the stream, read with ``os.pread`` (which releases the GIL, so threads
        read in parallel)."""
        if self.fds[0] != os.getpid():  # opened per process: dataloader workers are forked
            self.fds = (os.getpid(), {})
        pieces = []
        while length > 0:
            index = int(np.searchsorted(self.starts, start, side="right")) - 1
            size = min(length, int(self.starts[index + 1]) - start)
            if index not in self.fds[1]:
                self.fds[1][index] = os.open(self.files[index], os.O_RDONLY)
            dtype = self.dtypes[index]
            offset = (start - int(self.starts[index])) * dtype.itemsize
            pieces.append(np.frombuffer(os.pread(self.fds[1][index], size * dtype.itemsize, offset), dtype=dtype))
            start += size
            length -= size
        return np.concatenate(pieces).astype(np.int64)


def document_ends(files: TokenFiles) -> np.ndarray:
    """End offset in the stream of every document of the files, from their ``.ds.index`` files."""
    ends = [
        np.fromfile(f"{file}.index", dtype="<u8").astype(np.int64) + start
        for file, start in zip(files.files, files.starts)
    ]
    return np.concatenate(ends)


@dataclass
class EpochLayout:
    """The rows of one epoch of a source: its documents in ``order`` (None: file order), packed back to back with
    their prefixes into one virtual stream (``ends``: end of every document in it). Row ``r`` holds the virtual tokens
    ``[row_starts[r], row_starts[r + 1]]`` (one past the row for the last target), after the prefix of the document
    it continues, so that every row has ``seq_len`` tokens."""

    order: np.ndarray | None
    ends: np.ndarray
    prefix_lens: np.ndarray | None
    row_starts: np.ndarray
    read_order: np.ndarray | None
    """The rows in reading order (None: in order)."""
    field_mask: np.ndarray | None = None
    """The fields in each document's prefix (file order; bit ``i`` = field ``i``; None: all fields)."""


class TokenShards:
    """Datatrove tokenized shards in one folder: the documents of the files in name order, read as rows of
    ``seq_len`` tokens.

    ``<name>.ds`` holds the token ids (little-endian, 2 or 4 bytes as recorded on the first line of
    ``<name>.ds.metadata``) and ``<name>.ds.index`` the uint64 end offset of every document.

    Every epoch packs all documents back to back into rows; a document that does not fit continues in the next row,
    and the last partial row of an epoch is skipped. With a ``seed``, each epoch packs the documents in a new seeded
    random order and reads its rows in a seeded random order over the whole epoch (so the pieces of a long document
    are spread over the epoch); part of an epoch is a sample of the whole source. Without one (held-out sets),
    documents and rows are in file order, which reads the files as one token stream.

    With ``metadata``, every row piece of a document in an epoch with prefixes starts with the document's prefix:
    the tokens of its sidecar ``fields``. Rows with prefixes are packed sequentially, which takes a few seconds per
    epoch for millions of rows."""

    READ_THREADS = 16

    def __init__(
        self, config: TokenSourceConfig, seed: tuple[int, str] | None = None, metadata: MetadataConfig | None = None
    ):
        files = sorted(config.path.glob("*.ds"))
        if not files:
            raise FileNotFoundError(f"No datatrove .ds files in {config.path}")
        self.content = TokenFiles(files)
        ends = document_ends(self.content)
        if config.max_tokens is not None and config.max_tokens > ends[-1]:
            raise ValueError(f"{config.path} has {ends[-1]:,} tokens, fewer than max_tokens={config.max_tokens:,}")
        num_docs = len(ends) if config.max_tokens is None else int(np.searchsorted(ends, config.max_tokens, "right"))
        self.doc_ends = ends[:num_docs]
        self.doc_lens = np.diff(self.doc_ends, prepend=0)
        self.num_tokens = int(self.doc_ends[-1])
        self.seed = None if seed is None else [seed[0], zlib.crc32(seed[1].encode())]

        self.metadata = metadata
        self.fields: list[tuple[TokenFiles, np.ndarray]] = []
        self.prefix_lens = None
        if metadata is not None:
            for field in metadata.fields:
                field_files = TokenFiles(
                    [config.metadata / f"{file.stem}.{field}.ds" for file in files], np.dtype("<u4")
                )
                self.fields.append((field_files, document_ends(field_files)[:num_docs]))
            self.prefix_lens = sum(np.diff(field_ends, prepend=0) for _, field_ends in self.fields)
        self.prefix_end_row: int | None = None
        """Rows of the source (over all epochs) before the metadata-free end (``metadata.end_step``)."""

        self.epoch_rows: dict[int, int] = {}
        self.layout: tuple[int | None, EpochLayout | None] = (None, None)
        self.pool: tuple[int | None, ThreadPoolExecutor | None] = (None, None)

    def prefixed(self, epoch: int) -> bool:
        return self.metadata is not None and (self.metadata.epochs is None or epoch in self.metadata.epochs)

    def epoch_layout(self, epoch: int, rows_before: int, seq_len: int) -> EpochLayout:
        if self.layout[0] == epoch:
            return self.layout[1]
        order = None
        if self.seed is not None:
            order = np.random.default_rng([*self.seed, epoch]).permutation(len(self.doc_lens)).astype(np.uint32)
        lens = self.doc_lens if order is None else self.doc_lens[order]
        prefixed, switch_row = self.prefixed(epoch), None
        if prefixed and self.prefix_end_row is not None:
            switch_row = self.prefix_end_row - rows_before
            prefixed = switch_row > 0
        if not prefixed:
            prefix_lens = field_mask = None
            ends = np.cumsum(lens)
            rows = (int(ends[-1]) - 1) // seq_len
            row_starts = np.arange(rows + 1, dtype=np.int64) * seq_len
        else:
            prefix_lens, field_mask = self.epoch_prefixes(epoch)
            if order is not None:
                prefix_lens = prefix_lens[order]
            if prefix_lens.max() >= seq_len:
                raise ValueError(f"A metadata prefix of {prefix_lens.max()} tokens does not fit in seq_len={seq_len}")
            ends = np.cumsum(lens + prefix_lens)
            row_starts, position = [0], 0
            while True:
                doc = int(np.searchsorted(ends, position, side="right"))
                doc_start = int(ends[doc - 1]) if doc else 0
                if len(row_starts) - 1 == switch_row:
                    # The documents from this row on lose their prefixes, except the one it continues.
                    first = doc + 1 if position > doc_start else doc
                    prefix_lens = prefix_lens.copy()
                    prefix_lens[first:] = 0
                    ends = ends.copy()
                    ends[first:] = (int(ends[first - 1]) if first else 0) + np.cumsum(lens[first:])
                # A row that starts inside a document begins with its prefix: the whole prefix, or the part of it
                # before the row start when the row starts inside the prefix.
                position += seq_len - min(position - doc_start, int(prefix_lens[doc]))
                if position >= ends[-1]:
                    break
                row_starts.append(position)
            row_starts = np.array(row_starts, dtype=np.int64)
        read_order = None
        if self.seed is not None:
            # Rows with prefixes are read before the rows of the metadata-free end, each part in a random order.
            rows = len(row_starts) - 1
            split = min(switch_row, rows) if prefixed and switch_row is not None else rows
            rng = np.random.default_rng([*self.seed, epoch, 1])
            read_order = np.concatenate([rng.permutation(split), split + rng.permutation(rows - split)]).astype(
                np.uint32
            )
        layout = EpochLayout(order, ends, prefix_lens, row_starts, read_order, field_mask)
        self.epoch_rows[epoch] = len(layout.row_starts) - 1
        self.layout = (epoch, layout)
        return layout

    def locate(self, index: int, seq_len: int) -> tuple[EpochLayout, int]:
        """Layout of the epoch of the source's ``index``-th row and the row's index in it."""
        epoch, rows_before = 0, 0
        while True:
            if epoch not in self.epoch_rows:
                if self.prefixed(epoch):
                    self.epoch_layout(epoch, rows_before, seq_len)
                else:
                    self.epoch_rows[epoch] = (self.num_tokens - 1) // seq_len
            if index < rows_before + self.epoch_rows[epoch]:
                break
            rows_before += self.epoch_rows[epoch]
            epoch += 1
        layout = self.epoch_layout(epoch, rows_before, seq_len)
        row = index - rows_before
        return layout, row if layout.read_order is None else int(layout.read_order[row])

    def epoch_prefixes(self, epoch: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Prefix tokens of every document (file order) in ``epoch`` and the fields in its prefix (bit ``i`` = field
        ``i``; None: all fields)."""
        metadata = self.metadata
        rng = np.random.default_rng([*(self.seed or [0]), epoch, 2])
        prefix_lens, field_mask = self.prefix_lens, None
        if metadata.field_keep < 1 or metadata.max_share is not None:
            field_lens = np.stack([np.diff(ends, prepend=0).astype(np.int32) for _, ends in self.fields], axis=1)
            keep = rng.random(field_lens.shape) < metadata.field_keep
            if metadata.max_share is not None:
                keep &= np.cumsum(field_lens * keep, axis=1) <= metadata.max_share * self.doc_lens[:, None]
            prefix_lens = (field_lens * keep).sum(axis=1)
            field_mask = (keep << np.arange(len(self.fields))).sum(axis=1).astype(np.uint8)
        if metadata.drop:
            prefix_lens = prefix_lens * (rng.random(len(prefix_lens)) >= metadata.drop)
        return prefix_lens, field_mask

    def prefix_reads(
        self, doc: int, start: int, stop: int, field_mask: np.ndarray | None
    ) -> list[tuple[TokenFiles, int, int]]:
        """Reads of the prefix tokens ``[start, stop)`` of document ``doc``."""
        reads, offset = [], 0
        for i, (files, ends) in enumerate(self.fields):
            if field_mask is not None and not field_mask[doc] >> i & 1:
                continue
            end = int(ends[doc])
            length = end - (int(ends[doc - 1]) if doc else 0)
            lo, hi = max(start, offset), min(stop, offset + length)
            if lo < hi:
                reads.append((files, end - length + lo - offset, hi - lo))
            offset += length
        return reads

    def row(self, index: int, seq_len: int) -> dict[str, np.ndarray]:
        """The source's ``index``-th row: ``seq_len + 1`` tokens (one past the row for the last target), the lengths
        of its document segments and the loss mask of its targets (no loss on prefix tokens)."""
        layout, row = self.locate(index, seq_len)
        start, stop = int(layout.row_starts[row]), int(layout.row_starts[row + 1]) + 1
        reads: list[tuple[TokenFiles, int, int, bool]] = []  # files, start, length, prefix
        doc_starts = []

        def add(new_reads: list[tuple[TokenFiles, int, int]], prefix: bool):
            for files, offset, length in new_reads:
                if not prefix and reads and not reads[-1][3] and reads[-1][1] + reads[-1][2] == offset:
                    reads[-1] = (files, reads[-1][1], reads[-1][2] + length, False)  # contiguous content
                else:
                    reads.append((files, offset, length, prefix))

        position = 0  # in the row
        i = int(np.searchsorted(layout.ends, start, side="right"))
        continued = seq_len - (stop - 1 - start)
        if continued:
            doc = int(layout.order[i]) if layout.order is not None else i
            add(self.prefix_reads(doc, 0, continued, layout.field_mask), True)
            position = continued
        while start < stop:
            doc = int(layout.order[i]) if layout.order is not None else i
            doc_end = int(layout.ends[i])
            prefix_len = int(layout.prefix_lens[i]) if layout.prefix_lens is not None else 0
            doc_start = doc_end - prefix_len - int(self.doc_lens[doc])
            if start == doc_start and position > 0:
                doc_starts.append(position)
            end = min(stop, doc_end)
            if start < doc_start + prefix_len:
                prefix_end = min(end, doc_start + prefix_len)
                add(self.prefix_reads(doc, start - doc_start, prefix_end - doc_start, layout.field_mask), True)
                position += prefix_end - start
                start = prefix_end
            if start < end:
                content_start = int(self.doc_ends[doc] - self.doc_lens[doc]) + start - doc_start - prefix_len
                add([(self.content, content_start, end - start)], False)
                position += end - start
                start = end
            i += 1

        if len(reads) > 1:
            if self.pool[0] != os.getpid():  # threads do not survive the fork into dataloader workers
                self.pool = (os.getpid(), ThreadPoolExecutor(self.READ_THREADS))
            pieces = list(self.pool[1].map(lambda read: read[0].read(read[1], read[2]), reads))
        else:
            pieces = [files.read(offset, length) for files, offset, length, _ in reads]
        is_prefix = np.concatenate([np.full(length, prefix) for _, _, length, prefix in reads])
        return {
            "tokens": np.concatenate(pieces),
            "seq_lens": np.diff([0, *(s for s in doc_starts if s < seq_len), seq_len]),
            "loss_mask": ~is_prefix[1:],
        }

    def first_token(self) -> int:
        return int(self.content.read(0, 1)[0])

    def last_token_of_first_document(self) -> int:
        return int(self.content.read(int(self.doc_ends[0]) - 1, 1)[0])


def token_dtype(file: Path) -> np.dtype:
    first_line = Path(f"{file}.metadata").read_text().splitlines()[0]
    return np.dtype(f"<u{int(first_line.rsplit('|', 1)[1])}")


class RandomTokens:
    """An endless stream of uniformly random token ids without document boundaries, for smoke tests."""

    num_tokens = 2**62

    def __init__(self, vocab_size: int, seed: int):
        self.vocab_size = vocab_size
        self.seed = seed

    def row(self, index: int, seq_len: int) -> dict[str, np.ndarray]:
        tokens = np.random.default_rng([self.seed, index * seq_len]).integers(0, self.vocab_size, seq_len + 1)
        return {"tokens": tokens, "seq_lens": np.array([seq_len]), "loss_mask": np.ones(seq_len, dtype=bool)}


class TokenDataset(StatefulIterableDataset):
    """``seq_len`` rows of token sources, each row from one source of a weighted mixture.

    Row ``w``'s source is drawn from the mixture ``weights`` with an RNG keyed by ``(seed, w)``, and it is that
    source's next row (``TokenShards.row``). Every data rank walks the whole row sequence but reads only its own rows;
    the state is the row counter and the tokens read per source. Each row reads one token past its end for the last
    target. With ``num_windows`` the dataset stops after that many rows."""

    def __init__(
        self,
        sources: dict[str, TokenShards | RandomTokens],
        weights: dict[str, float],
        seq_len: int,
        seed: int = 0,
        non_dp_size: int = 1,
        num_windows: int | None = None,
    ):
        super().__init__(non_dp_size)
        self.sources = sources
        self.names = list(sources)
        self.weights = np.array([weights.get(name, 0.0) for name in self.names])
        self.cdf = np.cumsum(self.weights)
        self.seq_len = seq_len
        self.seed = seed
        self.num_windows = num_windows
        self.tokens = dict.fromkeys(self.names, 0)

    def epochs(self) -> dict[str, float]:
        return {name: self.tokens[name] / self.sources[name].num_tokens for name in self.names}

    def state_dict(self) -> dict:
        return {"dataset": {"step": self.step, "tokens": dict(self.tokens), "epochs": self.epochs()}}

    def load_state_dict(self, state_dict: dict):
        self.step = state_dict["dataset"]["step"]
        self.tokens.update(state_dict["dataset"]["tokens"])

    def planned_epochs(self, num_tokens: int) -> dict[str, float]:
        """Expected passes over each source's tokens when ``num_tokens`` tokens are read."""
        return {
            name: num_tokens * float(weight) / self.sources[name].num_tokens
            for name, weight in zip(self.names, self.weights)
        }

    def source(self, window: int) -> str:
        if len(self.names) == 1:
            return self.names[0]
        draw = np.random.default_rng([self.seed, window]).random() * self.cdf[-1]
        return self.names[min(int(np.searchsorted(self.cdf, draw, side="right")), len(self.names) - 1)]

    def __iter__(self):
        self._setup_world_info()
        while self.num_windows is None or self.step < self.num_windows:
            window = self.step
            name = self.source(window)
            index = self.tokens[name] // self.seq_len
            self.tokens[name] += self.seq_len
            self.step += 1
            if window % self.data_world_size != self.data_rank:
                continue
            yield self.sources[name].row(index, self.seq_len)


def source_tokens(dataset: TokenDataset, start: int, end: int) -> dict[str, int]:
    """Tokens each source contributes to windows ``[start, end)``. Every data rank walks the same window
    sequence, so these are global counts, computed without communication."""
    tokens = dict.fromkeys(dataset.names, 0)
    for window in range(start, end):
        tokens[dataset.source(window)] += dataset.seq_len
    return tokens


def collate_rows(rows: list[dict]) -> dict:
    """Concatenate rows into one packed sample of shape ``[1, len(rows) * seq_len]``. Positions restart at
    every document; attention never crosses a document or a row."""
    tokens = torch.from_numpy(np.stack([row["tokens"] for row in rows]))
    seq_lens = torch.from_numpy(np.concatenate([row["seq_lens"] for row in rows]))
    return {
        "input_ids": tokens[:, :-1].reshape(1, -1),
        "target_ids": tokens[:, 1:].reshape(1, -1),
        "loss_mask": torch.from_numpy(np.stack([row["loss_mask"] for row in rows])).reshape(1, -1),
        "position_ids": torch.cat([torch.arange(length) for length in seq_lens.tolist()]).unsqueeze(0),
        "seq_lens": seq_lens,
        "mm_kwargs": None,
        "mm_token_type_ids": None,
    }


def setup_dataloader(dataset: TokenDataset, micro_batch_size: int, num_workers: int) -> StatefulDataLoader:
    return StatefulDataLoader(
        dataset, batch_size=micro_batch_size, collate_fn=collate_rows, num_workers=num_workers, pin_memory=True
    )


def setup_train_dataset(
    config: PretrainDataConfig, max_steps: int, vocab_size: int, eos_token_id: int, non_dp_size: int
) -> TokenDataset:
    if config.type == "fake":
        sources = {"fake": RandomTokens(vocab_size, config.seed)}
        weights = {"fake": 1.0}
    else:
        sources = {
            name: TokenShards(source, seed=(config.seed, name), metadata=config.metadata)
            for name, source in config.sources.items()
        }
        check_eos(sources, eos_token_id)
        weights = config.weights
    dataset = TokenDataset(sources, weights, config.seq_len, seed=config.seed, non_dp_size=non_dp_size)
    if config.type == "tokens" and config.metadata is not None and config.metadata.end_step is not None:
        # Step s trains on rows [(s - 1) * batch_size, s * batch_size).
        end_tokens = source_tokens(dataset, 0, (config.metadata.end_step - 1) * config.batch_size)
        for name, tokens in end_tokens.items():
            sources[name].prefix_end_row = tokens // config.seq_len
    if config.type == "tokens":
        epochs = dataset.planned_epochs(max_steps * config.batch_tokens)
        get_logger().info(f"Planned epochs per source: {epochs}")
        if over := {name: epoch for name, epoch in epochs.items() if epoch > config.max_epochs}:
            raise ValueError(f"Planned epochs {over} exceed data.max_epochs={config.max_epochs}")
    return dataset


def setup_val_dataset(source: TokenShards, seq_len: int, num_windows: int, non_dp_size: int) -> TokenDataset:
    """One pass over the first ``num_windows`` windows of a held-out source."""
    return TokenDataset({"val": source}, {"val": 1.0}, seq_len, non_dp_size=non_dp_size, num_windows=num_windows)


def check_eos(sources: dict[str, TokenShards], eos_token_id: int) -> None:
    """Every document ends with the tokenizer's EOS. A different token there means the shards were made with
    another tokenizer."""
    for name, source in sources.items():
        last = source.last_token_of_first_document()
        if last != eos_token_id:
            raise ValueError(
                f"The first document of source {name!r} ends with token {last}, not the tokenizer's EOS "
                f"{eos_token_id}. Were the shards tokenized with tokenizer.name?"
            )


def starts_with_bos(sources: dict[str, TokenShards], bos_token_id: int | None) -> bool:
    """Whether the documents of the sources start with the tokenizer's BOS, read off the first document of every
    source. The eval suite formats its requests the same way."""
    starts = {name: source.first_token() == bos_token_id for name, source in sources.items()}
    if len(set(starts.values())) > 1:
        raise ValueError(f"Only some token sources start their documents with BOS: {starts}")
    return all(starts.values())
