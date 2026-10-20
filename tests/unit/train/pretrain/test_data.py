from itertools import islice
from pathlib import Path

import numpy as np

from prime_rl.configs.pretrain import MetadataConfig, TokenSourceConfig
from prime_rl.trainer.pretrain.data import TokenDataset, TokenShards, collate_rows, source_tokens


def write_shard(path: Path, documents: list[list[int]]) -> None:
    """A datatrove DocumentTokenizer shard: uint32 tokens, uint64 cumulative document ends, metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.array([token for document in documents for token in document], dtype="<u4").tofile(path)
    np.cumsum([len(document) for document in documents], dtype="<u8").tofile(f"{path}.index")
    Path(f"{path}.metadata").write_text("tokenizer|4\n0\n0")


def write_sidecar(path: Path, prefixes: list[list[int]]) -> None:
    """A metadata sidecar field file: uint32 prefix tokens and uint64 cumulative ends, one per document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.array([token for prefix in prefixes for token in prefix], dtype="<u4").tofile(path)
    np.cumsum([len(prefix) for prefix in prefixes], dtype="<u8").tofile(f"{path}.index")


def documents(row: dict) -> list[list[int]]:
    """The row's tokens (without the extra target token) split at its document segments."""
    tokens = row["tokens"][:-1].tolist()
    ends = np.cumsum(row["seq_lens"]).tolist()
    return [tokens[start:end] for start, end in zip([0, *ends[:-1]], ends)]


def test_unshuffled_rows_read_the_files_as_one_stream(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [[1, 2, 3], [4, 5]])
    write_shard(tmp_path / "a" / "01.ds", [[6, 7, 8, 9]])
    shards = TokenShards(TokenSourceConfig(path=tmp_path / "a"))
    assert shards.num_tokens == 9
    row = shards.row(0, 4)
    assert row["tokens"].tolist() == [1, 2, 3, 4, 5]
    assert row["seq_lens"].tolist() == [3, 1]
    assert row["loss_mask"].all()
    assert shards.row(1, 4)["tokens"].tolist() == [5, 6, 7, 8, 9]
    # max_tokens keeps the documents that end within it.
    assert TokenShards(TokenSourceConfig(path=tmp_path / "a", max_tokens=8)).num_tokens == 5


def test_shuffled_epochs_repack_every_document_once_in_a_new_order(tmp_path):
    docs = [list(range(start, start + length)) for start, length in [(0, 30), (30, 5), (35, 15), (50, 9), (59, 13)]]
    write_shard(tmp_path / "a" / "00.ds", docs[:3])
    write_shard(tmp_path / "a" / "01.ds", docs[3:])
    shards = TokenShards(TokenSourceConfig(path=tmp_path / "a"), seed=(0, "a"))
    # 72 tokens fill 10 rows of 7 (plus the extra target token); the rest of each epoch is skipped.
    orders = []
    for epoch in range(3):
        rows = [shards.row(index, 7) for index in range(epoch * 10, epoch * 10 + 10)]
        tokens = [token for row in rows for token in row["tokens"][:-1].tolist()]
        assert len(tokens) == len(set(tokens)) == 70
        # Every segment is a contiguous piece of one document.
        for piece in (piece for row in rows for piece in documents(row)):
            assert any(piece == doc[i : i + len(piece)] for doc in docs for i in range(len(doc)))
        orders.append(tuple(tokens))
    assert len(set(orders)) == 3
    # Another reader (a dataloader worker, a resumed run) reads the same rows.
    again = TokenShards(TokenSourceConfig(path=tmp_path / "a"), seed=(0, "a"))
    assert tuple(token for index in range(10, 20) for token in again.row(index, 7)["tokens"][:-1].tolist()) == orders[1]


def test_metadata_prefixes_every_row_piece_without_loss(tmp_path):
    docs = [list(range(100, 112)), list(range(200, 203)), list(range(300, 309))]
    prefixes = [[1, 2], [], [3]]
    write_shard(tmp_path / "a" / "00.ds", docs)
    write_sidecar(tmp_path / "meta" / "00.domain.ds", prefixes)
    config = TokenSourceConfig(path=tmp_path / "a", metadata=tmp_path / "meta")
    shards = TokenShards(config, metadata=MetadataConfig(fields=["domain"]))
    rows = [shards.row(index, 5) for index in range(6)]
    content = []
    for row in rows:
        assert row["tokens"].shape == (6,) and row["loss_mask"].shape == (5,)
        targets = row["tokens"][1:]
        # No loss exactly on the prefix targets.
        assert (row["loss_mask"] == (targets >= 100)).all()
        for piece in documents(row):
            doc = next(i for i, d in enumerate(docs) if piece[-1] in d)
            prefix = prefixes[doc]
            assert piece[: len(prefix)] == prefix  # every piece starts with its document's prefix
            content += piece[len(prefix) :]
    assert content == [token for doc in docs for token in doc][: len(content)]
    assert len(content) == 20  # of 24: the last partial row is skipped

    # Epochs without prefixes read the plain documents.
    plain = TokenShards(config, metadata=MetadataConfig(fields=["domain"], epochs=[1]))
    assert plain.row(0, 5)["tokens"].tolist() == docs[0][:6]


def test_metadata_ends_at_its_row(tmp_path):
    docs = [[100 + i, 101 + i, 102 + i] for i in range(0, 300, 3)]
    write_shard(tmp_path / "a" / "00.ds", docs)
    write_sidecar(tmp_path / "meta" / "00.domain.ds", [[1]] * len(docs))
    config = TokenSourceConfig(path=tmp_path / "a", metadata=tmp_path / "meta")
    shards = TokenShards(config, metadata=MetadataConfig(fields=["domain"]))
    shards.prefix_end_row = 6
    prefixed = [shards.row(index, 4)["tokens"][0] == 1 for index in range(20)]
    assert prefixed == [True] * 6 + [False] * 14
    # Shuffled, the rows with prefixes are still the ones read before the end.
    shuffled = TokenShards(config, seed=(0, "a"), metadata=MetadataConfig(fields=["domain"]))
    shuffled.prefix_end_row = 6
    prefixed = [bool((shuffled.row(index, 4)["tokens"] == 1).any()) for index in range(20)]
    assert all(prefixed[:6]) and not any(prefixed[6:])


def test_token_dataset_resumes_and_continues_each_source(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [list(range(100, 200))])
    write_shard(tmp_path / "b" / "00.ds", [list(range(200, 300))])
    sources = {name: TokenShards(TokenSourceConfig(path=tmp_path / name)) for name in "ab"}

    def dataset():
        return TokenDataset(sources, {"a": 0.5, "b": 0.5}, seq_len=4)

    full = [window["tokens"].tolist() for window in islice(dataset(), 8)]
    for start in (100, 200):
        windows = [window for window in full if start <= window[0] < start + 100]
        assert windows and windows == [list(range(start + 4 * i, start + 4 * i + 5)) for i in range(len(windows))]

    first = dataset()
    iterator = iter(first)
    for _ in range(3):
        next(iterator)
    resumed = dataset()
    resumed.load_state_dict(first.state_dict())
    assert [window["tokens"].tolist() for window in islice(resumed, 5)] == full[3:8]


def test_source_tokens_count_every_row_of_the_mixture(tmp_path):
    for name in "ab":
        write_shard(tmp_path / name / "00.ds", [list(range(100))])
    sources = {name: TokenShards(TokenSourceConfig(path=tmp_path / name)) for name in "ab"}
    dataset = TokenDataset(sources, {"a": 0.3, "b": 0.7}, seq_len=4)
    list(islice(dataset, 32))
    # The training loop's global per-source counts equal the tokens the dataset itself read.
    assert source_tokens(dataset, 0, 32) == dataset.tokens
    assert sum(source_tokens(dataset, 8, 16).values()) == 8 * 4


def test_collate_rows_restarts_positions_at_documents():
    rows = [
        {"tokens": np.arange(5), "seq_lens": np.array([3, 1]), "loss_mask": np.array([1, 1, 1, 1], dtype=bool)},
        {"tokens": np.arange(10, 15), "seq_lens": np.array([4]), "loss_mask": np.array([0, 1, 1, 1], dtype=bool)},
    ]
    batch = collate_rows(rows)
    assert batch["input_ids"].tolist() == [[0, 1, 2, 3, 10, 11, 12, 13]]
    assert batch["target_ids"].tolist() == [[1, 2, 3, 4, 11, 12, 13, 14]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 0, 1, 2, 3]]
    assert batch["seq_lens"].tolist() == [3, 1, 4]
    assert batch["loss_mask"].tolist() == [[True, True, True, True, False, True, True, True]]
