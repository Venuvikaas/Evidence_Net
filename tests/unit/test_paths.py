"""Tests for official dataset path resolution."""

from pathlib import Path

import pytest

from evidence_net.data.paths import (
    DatasetPathError,
    find_execution_parent,
    load_dotenv,
    resolve_dataset_paths,
)


def _make_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    parent = tmp_path / "project-parent"
    parent.mkdir()
    (parent / "EVIDENCE_NET_EXECUTION_WITH_DATASET.md").write_text("# plan", encoding="utf-8")
    repo = parent / "evidence-net"
    repo.mkdir()
    train = parent / "train"
    test = parent / "Test_NoisyLR"
    train.mkdir()
    test.mkdir()
    return parent, repo, train


def test_resolves_from_execution_parent(tmp_path: Path) -> None:
    parent, repo, _ = _make_project(tmp_path)
    datasets = resolve_dataset_paths(env={}, repo_root=repo)
    assert datasets.train_dir == parent / "train"
    assert datasets.test_noisylr_dir == parent / "Test_NoisyLR"
    assert datasets.execution_parent == parent
    assert datasets.source == "execution-parent"


def test_env_vars_override_discovery(tmp_path: Path) -> None:
    parent, repo, _ = _make_project(tmp_path)
    custom_train = tmp_path / "custom-train"
    custom_test = tmp_path / "custom-test"
    custom_train.mkdir()
    custom_test.mkdir()
    datasets = resolve_dataset_paths(
        env={"TRAIN_DATA_DIR": str(custom_train), "TEST_NOISY_LR_DIR": str(custom_test)},
        repo_root=repo,
    )
    assert datasets.train_dir == custom_train
    assert datasets.test_noisylr_dir == custom_test
    assert datasets.source == "env"


def test_missing_dataset_fails_with_message(tmp_path: Path) -> None:
    parent, repo, _ = _make_project(tmp_path)
    (parent / "train").rmdir()
    with pytest.raises(DatasetPathError, match="train directory not found"):
        resolve_dataset_paths(env={}, repo_root=repo)


def test_missing_execution_file_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(DatasetPathError, match="execution file"):
        resolve_dataset_paths(env={}, repo_root=repo)


def test_env_overrides_do_not_require_execution_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    train = tmp_path / "train"
    test = tmp_path / "Test_NoisyLR"
    train.mkdir()
    test.mkdir()
    datasets = resolve_dataset_paths(
        env={"TRAIN_DATA_DIR": str(train), "TEST_NOISY_LR_DIR": str(test)},
        repo_root=repo,
    )
    assert datasets.train_dir == train
    assert datasets.test_noisylr_dir == test
    assert datasets.source == "env"


def test_find_execution_parent(tmp_path: Path) -> None:
    parent, repo, _ = _make_project(tmp_path)
    assert find_execution_parent(repo) == parent


def test_load_dotenv_parses_basic_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nTRAIN_DATA_DIR=\"D:/data/train\"\nEMPTY=\nTEST_NOISY_LR_DIR='D:/data/test'\n",
        encoding="utf-8",
    )
    values = load_dotenv(env_file)
    assert values["TRAIN_DATA_DIR"] == "D:/data/train"
    assert values["TEST_NOISY_LR_DIR"] == "D:/data/test"
    assert "EMPTY" in values
