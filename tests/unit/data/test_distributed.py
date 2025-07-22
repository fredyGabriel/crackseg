from unittest import mock

import crackseg.data.distributed as dist


def test_is_distributed_available_and_initialized_true():
    """Should return True if torch.distributed is available and initialized."""
    with (
        mock.patch("torch.distributed.is_available", return_value=True),
        mock.patch("torch.distributed.is_initialized", return_value=True),
    ):
        assert dist.is_distributed_available_and_initialized() is True


def test_is_distributed_available_and_initialized_false():
    """
    Should return False if torch.distributed is not available or not
    initialized.
    """
    with (
        mock.patch("torch.distributed.is_available", return_value=False),
        mock.patch("torch.distributed.is_initialized", return_value=True),
    ):
        assert dist.is_distributed_available_and_initialized() is False
    with (
        mock.patch("torch.distributed.is_available", return_value=True),
        mock.patch("torch.distributed.is_initialized", return_value=False),
    ):
        assert dist.is_distributed_available_and_initialized() is False


def test_get_rank_and_world_size_distributed():
    """Should return correct rank and world size if distributed."""
    with (
        mock.patch("torch.distributed.is_available", return_value=True),
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_rank", return_value=3),
        mock.patch("torch.distributed.get_world_size", return_value=8),
    ):
        assert dist.get_rank() == 3  # noqa: PLR2004
        assert dist.get_world_size() == 8  # noqa: PLR2004


def test_get_rank_and_world_size_not_distributed():
    """Should return 0 and 1 if not distributed."""
    with (
        mock.patch("torch.distributed.is_available", return_value=False),
        mock.patch("torch.distributed.is_initialized", return_value=False),
    ):
        assert dist.get_rank() == 0
        assert dist.get_world_size() == 1


def test_sync_distributed_calls_barrier():
    """Should call barrier if distributed, else do nothing."""
    with (
        mock.patch("torch.distributed.is_available", return_value=True),
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.barrier") as mock_barrier,
    ):
        dist.sync_distributed()
        mock_barrier.assert_called_once()


def test_sync_distributed_no_barrier():
    """Should not call barrier if not distributed."""
    with (
        mock.patch("torch.distributed.is_available", return_value=False),
        mock.patch("torch.distributed.is_initialized", return_value=False),
        mock.patch("torch.distributed.barrier") as mock_barrier,
    ):
        dist.sync_distributed()
        mock_barrier.assert_not_called()
