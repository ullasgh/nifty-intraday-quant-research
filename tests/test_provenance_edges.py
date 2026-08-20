"""Edge case tests for provenance.py, covering exception paths and default_factory."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from unittest import mock

from nifty_quant.research.provenance import canonical_model_id, get_git_sha


class TestGetGitShaNoGit:
    """Test get_git_sha returns 'no-git' when git is unavailable."""

    def test_git_sha_no_git_repo(self, tmp_path: Path) -> None:
        """get_git_sha returns 'no-git' when run in a non-git directory.

        This exercises the return path when git rev-parse returns non-zero.
        Tests that the result is "no-git" for a missing repository.
        """
        # tmp_path is a temporary directory that is NOT a git repository
        sha = get_git_sha(repo_root=tmp_path)
        assert sha == "no-git"

    def test_git_sha_subprocess_error(self, tmp_path: Path) -> None:
        """get_git_sha returns 'no-git' when subprocess raises an exception.

        This directly tests the exception handler on lines 60-61,
        by patching subprocess.run at the module level where it's imported.
        """
        with mock.patch(
            "nifty_quant.research.provenance.subprocess.run",
            side_effect=subprocess.SubprocessError("Subprocess failed"),
        ):
            sha = get_git_sha(repo_root=tmp_path)
            assert sha == "no-git"

    def test_git_sha_os_error(self, tmp_path: Path) -> None:
        """get_git_sha returns 'no-git' when git command is not found (OSError).

        This directly tests the OSError handler on lines 60-61.
        Exercises the exception path that catches OSError and returns "no-git".
        """
        with mock.patch(
            "nifty_quant.research.provenance.subprocess.run",
            side_effect=OSError("git not found"),
        ):
            sha = get_git_sha(repo_root=tmp_path)
            assert sha == "no-git"


class TestGetGitShaDirtyCheckFailure:
    """Test get_git_sha returns plain sha when status check fails."""

    def test_git_sha_status_fails(self, tmp_path: Path) -> None:
        """get_git_sha returns plain sha (no -dirty) when git status fails.

        This exercises lines 75-77: the exception handler for when
        `git rev-parse HEAD` succeeds but `git status --porcelain` fails.

        Rationale: HEAD is successfully resolved (the commit is known), so
        the bare sha is the most informative response. Treating an
        unreadable status as "can't prove it's dirty" is conservative and
        safe -- it avoids making false dirty claims.
        """
        test_sha = "abc123def456"

        call_count = 0

        def mock_subprocess_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: git rev-parse HEAD
                result = mock.Mock()
                result.returncode = 0
                result.stdout = test_sha + "\n"
                return result
            else:
                # Second call: git status --porcelain
                raise subprocess.SubprocessError("status failed")

        with mock.patch(
            "nifty_quant.research.provenance.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            sha = get_git_sha(repo_root=tmp_path)

        # Should return the plain sha without -dirty suffix
        assert sha == test_sha
        assert not sha.endswith("-dirty")

    def test_git_sha_status_os_error(self, tmp_path: Path) -> None:
        """get_git_sha returns plain sha when git status raises OSError.

        Tests the same path as above but with OSError instead of
        SubprocessError on the status check.
        """
        test_sha = "deadbeefcafe"

        call_count = 0

        def mock_subprocess_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: git rev-parse HEAD
                result = mock.Mock()
                result.returncode = 0
                result.stdout = test_sha + "\n"
                return result
            else:
                # Second call: git status --porcelain
                raise OSError("permission denied")

        with mock.patch(
            "nifty_quant.research.provenance.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            sha = get_git_sha(repo_root=tmp_path)

        # Should return the plain sha without -dirty suffix
        assert sha == test_sha


class TestCanonicalModelIdDefaultFactory:
    """Test canonical_model_id with default_factory fields."""

    def test_default_factory_omitted_when_default(self) -> None:
        """canonical_model_id omits default_factory fields when set to default.

        This exercises lines 104-105: the condition that skips a field
        whose value equals the result of calling default_factory().
        """

        @dataclasses.dataclass
        class ModelWithFactory:
            name: str
            options: list = dataclasses.field(default_factory=list)

        # Instance with default empty list
        instance = ModelWithFactory(name="test")
        model_id = canonical_model_id(instance)

        # Should only include 'name', not 'options' (since options == [])
        assert model_id == "ModelWithFactory(name='test')"
        assert "options" not in model_id

    def test_default_factory_included_when_non_default(self) -> None:
        """canonical_model_id includes default_factory fields when non-default.

        This is the second half of testing lines 104-105: ensuring that
        non-default values are included in the canonical form.
        """

        @dataclasses.dataclass
        class ModelWithFactory:
            name: str
            options: list = dataclasses.field(default_factory=list)

        # Instance with non-default list
        instance = ModelWithFactory(name="test", options=["a", "b"])
        model_id = canonical_model_id(instance)

        # Should include both fields
        assert "name='test'" in model_id
        assert "options=['a', 'b']" in model_id

    def test_default_factory_dict(self) -> None:
        """canonical_model_id correctly handles dict default_factory.

        Extends the default_factory test to another common factory type.
        """

        @dataclasses.dataclass
        class ModelWithDictFactory:
            name: str
            metadata: dict = dataclasses.field(default_factory=dict)

        # Default case: empty dict should be omitted
        instance1 = ModelWithDictFactory(name="test")
        model_id1 = canonical_model_id(instance1)
        assert model_id1 == "ModelWithDictFactory(name='test')"
        assert "metadata" not in model_id1

        # Non-default case: dict with content should be included
        instance2 = ModelWithDictFactory(name="test", metadata={"key": "value"})
        model_id2 = canonical_model_id(instance2)
        assert "metadata=" in model_id2
        assert "key" in model_id2

    def test_default_factory_with_multiple_fields(self) -> None:
        """canonical_model_id handles multiple default_factory fields correctly.

        Ensures sorting and correct inclusion logic with multiple factories.
        """

        @dataclasses.dataclass
        class ModelWithMultipleFactories:
            alpha: list = dataclasses.field(default_factory=list)
            beta: dict = dataclasses.field(default_factory=dict)
            gamma: str = "default"

        # All defaults
        instance1 = ModelWithMultipleFactories()
        model_id1 = canonical_model_id(instance1)
        # gamma is at its default, alpha/beta factories produce defaults
        assert model_id1 == "ModelWithMultipleFactories()"

        # Mix of defaults and non-defaults
        instance2 = ModelWithMultipleFactories(alpha=[1, 2], gamma="custom")
        model_id2 = canonical_model_id(instance2)
        # Should be sorted by field name and include only non-defaults
        assert "alpha=[1, 2]" in model_id2
        assert "gamma='custom'" in model_id2
        assert "beta" not in model_id2
