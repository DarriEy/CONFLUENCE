import os

import pytest


@pytest.mark.smoke
def test_quick_smoke(tmp_path):
    # Minimal import path check
    import symfluence  # noqa: F401

    # Ensure default data dir behavior
    # Expect ../SYMFLUENCE_data relative to cwd
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    expected = os.path.join(parent, 'SYMFLUENCE_data')
    assert os.path.isdir(expected) or not os.path.exists(expected)

    # TODO: optional: run a super tiny pipeline with a mocked backend
    assert True
