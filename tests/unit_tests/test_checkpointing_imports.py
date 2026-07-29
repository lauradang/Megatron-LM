# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import subprocess
import sys


def test_checkpointing_does_not_eagerly_import_rl_utils():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import megatron.training.checkpointing; "
                "assert 'megatron.rl.rl_utils' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
