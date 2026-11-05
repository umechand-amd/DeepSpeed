# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from unittest.mock import patch

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
from deepspeed.accelerator.xpu_accelerator import XPU_Accelerator


@pytest.mark.parametrize("name, accel_cls", [("cuda", CUDA_Accelerator), ("xpu", XPU_Accelerator)])
def test_synchronize_passes_explicit_device(name, accel_cls):
    # A None device index breaks torch.compile, and no index makes Dynamo sync device 0,
    # so synchronize() must forward the current device explicitly.
    backend = getattr(torch, name, None)
    if backend is None:
        pytest.skip(f"this torch build has no torch.{name} module")
    accel = accel_cls()
    with patch.object(backend, "current_device", return_value=3), \
            patch.object(backend, "synchronize") as sync:
        accel.synchronize()
        accel.synchronize(None)
        accel.synchronize(1)
    assert [c.args for c in sync.call_args_list] == [(3, ), (3, ), (1, )]


def test_synchronize_under_compile():
    accel = get_accelerator()
    if accel.device_name() not in ("cuda", "xpu"):
        pytest.skip(f"{accel.device_name()} accelerator is not covered by this test")

    def fn(x):
        accel.synchronize()
        return x + 1

    torch._dynamo.reset()
    x = torch.ones(4, device=accel.current_device_name())
    out = torch.compile(fn, backend="eager")(x)
    assert torch.equal(out, x + 1)
