# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression guard: the plain warmup branch must not recompile per bucket.

Qwen3TTSCodePredictor._warmup_buckets()'s plain branch (no KV-cache, no
prefix-reprefill) marks the batch dimension dynamic after the first bucket so
later buckets reuse one compiled graph instead of each bucket triggering its
own Inductor compile (see the sibling KV-cache-enabled branch's existing use
of the same pattern a few lines above). This test counts actual Dynamo
compiles via CompileCounter so a future change to `forward()` that
reintroduces per-bucket specialization (e.g. a new `x.shape[0]` branch) fails
CI instead of silently costing ~5x the warmup time again.
"""

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]

_BUCKET_SIZES = (1, 2, 4, 8, 16)
_SEQLEN = 5
_HIDDEN = 64


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA")
def test_plain_warmup_compiles_at_most_two_graphs():
    """Mirrors _warmup_buckets()'s plain branch; asserts <=2 Dynamo compiles for 5 buckets."""
    torch.manual_seed(171)
    config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=64,
        hidden_size=_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_code_groups=4,
    )
    model = CodePredictorBaseModel(config).to(device="cuda", dtype=torch.bfloat16).eval()

    counter = CompileCounter()
    compiled_fwd = torch.compile(model.forward, backend=counter, dynamic=False)

    with torch.inference_mode():
        for bsz in _BUCKET_SIZES:
            x = torch.randn(bsz, _SEQLEN, _HIDDEN, device="cuda", dtype=torch.bfloat16)
            pos = torch.arange(_SEQLEN, device="cuda").unsqueeze(0).expand(bsz, -1).contiguous()
            if bsz > 1:
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(pos, 0)
            compiled_fwd(x, pos)

    # 1 compile for bsz==1 (kept static) + 1 dynamic compile reused for bsz>1.
    # Without mark_dynamic this would be len(_BUCKET_SIZES) (one per bucket).
    assert counter.frame_count <= 2, (
        f"expected <=2 Dynamo compiles across {len(_BUCKET_SIZES)} buckets, "
        f"got {counter.frame_count} -- a change to forward() may have reintroduced "
        "per-bucket shape specialization"
    )
