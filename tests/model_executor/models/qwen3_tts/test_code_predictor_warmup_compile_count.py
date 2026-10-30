# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression guard: the plain warmup branch must not recompile per bucket.

Qwen3TTSCodePredictor._warmup_buckets()'s plain branch (no KV-cache, no
prefix-reprefill) marks the batch dimension dynamic after the first bucket so
later buckets reuse one compiled graph instead of each bucket triggering its
own Inductor compile (see the sibling KV-cache-enabled branch's existing use
of the same pattern a few lines above). This test drives the real
_setup_compile() / _warmup_buckets() call chain -- not a reimplementation of
the warmup loop -- with torch.compile patched to count Dynamo frames instead
of invoking Inductor, so a future change that drops/moves the mark_dynamic
calls, or reintroduces per-bucket specialization in forward() (including via
a new graph break), fails this test instead of silently costing the warmup
time back.

Uses plain (unmocked) imports, like test_cached_code_predictor.py, rather
than test_code_predictor_dtype.py's CPU-oriented mocked-import harness: with
that harness's MagicMock-patched current_omni_platform, Dynamo takes a
different trace through a graph break inside CodePredictorAttention and
mis-specializes the resumed frame's batch dim to a constant, independent of
this fix (reproduced with CodePredictorBaseModel directly: a real
current_omni_platform singleton on real CUDA hits none of that). Mocking the
platform is exactly the kind of interference this test needs to avoid.
"""

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
    Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA")
def test_plain_warmup_compiles_exactly_two_graphs(mocker) -> None:
    """Real _setup_compile()/_warmup_buckets() plain branch; asserts == 2 Dynamo frames."""
    torch._dynamo.reset()
    try:
        cp_config = Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_code_groups=4,
            rms_norm_eps=1e-6,
        )
        talker_config = Qwen3TTSTalkerConfig(hidden_size=64, num_code_groups=4)

        vllm_config = mocker.MagicMock()
        vllm_config.scheduler_config.max_num_seqs = 16
        # Empty dict (not a MagicMock default) so .get(...) on every optional
        # key is None -- forces the plain branch, not KV-cache/prefix-reprefill.
        vllm_config.additional_config = {}

        counter = CompileCounter()
        real_compile = torch.compile

        def counting_compile(fn, *args, **kwargs):
            kwargs["backend"] = counter
            # "options" is Inductor-specific; CompileCounter.__call__ doesn't accept it.
            kwargs.pop("options", None)
            return real_compile(fn, *args, **kwargs)

        mocker.patch("torch.compile", counting_compile)

        predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        ).to(device="cuda", dtype=torch.bfloat16)

        predictor._setup_compile()  # drives the real plain-branch _warmup_buckets()

        assert not predictor._kv_cache_enabled
        assert not predictor._prefix_reprefill_enabled
        # Exactly 1 static graph for bsz==1 + 1 dynamic graph reused for bsz>1
        # across all of predictor._bucket_sizes. frame_count also rises on a
        # graph break in forward(), which should trip this test too.
        assert counter.frame_count == 2, (
            f"expected exactly 2 Dynamo frames (1 static bsz==1 graph + 1 dynamic graph "
            f"for bsz>1) across buckets {predictor._bucket_sizes}, got {counter.frame_count} "
            "-- either per-bucket shape specialization returned (mark_dynamic dropped/moved) "
            "or forward() gained a new graph break"
        )
    finally:
        torch._dynamo.reset()
