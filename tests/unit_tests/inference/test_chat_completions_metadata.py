# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

quart = pytest.importorskip("quart")

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (  # noqa: E501
    bp,
)


class DummyTokenizer:
    chat_template = "{{ messages }}"
    bos = None
    eos_id = 0

    def apply_chat_template(self, *args, **kwargs):
        return [101, 102]

    def detokenize(self, token_ids):
        return f"token-{token_ids[0]}"


class DummyClient:
    async def add_request(self, prompt_tokens, sampling_params):
        return {
            "status": "COMPLETED",
            "prompt": "prompt",
            "generated_text": " answer",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": [201, 202],
            "generated_log_probs": [-0.1, -0.2],
            "log_probs": [-0.1, -0.2],
            "generated_top_n_logprobs": None,
            "policy_epoch": [(0, 3)],
            "kv_cache_epoch": [(0, 5)],
            "events": [{"type": "EVICT"}, {"type": "OTHER"}, {"type": "EVICT"}],
            "sampling_params": {"num_tokens_to_generate": 4},
            "routing_indices": None,
        }


@pytest.mark.asyncio
async def test_chat_completion_copies_metadata_to_choice_and_message():
    app = quart.Quart(__name__)
    app.config.update(
        client=DummyClient(),
        tokenizer=DummyTokenizer(),
        parsers=[],
        verbose=False,
    )
    app.register_blueprint(bp)

    client = app.test_client()
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "megatron-policy",
            "messages": [{"role": "user", "content": "hello"}],
            "logprobs": True,
            "max_tokens": 4,
        },
    )

    assert response.status_code == 200
    payload = await response.get_json()
    choice = payload["choices"][0]
    message = choice["message"]

    assert choice["prompt_token_ids"] == [101, 102]
    assert choice["generation_token_ids"] == [201, 202]
    assert choice["generation_log_probs"] == [-0.1, -0.2]
    assert choice["policy_epoch"] == [[0, 3]]
    assert choice["kv_cache_epoch"] == [[0, 5]]
    assert choice["num_evictions"] == 2

    assert message["prompt_token_ids"] == choice["prompt_token_ids"]
    assert message["generation_token_ids"] == choice["generation_token_ids"]
    assert message["generation_log_probs"] == choice["generation_log_probs"]
    assert message["policy_epoch"] == choice["policy_epoch"]
    assert message["kv_cache_epoch"] == choice["kv_cache_epoch"]
    assert message["num_evictions"] == choice["num_evictions"]
