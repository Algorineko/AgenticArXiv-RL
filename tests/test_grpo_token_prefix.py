"""Append-only GRPO token boundaries, without models, downloads, or training."""

import os
import re
from unittest.mock import patch

import pytest

from agentic_arxiv.runtime.react_contract import append_react_turn
from agentic_arxiv.training.grpo_reward import (
    _append_environment_tokens,
    make_multiturn_rollout_func,
)


CALL = 'Thought: look !\nAction: {"name":"search_arxiv_papers","args":{"query":"agent"}}'
FINISH = "Thought: done\nAction: FINISH"


@pytest.mark.local
@pytest.mark.skipif(not os.getenv("AGENTIC_ARXIV_TEST_TOKENIZER"), reason="explicit local tokenizer path required")
@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("with_eos", [False, True])
def test_real_local_tokenizer_preserves_original_prefix_across_turns(alternate, with_eos):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.environ["AGENTIC_ARXIV_TEST_TOKENIZER"], local_files_only=True,
    )
    messages = [{"role": "user", "content": "task"}]
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    for turn in range(2):
        policy = 'Thought: 检索 agent !\nAction: {"name":"search_arxiv_papers","args":{"query":"agent"}}'
        canonical = tokenizer(policy, add_special_tokens=False)["input_ids"]
        if alternate and turn == 0:
            sampled = [t for ch in policy for t in tokenizer(ch, add_special_tokens=False)["input_ids"]]
            assert sampled != canonical
        else:
            sampled = canonical
        sampled = sampled + ([tokenizer.eos_token_id] if with_eos else [])
        prefix = full + sampled
        expected_messages = append_react_turn(messages, policy, "REAL")
        candidate = tokenizer.apply_chat_template(expected_messages, tokenize=True, add_generation_prompt=True)
        if alternate:
            assert candidate[:len(prefix)] != prefix
        suffix, messages = _append_environment_tokens(
            tokenizer, messages, policy, "REAL", prefix, sampled_policy_ids=sampled,
        )
        full = prefix + suffix
        assert full[:len(prefix)] == prefix
        assert tokenizer.decode(full, skip_special_tokens=False, clean_up_tokenization_spaces=False) == (
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )


class DualEncodingTokenizer:
    """Two distinct valid token sequences decode to the same assistant text."""

    eos_token_id = 3
    all_special_ids = [3, 4, 5]
    specials = {"<end>": 3, "<user>": 4, "<assistant>": 5}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for piece in re.split(r"(<end>|<user>|<assistant>)", text):
            ids.extend([self.specials[piece]] if piece in self.specials
                       else [ord(ch) + 100 for ch in piece])
        return {"input_ids": ids}

    def alternate(self, text):
        return [ord(ch) + 100000 for ch in text]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        inverse = {v: k for k, v in self.specials.items()}
        text = "".join(
            ("" if skip_special_tokens else inverse[token]) if token in inverse
            else chr(token - (100000 if token >= 100000 else 100))
            for token in ids
        )
        return text.replace(" !", "!") if clean_up_tokenization_spaces else text

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = "".join(f"<{m['role']}>{m['content']}<end>\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self(text)["input_ids"] if tokenize else text


@pytest.mark.parametrize("with_eos", [False, True])
def test_append_preserves_noncanonical_sampled_ids_and_does_not_duplicate_eos(with_eos):
    tokenizer = DualEncodingTokenizer()
    messages = [{"role": "user", "content": "task"}]
    prompt = tokenizer.apply_chat_template(messages)
    sampled = tokenizer.alternate(CALL) + ([3] if with_eos else [])
    original = prompt + sampled
    # The old full-transcript re-encoding guard rejects these legal samples.
    candidate = tokenizer.apply_chat_template(append_react_turn(messages, CALL, "REAL"))
    assert candidate[:len(original)] != original
    suffix, appended = _append_environment_tokens(
        tokenizer, messages, CALL, "REAL", original, sampled_policy_ids=sampled,
    )
    assert original == prompt + sampled
    assert (original + suffix)[:len(original)] == original
    assert tokenizer.decode(original + suffix, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False) == tokenizer.apply_chat_template(
                                appended, tokenize=False,
                            )
    assert (original + suffix).count(3) == 3  # user, assistant, observation
    assert suffix[0] == (ord("\n") + 100 if with_eos else 3)


@pytest.mark.parametrize("bad_ids", ["embedded_eos", "role_injection", "different_text", "wrong_tail"])
def test_append_rejects_unexplained_sample_or_control_token_changes(bad_ids):
    tokenizer = DualEncodingTokenizer()
    messages = [{"role": "user", "content": "task"}]
    sampled = tokenizer.alternate(CALL)
    prefix = tokenizer.apply_chat_template(messages) + sampled
    if bad_ids == "embedded_eos":
        sampled.insert(4, 3)
    elif bad_ids == "role_injection":
        sampled.append(4)
    elif bad_ids == "different_text":
        sampled = tokenizer.alternate("other text")
    if bad_ids != "wrong_tail":
        prefix = tokenizer.apply_chat_template(messages) + sampled
    else:
        prefix.append(ord("x") + 100)
    with pytest.raises(ValueError, match="无法保留原始采样 token 前缀"):
        _append_environment_tokens(tokenizer, messages, CALL, "REAL", prefix,
                                   sampled_policy_ids=sampled)


def test_append_still_rejects_templates_that_rewrite_history():
    class RewritingTokenizer(DualEncodingTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            rendered = super().apply_chat_template(messages, False, add_generation_prompt)
            if len(messages) > 1:
                rendered = "rewritten:" + rendered
            return self(rendered)["input_ids"] if tokenize else rendered

    tokenizer = RewritingTokenizer()
    messages = [{"role": "user", "content": "task"}]
    sampled = tokenizer.alternate(CALL) + [3]
    with pytest.raises(ValueError, match="无法保留原始采样 token 前缀"):
        _append_environment_tokens(tokenizer, messages, CALL, "REAL",
                                   tokenizer.apply_chat_template(messages) + sampled,
                                   sampled_policy_ids=sampled)


def test_append_rejects_a_suffix_that_does_not_round_trip():
    class LossySuffixTokenizer(DualEncodingTokenizer):
        def __call__(self, text, add_special_tokens=False):
            encoded = super().__call__(text, add_special_tokens)
            if text.startswith("<end>"):
                encoded["input_ids"].append(ord("x") + 100)
            return encoded

    tokenizer = LossySuffixTokenizer()
    messages = [{"role": "user", "content": "task"}]
    sampled = tokenizer.alternate(CALL) + [3]
    with pytest.raises(ValueError, match="环境 suffix 编码改变了模板文本"):
        _append_environment_tokens(tokenizer, messages, CALL, "REAL",
                                   tokenizer.apply_chat_template(messages) + sampled,
                                   sampled_policy_ids=sampled)


@pytest.mark.parametrize("budget", [len(CALL) + 8, 4096])
@patch("agentic_arxiv.training.grpo_reward.require_rollout_func_support")
def test_noncanonical_observation_stop_preserves_policy_prefix_and_budget(_support, budget):
    class Trainer:
        processing_class = DualEncodingTokenizer()
        max_completion_length = budget

        def __init__(self):
            self.inputs = []

        def _generate_single_turn(self, prompts, images, fields):
            self.inputs.append([list(p) for p in prompts])
            text = CALL + "\nObservation: FABRICATED" if len(self.inputs) == 1 else FINISH
            return [self.processing_class.alternate(text) + [3]], None, {}

    class Environment:
        def reset(self):
            pass

        def search_arxiv_papers(self, **kwargs):
            return "REAL"

    trainer = Trainer()
    result = make_multiturn_rollout_func(Environment, max_turns=2)(["task"], trainer)
    ids, mask = result["completion_ids"][0], result["env_mask"][0]
    retained = trainer.processing_class.alternate(CALL + "\n")
    assert ids[:len(retained)] == retained and mask[:len(retained)] == [1] * len(retained)
    assert len(ids) == len(mask) <= budget
    assert "FABRICATED" not in trainer.processing_class.decode(ids)
    if budget < 4096:
        assert len(ids) == budget and len(trainer.inputs) == 1
        assert result["trajectory_results"][0]["termination_reason"] == "TOKEN_LIMIT"
    else:
        assert len(trainer.inputs) == 2
        assert "FABRICATED" not in trainer.processing_class.decode(trainer.inputs[1][0])
        assert result["trajectory_results"][0]["termination_reason"] == "FINISH"


@pytest.mark.parametrize("with_eos", [False, True])
@patch("agentic_arxiv.training.grpo_reward.require_rollout_func_support")
def test_multiple_noncanonical_turns_keep_sampled_tokens_and_environment_masks(_support, with_eos):
    class Trainer:
        processing_class = DualEncodingTokenizer()
        max_completion_length = 4096

        def __init__(self):
            self.inputs = []
            self.sampled = []

        def _generate_single_turn(self, prompts, images, fields):
            self.inputs.append([list(p) for p in prompts])
            text = CALL if len(self.inputs) < 3 else FINISH
            ids = self.processing_class.alternate(text) + ([3] if with_eos else [])
            self.sampled.extend(ids)
            return [ids], None, {}

    class Environment:
        def reset(self):
            self.closed = False

        def search_arxiv_papers(self, **kwargs):
            return "REAL"

        def close(self):
            self.closed = True

    trainer = Trainer()
    env = Environment()
    result = make_multiturn_rollout_func(lambda: env, max_turns=3)(["task"], trainer)
    ids, mask = result["completion_ids"][0], result["env_mask"][0]
    assert [t for t, m in zip(ids, mask) if m] == trainer.sampled
    assert len(ids) == len(mask) and 0 in mask and 1 in mask
    assert result["trajectory_results"][0]["termination_reason"] == "FINISH"
    assert result["trajectory_results"][0]["raw_assistant_turns"][0] == CALL
    assert env.closed
    first_sample = trainer.processing_class.alternate(CALL) + ([3] if with_eos else [])
    assert trainer.inputs[1][0][:len(trainer.inputs[0][0]) + len(first_sample)] == (
        trainer.inputs[0][0] + first_sample
    )
    assert ids.count(3) == (5 if with_eos else 4)
