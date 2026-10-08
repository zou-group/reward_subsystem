"""How prompts and responses are turned into model input text.

All probe inputs use the same "math" chat format::

    system:    You are a helpful assistant.
    user:      <prompt>\\nPlease reason step by step, and put your final answer within \\boxed{}.
    assistant: <response>            (omitted for question-only scoring)

Note that ``<prompt>`` is the output of ``benchmarks.construct_prompt``. For the
dopamine pipeline the paper used ``prompt_type=qwen-boxed``, whose prompt is
itself a full ChatML string; it is nevertheless wrapped in this chat template
again (the paper's behaviour, kept on purpose).
"""

from typing import List, Optional

import torch


MATH_SYSTEM_PROMPT = "You are a helpful assistant."
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def _math_messages(prompt: str) -> List[dict]:
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip() + "\n" + MATH_INSTRUCTION},
    ]


def chat_text(tokenizer, prompt: str, response: Optional[str] = None) -> str:
    """Chat-formatted text of a prompt, optionally followed by an assistant response.

    ``response=None`` gives the question-only text (no assistant turn).
    ``response=""`` gives an *empty, closed* assistant turn (used to measure the
    prompt length in the dopamine feature extraction).
    """
    messages = _math_messages(prompt)
    if response is not None:
        messages.append({"role": "assistant", "content": response.strip()})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def plain_text(prompt: str, response: Optional[str] = None) -> str:
    """Text used when no chat template is applied."""
    if response is None:
        return prompt.strip()
    return (prompt + "\n\n" + response).strip()


def probe_text(tokenizer, prompt: str, response: Optional[str], apply_chat_template: bool) -> str:
    """Value-probe input text (chat-formatted or plain)."""
    if apply_chat_template:
        return chat_text(tokenizer, prompt, response)
    return plain_text(prompt, response)


def user_start_token_index(full_text: str, prompt_text: str,
                           attention_mask_row: torch.Tensor,
                           offsets_row: torch.Tensor) -> int:
    """First token whose character span ends after the start of ``prompt_text`` in ``full_text``.

    Returns 0 if the prompt is not found. Used so that the value probe ignores
    the system-prompt tokens of the chat template.
    """
    pos_char = full_text.find(prompt_text.strip())
    if pos_char < 0:
        pos_char = 0
    length = int(attention_mask_row.sum().item())
    for j in range(length):
        _, end = offsets_row[j].tolist()
        if end > pos_char:
            return j
    return 0


def split_into_paragraphs(text: str) -> List[str]:
    """Split a response into non-empty paragraphs at blank lines (``\\n\\n``)."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def rollout_prefix(tokenizer, prompt: str, partial_response: str) -> str:
    """Generation prefix for Monte-Carlo rollouts from paragraph boundary ``t``.

    * ``partial_response == ""`` (t = 0): chat template with a generation prompt.
    * otherwise: the partial response is placed in the assistant turn and a
      trailing end-of-turn token is removed only if the text *ends* with it.
      For Qwen2.5 templates the text ends with ``<|im_end|>\\n``, so nothing is
      removed and the rollout continues after a closed assistant turn. This is
      the behaviour of the paper's code and is kept as is.
    """
    messages = _math_messages(prompt)
    if not partial_response:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": partial_response})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    for eos in ["<|im_end|>", "<|eot_id|>", "</s>"]:
        if text.endswith(eos):
            text = text[: -len(eos)]
            break
    return text
