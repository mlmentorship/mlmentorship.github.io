from collections.abc import Callable


StepFunction = Callable[[tuple[int, ...]], list[float]]


def beam_search(
    step: StepFunction,
    *,
    bos_token: int,
    eos_token: int,
    beam_size: int,
    max_new_tokens: int,
    length_penalty: float = 0.0,
) -> tuple[int, ...]:
    """Return the highest-scoring completed sequence, including BOS and EOS.

    `step(prefix)` returns log probabilities for every token in the vocabulary.
    Keep at most `beam_size` live hypotheses. Finished hypotheses must not be
    expanded. Rank final sequences by log-probability divided by
    `generated_length ** length_penalty` when the penalty is positive.
    """
    raise NotImplementedError("implement bounded beam expansion, EOS handling, and final ranking")
