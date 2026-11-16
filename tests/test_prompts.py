import math

from medeval.prompts import CONDITIONS, FLASHCARD, mmlu_prompt
from medeval.run import letter_logprobs


def test_flashcard_prompt_matches_training_format():
    p = FLASHCARD.format(question="What is X?")
    assert p == "### Instruction:\nAnswer this question truthfully\n\n### Input:\nWhat is X?\n\n### Response:\n"


def test_finetuned_is_the_only_condition_with_adapter():
    scales = {name: s for name, _, s in CONDITIONS}
    assert scales == {"base_flashcard": 0.0, "base_chat": 0.0, "finetuned": 1.0}


def test_mmlu_prompt_lists_four_options():
    p = mmlu_prompt({"subject": "clinical_knowledge", "question": "Q?", "choices": ["a", "b", "c", "d"]})
    assert "clinical knowledge" in p and p.endswith("D. d\nAnswer:")


def test_letter_logprobs_picks_best_spelling_and_marks_missing():
    res = {"completion_probabilities": [{"top_logprobs": [
        {"token": " C", "logprob": -0.1}, {"token": "C", "logprob": -3.0},
        {"token": " A", "logprob": -2.0}, {"token": " The", "logprob": -4.0}]}]}
    lp = letter_logprobs(res)
    assert lp[0] == -2.0 and lp[2] == -0.1
    assert math.isinf(lp[1]) and math.isinf(lp[3])
