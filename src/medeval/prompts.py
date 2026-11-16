"""Prompt formats.

`flashcard` is the format the adapter was trained on. `chat` is Mistral
Instruct's own format, which is the fair way to ask the base model.
"""

FLASHCARD = "### Instruction:\nAnswer this question truthfully\n\n### Input:\n{question}\n\n### Response:\n"
CHAT = "[INST] {question} [/INST]"
STOP = ["###", "[INST]"]

# (condition name, prompt template, adapter scale)
CONDITIONS = [
    ("base_flashcard", FLASHCARD, 0.0),
    ("base_chat", CHAT, 0.0),
    ("finetuned", FLASHCARD, 1.0),
]

LETTERS = ["A", "B", "C", "D"]


def mmlu_prompt(item: dict) -> str:
    subject = item["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subject}.", "", item["question"].strip()]
    lines += [f"{L}. {c}" for L, c in zip(LETTERS, item["choices"], strict=True)]
    lines.append("Answer:")
    return "\n".join(lines)
