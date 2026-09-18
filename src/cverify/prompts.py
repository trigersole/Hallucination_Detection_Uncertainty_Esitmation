"""Fixed, paired verification and control prompts."""

VERIFY_TEMPLATES = (
    "Does the proposed answer correctly answer the question?",
    "Assess the factual correctness of this proposed answer.",
    "Is the proposed answer factually accurate for this question?",
)

CONTROL_TEMPLATES = (
    "Does the proposed answer use readable formatting for the question?",
    "Assess the formatting quality of this proposed answer.",
    "Is the proposed answer clearly formatted for this question?",
)


def evaluation_prompt(question: str, answer: str, instruction: str) -> str:
    """Build a prompt with a common response cue for state extraction."""
    return (
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n"
        f"Task: {instruction}\n"
        "Respond with exactly one label: Correct or Incorrect.\n"
        "Label:"
    )


def generation_prompt(question: str) -> str:
    return (
        "Answer the factual question briefly. Give only the answer and no explanation.\n"
        f"Question: {question}\nAnswer:"
    )

