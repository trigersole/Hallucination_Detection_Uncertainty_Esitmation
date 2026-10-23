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

# Keep the original profile reproducible; corrected runs opt into yes-no-v2.
YES_NO_VERIFY_TEMPLATES = (
    "Does the proposed answer correctly answer the question?",
    "Is the proposed answer factually correct for this question?",
    "Is the proposed answer factually accurate for this question?",
)
YES_NO_CONTROL_TEMPLATES = (
    "Does the proposed answer use readable formatting for the question?",
    "Is the proposed answer formatted readably for this question?",
    "Is the proposed answer clearly formatted for this question?",
)
EVALUATION_PROFILES = {
    "legacy": (VERIFY_TEMPLATES, CONTROL_TEMPLATES, (" Correct", " Incorrect")),
    "yes-no-v2": (YES_NO_VERIFY_TEMPLATES, YES_NO_CONTROL_TEMPLATES, (" Yes", " No")),
    "assessment-v3": (
        ("Assess the factual correctness of the proposed answer for this question.",),
        ("Assess the formatting clarity of the proposed answer.",),
        (),  # Representation-only: no label vocabulary or probability scoring.
    ),
}


def evaluation_spec(profile: str) -> dict:
    verify, control, labels = EVALUATION_PROFILES[profile]
    spec = {
        "profile": profile,
        "verify_instructions": list(verify),
        "control_instructions": list(control),
        "labels": list(labels),
        "prompt_template": evaluation_prompt("{question}", "{answer}", "{instruction}", profile),
        "hidden_position": "last_prompt_token_before_response",
    }
    if profile == "assessment-v3":
        spec.update({
            "mode": "representation_only",
            "conditions": ["verify", "control", "none"],
            "none_prompt_template": evaluation_prompt("{question}", "{answer}", None, profile),
            "hidden_position": "last_token_of_Assessment_cue_without_generated_response",
        })
    return spec


def evaluation_prompt(question: str, answer: str, instruction: str | None, profile: str = "legacy") -> str:
    """Build a prompt with a common response cue for state extraction."""
    EVALUATION_PROFILES[profile]  # Reject unknown profiles before building a prompt.
    if profile == "assessment-v3":
        task = f"Task: {instruction}\n" if instruction is not None else ""
        return f"Question: {question}\nProposed answer: {answer}\n{task}Assessment:"
    if instruction is None:
        raise ValueError("The None condition is available only in assessment-v3")
    response = (
        "Judge only the property asked about in the task.\n"
        "Respond with exactly one label: Yes or No.\n"
        "Answer:"
        if profile == "yes-no-v2" else
        "Respond with exactly one label: Correct or Incorrect.\nLabel:"
    )
    return (
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n"
        f"Task: {instruction}\n"
        f"{response}"
    )


GENERATION_INSTRUCTIONS = {
    "concise": "Answer the question concisely.",
    "brief": "Answer the factual question briefly. Give only the answer and no explanation.",
    "complete-sentence": (
        "Answer the factual question in one complete sentence. "
        "Do not reply with only yes, no, or an isolated name. "
        "State the answer explicitly without adding unsupported details."
    ),
}


def generation_prompt(question: str, style: str = "brief") -> str:
    try:
        instruction = GENERATION_INSTRUCTIONS[style]
    except KeyError as exc:
        raise ValueError(f"Unknown generation prompt style: {style}") from exc
    return f"{instruction}\nQuestion: {question}\nAnswer:"
