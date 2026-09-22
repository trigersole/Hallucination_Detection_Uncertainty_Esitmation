import math

from cverify.data import exact_match, normalize_answer
from cverify.prompts import CONTROL_TEMPLATES, VERIFY_TEMPLATES, evaluation_prompt, generation_prompt
from cverify.signals import lexical_cluster_statistics, rejection_and_disagreement


def test_answer_normalization_and_aliases():
    assert normalize_answer("The Paris!") == "paris"
    assert exact_match("George Orwell.", ("Orwell", "George Orwell"))
    assert not exact_match("London", ("Paris",))


def test_jsonl_offset(tmp_path):
    from cverify.data import load_jsonl

    path = tmp_path / "data.jsonl"
    path.write_text(
        '\n'.join(
            f'{{"id":"q{i}","question":"Question {i}?","answers":["a{i}"]}}'
            for i in range(5)
        ),
        encoding="utf-8",
    )
    assert [row.id for row in load_jsonl(path, limit=2, offset=2)] == ["q2", "q3"]


def test_cluster_statistics():
    consistency, entropy = lexical_cluster_statistics(["Paris", "paris.", "London"])
    assert consistency == 2 / 3
    assert entropy > 0


def test_rejection_is_distinct_from_disagreement():
    rejection, disagreement = rejection_and_disagreement([0.01, 0.01, 0.01])
    assert rejection == 0.99
    assert math.isclose(disagreement, 0.0, abs_tol=1e-10)


def test_prompts_are_paired_and_share_cue():
    assert len(VERIFY_TEMPLATES) == len(CONTROL_TEMPLATES) == 3
    prompt = evaluation_prompt("Q?", "A", VERIFY_TEMPLATES[0])
    assert prompt.endswith("Label:")


def test_generation_prompt_profiles():
    brief = generation_prompt("Can reindeer fly?")
    sentence = generation_prompt("Can reindeer fly?", "complete-sentence")
    assert brief == (
        "Answer the factual question briefly. Give only the answer and no explanation.\n"
        "Question: Can reindeer fly?\nAnswer:"
    )
    assert "one complete sentence" in sentence
    assert sentence.endswith("Question: Can reindeer fly?\nAnswer:")
