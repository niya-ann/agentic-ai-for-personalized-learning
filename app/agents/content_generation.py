"""
Content Generation Agent.

Template-based by default (MVP requirement, Section 5/6): produces a short
explanation + one practice question per concept. An optional local Ollama model
can be plugged in later behind USE_LLM -- the system must work with zero LLM.
"""
import random
from app import kg as kg_mod

USE_LLM = False  # flip to True only if you have Ollama running locally (Phase 9)

TEMPLATE_EXPLANATIONS = {
    "variables": "A variable stores a value under a name, e.g. `x = 5`. You can reassign it later.",
    "operators": "Operators combine values: arithmetic (+ - * /), comparison (== <), logical (and/or).",
    "conditionals": "`if`/`elif`/`else` let your program branch based on a condition.",
    "loops": "`for` loops iterate over a sequence; `while` loops repeat until a condition is false.",
    "functions": "`def name(params):` groups reusable logic; `return` sends a value back to the caller.",
    "lists": "A list `[1, 2, 3]` is an ordered, mutable collection; a tuple `(1, 2)` is immutable.",
    "dictionaries": "A dict `{'key': 'value'}` maps keys to values for fast lookup.",
    "strings": "Strings support slicing, `.split()`, `.join()`, f-strings for formatting.",
    "file_io": "`open(path)` reads/writes files; use `with open(...) as f:` to auto-close.",
    "exceptions": "`try/except` catches runtime errors so your program can recover gracefully.",
    "recursion": "A recursive function calls itself on a smaller input until it hits a base case.",
    "modules": "`import module_name` reuses code from another file or the standard library.",
}

TEMPLATE_QUESTIONS = {
    "variables": ("What does `x = 5` followed by `x = x + 1` leave `x` equal to?", "6"),
    "operators": ("What is the result of `7 // 2` in Python?", "3"),
    "conditionals": ("Which keyword runs code only if a prior `if` condition was False?", "else"),
    "loops": ("Which loop keyword repeats while a condition stays true?", "while"),
    "functions": ("Which keyword defines a function in Python?", "def"),
    "lists": ("Which data type is ordered but immutable: list or tuple?", "tuple"),
    "dictionaries": ("What do you call the pairs stored in a dict?", "key-value pairs"),
    "strings": ("Which method splits a string into a list by a delimiter?", "split"),
    "file_io": ("Which statement auto-closes a file after use?", "with"),
    "exceptions": ("Which block catches an error raised in a `try` block?", "except"),
    "recursion": ("What do you call the condition that stops a recursive call?", "base case"),
    "modules": ("Which keyword imports another Python file's code?", "import"),
}


def generate_content(concept_id, g):
    explanation = TEMPLATE_EXPLANATIONS.get(
        concept_id, f"{kg_mod.concept_name(g, concept_id)}: see resource notes."
    )
    question, answer = TEMPLATE_QUESTIONS.get(
        concept_id, (f"Briefly describe {kg_mod.concept_name(g, concept_id)}.", "")
    )
    resource = kg_mod.get_resource_text(g, concept_id)

    if USE_LLM:
        explanation = _try_llm_rewrite(explanation, concept_id, g)

    return {
        "concept_id": concept_id,
        "concept_name": kg_mod.concept_name(g, concept_id),
        "explanation": explanation,
        "question": question,
        "answer": answer,
        "resource": resource,
    }


def _try_llm_rewrite(text, concept_id, g):
    """Optional Ollama polish (Phase 9). Falls back to the template on any failure
    so a live demo never depends on model load time."""
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": f"Rewrite this programming explanation for a beginner in 2 sentences: {text}",
                "stream": False,
            },
            timeout=5,
        )
        if resp.ok:
            return resp.json().get("response", text).strip()
    except Exception:
        pass
    return text


def grade_answer(concept_id, given_answer, correct_answer):
    if not given_answer:
        return False
    return given_answer.strip().lower() in correct_answer.strip().lower() or \
        correct_answer.strip().lower() in given_answer.strip().lower()
