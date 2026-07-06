"""Prompt templates for the three coherence methods.

Ported verbatim from the EMNLP/llm-response-pipeline paper
(src/prompt_generation.py) so results are comparable to the original study.
"""

SYSTEM_PROMPT = "You are a helpful assistant who gives responses to questions."

# Few-shot exemplars for BASE models (no instruction-following). Prepended as a
# plain-text demonstration block so the base continues the pattern. Uses concepts
# NOT in the 30-item stimulus set to avoid leaking answers.
FEWSHOT_TRIPLET = (
    "Answer using only one word - Dog or Chair and not Cat. "
    "Which is more similar in semantic meaning to Cat?\nDog\n\n"
    "Answer using only one word - Table or Apple and not Desk. "
    "Which is more similar in semantic meaning to Desk?\nTable\n\n"
    "Answer using only one word - Boat or Hammer and not Ship. "
    "Which is more similar in semantic meaning to Ship?\nBoat\n\n"
)
FEWSHOT_PAIRWISE = (
    "Answer with only one number from 1 to 7: How semantically similar is Cat and Dog?\n6\n\n"
    "Answer with only one number from 1 to 7: How semantically similar is Table and Apple?\n2\n\n"
    "Answer with only one number from 1 to 7: How semantically similar is Ship and Boat?\n7\n\n"
)


def triplet_prompt(anchor: str, concept1: str, concept2: str) -> str:
    """Anchored similarity: which of two items is more similar to the anchor."""
    return (
        f"Answer using only one word - {concept1} or {concept2} and not {anchor}. "
        f"Which is more similar in semantic meaning to {anchor}?"
    )


def pairwise_prompt(a: str, b: str) -> str:
    """1..7 semantic similarity rating."""
    return (
        "Answer with only one number from 1 to 7, considering 1 as 'extremely "
        "dissimilar', 2 as 'very dissimilar', 3 as 'likely dissimilar', 4 as "
        "'neutral', 5 as 'likely similar', 6 as 'very similar', and 7 as "
        f"'extremely similar': How semantically similar is {a} and {b}?"
    )


def listing_prompt(concept: str) -> str:
    """Free feature listing: ask the model to list the properties of a concept.

    Kept close to classic feature-norm elicitation (McRae/Leuven style): ask for
    a plain list of properties, one per line."""
    return (
        f"List the features and properties of a {concept}. "
        "Give a plain list, one property per line, no explanations."
    )


def feature_prompt(feature: str, concept: str) -> str:
    """True/False feature verification (one-shot, matching the paper)."""
    return (
        "Q: Is the property [is female] true for the concept [book]? \n A: False \n "
        "Q: Is the property [can be digital] true for the concept [book] \n A: True \n "
        "In one word True/False, answer the following question "
        f"Q: Is the property [{feature}] true for [{concept}]? \n A:"
    )


# Map method name -> (builder, expected_input_arity)
BUILDERS = {
    "triplet": (triplet_prompt, 3),
    "pairwise": (pairwise_prompt, 2),
    "feature": (feature_prompt, 2),
}
