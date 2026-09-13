"""Tests for the reasoning layer. No GPU and no checkpoint required, because
the layer only ever consumes plain dictionaries.

These double as the worked examples in the memo: the fish-school case near
the bottom is the "insufficient information" example to quote.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import reasoning
from app.scene import build_scene


def det(label, conf, box):
    return {"label": label, "confidence": conf, "box_xyxy": box}


def scene(dets, width=1600, height=1200, floor=0.25):
    return build_scene(dets, width, height, floor)


def answer(question, dets):
    """The whole /ask path, minus HTTP: route -> guard -> compose."""
    decision = reasoning.route(question)
    if not decision.needs_detection:
        return decision, None, reasoning.insufficient_message(decision.kind, [])
    facts = scene(dets)
    guard = reasoning.guardrail(decision, facts)
    if not guard.sufficient:
        return decision, guard, reasoning.insufficient_message(decision.kind, guard.reasons)
    return decision, guard, reasoning.compose(question, decision, facts)[0]


# A tidy tank: three sharks, two stingrays, one starfish, no overlap.
TIDY = [
    det("shark", 0.91, [100, 100, 500, 300]),
    det("shark", 0.88, [600, 120, 1000, 330]),
    det("shark", 0.79, [1100, 150, 1500, 340]),
    det("stingray", 0.86, [200, 600, 600, 800]),
    det("stingray", 0.74, [900, 650, 1300, 850]),
    det("starfish", 0.69, [1400, 1000, 1550, 1150]),
]


# --- routing ---------------------------------------------------------------

def test_count_question_names_its_target_and_calls_the_detector():
    r = reasoning.route("How many sharks are in this tank?")
    assert r.needs_detection is True
    assert r.kind == reasoning.KIND_COUNT
    assert r.targets == ["shark"]


def test_most_common_question_is_a_ranking():
    r = reasoning.route("What is the most common animal here?")
    assert r.kind == reasoning.KIND_RANKING and r.direction == "most"


def test_comparison_keeps_both_classes_in_question_order():
    r = reasoning.route("Are there more fish than jellyfish?")
    assert r.kind == reasoning.KIND_COMPARISON
    assert r.targets == ["fish", "jellyfish"]   # "jellyfish" must not also count as "fish"


def test_synonyms_map_to_classes():
    assert reasoning.extract_targets("any sea stars or rays?") == ["starfish", "stingray"]


def test_how_many_kinds_is_answerable_but_kinds_of_fish_is_not():
    assert reasoning.route("How many different kinds of animal are there?").kind == reasoning.KIND_KINDS
    r = reasoning.route("How many kinds of fish are there?")
    assert r.needs_detection is False and r.kind == reasoning.KIND_OUT_OF_SCOPE


def test_species_and_health_questions_are_refused_without_the_detector():
    for q in ("What species of shark is that?", "Is the penguin healthy?",
              "How big is the stingray?", "Are there any baby penguins?"):
        r = reasoning.route(q)
        assert r.needs_detection is False, q
        assert r.kind == reasoning.KIND_OUT_OF_SCOPE, q


def test_question_about_the_image_but_outside_the_classes_is_out_of_scope():
    r = reasoning.route("How many people are in this image?")
    assert r.needs_detection is False
    assert r.kind == reasoning.KIND_OUT_OF_SCOPE


def test_unrelated_question_skips_the_detector():
    r = reasoning.route("What is the capital of France?")
    assert r.needs_detection is False
    assert r.kind == reasoning.KIND_NOT_IMAGE


# --- scene facts -------------------------------------------------------------

def test_duplicate_queries_on_one_animal_are_collapsed():
    facts = scene([
        det("shark", 0.90, [100, 100, 500, 300]),
        det("shark", 0.31, [104, 96, 508, 306]),   # RT-DETR second query on the same shark
    ])
    assert facts.counts == {"shark": 1}
    assert facts.duplicates_suppressed == 1
    assert len(facts.detections) == 2            # the raw output is still reported


def test_crowding_is_measured_per_class():
    facts = scene([
        det("fish", 0.8, [100, 100, 200, 160]),
        det("fish", 0.7, [140, 110, 240, 170]),    # overlaps the first
        det("fish", 0.9, [800, 800, 900, 860]),    # alone
    ])
    cf = facts.facts_for("fish")
    assert cf.count == 3 and cf.crowded == 2
    assert abs(cf.crowding - 0.667) < 0.01


def test_conflicting_labels_on_the_same_pixels_are_recorded():
    facts = scene([
        det("shark", 0.62, [100, 100, 500, 300]),
        det("stingray", 0.55, [110, 105, 505, 310]),
    ])
    assert len(facts.conflicts) == 1
    assert facts.facts_for("shark").conflicts_with == ["stingray"]


# --- guardrail and answers ---------------------------------------------------

def test_tidy_tank_count_is_answered():
    decision, guard, text = answer("How many sharks are in this tank?", TIDY)
    assert guard.sufficient
    assert text == "I count 3 sharks in this image."


def test_absent_class_is_an_answer_not_a_refusal():
    decision, guard, text = answer("Are there any penguins?", TIDY)
    assert guard.sufficient
    assert "do not detect any penguins" in text


def test_ranking_with_a_clear_margin_is_answered():
    decision, guard, text = answer("What is the most common animal here?", TIDY)
    assert guard.sufficient
    assert text.startswith("The most common animal is the shark: 3 of the 6 animals")


def test_comparison_is_answered_from_counts():
    decision, guard, text = answer("Are there more stingrays than sharks?", TIDY)
    assert guard.sufficient
    assert text == "There are more sharks than stingrays: 3 sharks versus 2 stingrays."


def test_no_detections_is_insufficient():
    decision, guard, text = answer("How many fish are there?", [])
    assert guard.sufficient is False
    assert "do not have enough information" in text


def test_fish_school_forces_insufficient_information():
    """The memo's worked example.

    Fourteen fish: three swimming alone and a school of eleven whose boxes
    (220 px wide, 90 px apart) each overlap a neighbour. Every box is
    individually plausible, but the detector cannot separate animals that
    occlude each other: a merged pair reads as one, a long fish reads as
    two. The honest answer is not "fourteen"; it is that the count is
    unreliable, and why. Presence, which needs only one confident box, is
    still answered on the same scene.
    """
    school = [det("fish", 0.6 + 0.02 * (i % 5), [100 + 90 * i, 400, 320 + 90 * i, 560]) for i in range(11)]
    loners = [det("fish", 0.7, [100, 900, 300, 1000]), det("fish", 0.66, [700, 950, 900, 1050]),
              det("fish", 0.61, [1300, 880, 1500, 980])]
    decision, guard, text = answer("How many fish are in this image?", school + loners)
    assert decision.kind == reasoning.KIND_COUNT
    assert guard.sufficient is False
    assert guard.reasons == ["11 of the 14 fish boxes overlap another fish box (crowding 79%), so "
                             "individual animals in that group cannot be separated and the count "
                             "may be off in either direction"]
    assert "do not have enough information" in text
    decision, guard, text = answer("Are there any fish?", school + loners)
    assert guard.sufficient and text.startswith("Yes, I detect 14 fish")


def test_close_ranking_is_refused():
    close = TIDY[:3] + [det("stingray", 0.8, [200, 600, 600, 800]),
                        det("stingray", 0.8, [900, 650, 1300, 850]),
                        det("stingray", 0.8, [100, 900, 500, 1100])]   # 3 sharks vs 3 stingrays
    decision, guard, text = answer("Which animal is the most common?", close)
    assert guard.sufficient is False
    assert any("too close to rank" in r for r in guard.reasons)


def test_label_conflict_blocks_counting_that_class():
    dets = TIDY + [det("stingray", 0.52, [104, 98, 505, 305])]   # on top of the first shark
    decision, guard, text = answer("How many sharks are there?", dets)
    assert guard.sufficient is False
    assert any("stingray" in r for r in guard.reasons)


def test_all_weak_detections_are_insufficient():
    dets = [det("puffin", 0.31, [100, 100, 200, 200]), det("puffin", 0.28, [400, 100, 500, 200])]
    decision, guard, text = answer("How many puffins?", dets)
    assert guard.sufficient is False
    assert any("below 0.45" in r for r in guard.reasons)
