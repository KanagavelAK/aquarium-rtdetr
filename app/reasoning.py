"""The reasoning layer: routing, structured reasoning, confidence guardrail.

Hand written on purpose. No agent framework is used or needed -- the whole
control flow is the three functions below, called in order by the API:

    route()      decide whether the detector has to run at all, and what the
                 question is asking for (which animals, count or rank or presence)
    guardrail()  decide, from the census facts, whether an honest answer exists
    compose()    put the facts into plain language

The guardrail runs BEFORE the language model sees anything, and it is
deterministic. A model asked to grade its own confidence will talk itself
into an answer; a rule that says "11 of 14 fish boxes overlap another fish
box, so the count is unreliable" will not.

The language model is optional. With no API key configured the layer answers
from templates over the same structured facts, so the API is fully runnable
offline. Set ANTHROPIC_API_KEY to enable the fluent path.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.scene import ASSERTION_FLOOR, plural

log = logging.getLogger("aquarium.reasoning")

MODEL_ID = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
LOW_CONFIDENCE_FLOOR = ASSERTION_FLOOR

# A count is refused when more than this fraction of a class's boxes overlap
# another box of the same class (a school, a huddle, a pile).
CROWDING_MAX = float(os.getenv("CROWDING_MAX", "0.34"))
# A count is refused when fewer than this fraction of a class's boxes clear
# the assertion floor.
CONFIDENT_FRACTION_MIN = float(os.getenv("CONFIDENT_FRACTION_MIN", "0.5"))
# A ranking is refused when the top two counts are closer than this.
RANK_MARGIN_FRACTION = float(os.getenv("RANK_MARGIN_FRACTION", "0.15"))

# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------

CLASSES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]

# How people refer to each class. Longer phrases are matched first and blanked
# out, so "sea star" never leaks a stray "star", and "jelly fish" never counts
# as "fish".
SYNONYMS = {
    "starfish": ["starfishes", "starfish", "sea stars", "sea star", "sea-stars", "sea-star",
                 "star fish", "seastars", "seastar"],
    "jellyfish": ["jellyfishes", "jellyfish", "jelly fish", "jellies", "jelly"],
    "stingray": ["stingrays", "stingray", "sting rays", "sting ray", "sting-rays", "sting-ray",
                 "rays", "ray"],
    "fish": ["fishes", "fish"],
    "penguin": ["penguins", "penguin"],
    "puffin": ["puffins", "puffin"],
    "shark": ["sharks", "shark"],
}
CLASS_PATTERNS = [(label, re.compile(r"\b(?:" + "|".join(re.escape(s) for s in syns) + r")\b", re.I))
                  for label, syns in SYNONYMS.items()]

GENERIC_WORDS = {
    "animal", "animals", "creature", "creatures", "species", "kind", "kinds", "type", "types",
    "object", "objects", "thing", "things", "life", "wildlife", "population", "inhabitants",
    "everything", "anything", "something", "critters", "marine", "sea", "underwater",
}
SCENE_WORDS = {"tank", "aquarium", "exhibit", "enclosure", "pool", "water", "display", "habitat"}

IMAGE_REFERENCE = re.compile(
    r"\b(image|picture|photo|photograph|frame|shot|scene|here|visible|shown|see|"
    r"this|that|these|those|there)\b", re.I)

# Questions that are plainly not about this image at all.
NON_VISUAL_PATTERN = re.compile(
    r"\b(who\s+(?:are|is)\s+you|what\s+model|your\s+(?:name|architecture|training)|"
    r"capital\s+of|weather\s+(?:today|tomorrow)|what\s+time\s+is\s+it|tell\s+me\s+a\s+joke|"
    r"how\s+do\s+i\s+(?:train|install|deploy|run)|what\s+is\s+the\s+meaning|"
    r"translate|write\s+(?:me\s+)?(?:a|some)\s+(?:poem|code|essay|story)|"
    r"how\s+does\s+(?:rt-?detr|the\s+model|detection)\s+work|"
    r"what\s+(?:is|are)\s+(?:a|an)\s+(?:fish|jellyfish|penguin|puffin|shark|starfish|stingray)|"
    r"define|definition|"
    r"do\s+(?:fish|sharks|penguins|puffins|jellyfish|starfish|stingrays)\s+(?:sleep|dream|feel\s+pain))\b", re.I)

# Things a visitor may reasonably ask that this detector genuinely cannot
# measure: species-level identity, health, size, age, sex, behaviour, colour,
# water conditions. Refused up front, without running the detector.
OUT_OF_SCOPE_VISUAL = re.compile(
    r"\b(what\s+(?:kind|kinds|type|types|sort|species|breed)\s+of|which\s+species|what\s+species|"
    r"species\s+(?:is|are)\s+(?:this|that|these|those|it|they)|identify|"
    r"clownfish|clown\s+fish|goldfish|tuna|salmon|angelfish|nemo|dory|"
    r"great\s+white|hammerhead|tiger\s+shark|nurse\s+shark|reef\s+shark|whale|dolphin|"
    r"turtle|octopus|seal|otter|crab|lobster|eel|coral|manta|emperor|king\s+penguin|"
    r"healthy|health|sick|ill|injured|hurt|dying|dead|alive|disease|diseased|"
    r"how\s+(?:big|large|small|long|tall|heavy|old|deep|warm|cold)|size|length|weight|weigh|"
    r"age|old|young|male|female|gender|sex|pregnant|baby|babies|juvenile|adult|"
    r"colou?rs?|pattern|stripes?|spotted|"
    r"temperature|salinity|\bph\b|clean|dirty|murky|clear|"
    r"hungry|feeding|eating|fed|food|"
    r"named?|called|happy|sad|mood|feel|feels|"
    r"swimming\s+(?:fast|slowly|towards|away)|direction|speed|"
    r"time\s+of\s+day|brand|logo|text|sign|read)\b", re.I)

COUNT_PATTERN = re.compile(r"\b(how\s+many|count|number\s+of|total|how\s+much|tally|census)\b", re.I)
KINDS_PATTERN = re.compile(r"\b(species|kinds?|types?|sorts?|different|distinct|varieties|variety|diverse|diversity)\b", re.I)
MOST_PATTERN = re.compile(
    r"\b(most\s+common|most\s+frequent|most\s+numerous|most\s+abundant|most\s+of|dominant|dominates?|"
    r"majority|biggest\s+group|largest\s+group|largest\s+number|highest\s+number|"
    r"appears?\s+(?:the\s+)?most|see\s+(?:the\s+)?most|most\s+often|the\s+most|mostly|"
    r"main\s+animal|primary\s+animal)\b", re.I)
LEAST_PATTERN = re.compile(
    r"\b(least\s+common|least\s+frequent|least\s+numerous|least\s+abundant|rarest|fewest|"
    r"smallest\s+group|smallest\s+number|lowest\s+number|minority|the\s+least|least\s+of)\b", re.I)
COMPARISON_PATTERN = re.compile(
    r"\b(more\s+.*?\s+than|fewer\s+.*?\s+than|less\s+.*?\s+than|outnumber|outnumbers|outnumbered|"
    r"compared?\s+(?:to|with)|versus|vs\.?|as\s+many\s+.*?\s+as)\b", re.I)
PRESENCE_PATTERN = re.compile(
    r"\b(is\s+there|are\s+there|any|anyone|anybody|someone|is\s+a|is\s+an|"
    r"does\s+it\s+(?:show|contain|have)|do\s+you\s+see|can\s+you\s+see|can\s+i\s+see|"
    r"contains?|present|visible|spot|is\s+(?:this|that|it)\s+an?|are\s+these)\b", re.I)

KIND_COUNT = "count"
KIND_KINDS = "count_kinds"
KIND_RANKING = "ranking"
KIND_COMPARISON = "comparison"
KIND_PRESENCE = "presence"
KIND_SUMMARY = "summary"
KIND_NOT_IMAGE = "not_about_the_image"
KIND_OUT_OF_SCOPE = "out_of_detector_scope"


@dataclass
class Route:
    needs_detection: bool
    kind: str
    rationale: str
    targets: List[str] = field(default_factory=list)   # classes the question names, in order
    direction: str = ""                                # ranking: "most" or "least"
    decided_by: str = "rules"


def extract_targets(question: str) -> List[str]:
    """Classes named in the question, in the order they appear."""
    text = question.lower()
    found = []
    for label, pattern in CLASS_PATTERNS:
        for match in pattern.finditer(text):
            found.append((match.start(), label))
            # blank the span so a shorter synonym of another class cannot re-match it
            text = text[:match.start()] + " " * (match.end() - match.start()) + text[match.end():]
    found.sort()
    ordered = []
    for _, label in found:
        if label not in ordered:
            ordered.append(label)
    return ordered


def route(question: str) -> Route:
    """Decide whether the detector has to run, and what for.

    Deterministic by design. Routing is a cheap, high-traffic decision with a
    small, closed vocabulary, so a rule set is both faster and easier to defend
    than a model call, and it cannot hallucinate a route.
    """
    q = (question or "").strip()
    if not q:
        return Route(False, KIND_NOT_IMAGE, "the question was empty")

    if NON_VISUAL_PATTERN.search(q):
        return Route(False, KIND_NOT_IMAGE,
                     "the question asks about something other than the contents "
                     "of the image, so running the detector would not inform it")

    targets = extract_targets(q)
    tokens = set(re.findall(r"[a-z-]+", q.lower()))
    mentions_generic = bool(tokens & GENERIC_WORDS)
    mentions_scene = bool(tokens & SCENE_WORDS)
    asks_count = bool(COUNT_PATTERN.search(q))
    asks_kinds = bool(KINDS_PATTERN.search(q))

    # "How many kinds of animal are there?" is answerable: it is a count of the
    # detector's own categories. "How many kinds of fish?" is not: within a
    # category the detector cannot tell species apart.
    if asks_count and asks_kinds:
        if targets:
            return Route(False, KIND_OUT_OF_SCOPE,
                         "the question asks how many varieties of {} there are, and this "
                         "detector recognises {} as one category without telling species "
                         "apart".format(plural(targets[0], 2), plural(targets[0], 2)))
        return Route(True, KIND_KINDS,
                     "the question asks how many different kinds of animal are present, "
                     "which is a count over the detector's categories")

    if OUT_OF_SCOPE_VISUAL.search(q):
        return Route(False, KIND_OUT_OF_SCOPE,
                     "the question is about the image but asks for an attribute this "
                     "detector does not predict (species, health, size, age, sex, colour "
                     "or behaviour), so no amount of detection would answer it")

    if not targets and not mentions_generic and not mentions_scene:
        refers_to_image = bool(IMAGE_REFERENCE.search(q))
        descriptive = bool(re.search(r"\b(what|describe|summari[sz]e|tell\s+me|explain|going\s+on)\b", q, re.I))
        if refers_to_image and descriptive and not asks_count and not PRESENCE_PATTERN.search(q):
            return Route(True, KIND_SUMMARY,
                         "the question asks what the image shows in general")
        if refers_to_image or asks_count or PRESENCE_PATTERN.search(q):
            return Route(False, KIND_OUT_OF_SCOPE,
                         "the question seems to be about the image but names nothing among "
                         "the seven categories this detector predicts, so detection cannot "
                         "answer it")
        return Route(False, KIND_NOT_IMAGE,
                     "the question names nothing this detector can see and does "
                     "not refer to the image")

    if COMPARISON_PATTERN.search(q) and len(targets) >= 2:
        return Route(True, KIND_COMPARISON,
                     "the question compares the number of {} with the number of {}".format(
                         plural(targets[0], 2), plural(targets[1], 2)),
                     targets=targets[:2])
    if MOST_PATTERN.search(q):
        return Route(True, KIND_RANKING,
                     "the question asks which animal is the most common, which needs "
                     "reliable counts for every class present", targets=targets, direction="most")
    if LEAST_PATTERN.search(q):
        return Route(True, KIND_RANKING,
                     "the question asks which animal is the least common, which needs "
                     "reliable counts for every class present", targets=targets, direction="least")
    if asks_count:
        what = ", ".join(plural(t, 2) for t in targets) if targets else "all animals"
        return Route(True, KIND_COUNT,
                     "the question asks for a count of {} in the image".format(what), targets=targets)
    if PRESENCE_PATTERN.search(q) and (targets or mentions_generic):
        what = ", ".join(plural(t, 2) for t in targets) if targets else "any animal"
        return Route(True, KIND_PRESENCE,
                     "the question asks whether {} appear in the image".format(what), targets=targets)
    if targets and not mentions_generic:
        # "Sharks?" or "Tell me about the penguins" -- a bare mention reads as presence.
        return Route(True, KIND_PRESENCE,
                     "the question names {} and asks nothing more specific, so it is "
                     "answered as a presence question".format(", ".join(plural(t, 2) for t in targets)),
                     targets=targets)
    return Route(True, KIND_SUMMARY,
                 "the question is about the contents of the image in general")


# ---------------------------------------------------------------------------
# Confidence guardrail
# ---------------------------------------------------------------------------

@dataclass
class Guard:
    sufficient: bool
    reasons: List[str] = field(default_factory=list)


def _count_problems(cf) -> List[str]:
    """Why a per-class count cannot be trusted. Empty means it can."""
    reasons = []
    if cf.count == 0:
        return reasons
    name = plural(cf.label, 2)
    if cf.crowding > CROWDING_MAX:
        reasons.append("{} of the {} {} boxes overlap another {} box (crowding {:.0%}), so "
                       "individual animals in that group cannot be separated and the count "
                       "may be off in either direction".format(
                           cf.crowded, cf.count, cf.label, cf.label, cf.crowding))
    if cf.confident / cf.count < CONFIDENT_FRACTION_MIN:
        reasons.append("only {} of the {} {} detections scored at or above {:.2f}, so most of "
                       "that count rests on weak evidence".format(
                           cf.confident, cf.count, cf.label, LOW_CONFIDENCE_FLOOR))
    if cf.conflicts:
        reasons.append("{} of the {} {} box{} also carr{} a {} label on the same pixels, so the "
                       "detector is unsure what {} animal{} {}".format(
                           cf.conflicts, cf.count, cf.label, "es" if cf.count != 1 else "",
                           "y" if cf.conflicts != 1 else "ies", " or ".join(cf.conflicts_with),
                           "those" if cf.conflicts != 1 else "that",
                           "s" if cf.conflicts != 1 else "", "are" if cf.conflicts != 1 else "is"))
    return reasons


def guardrail(decision: Route, facts) -> Guard:
    """Decide whether the census facts support an honest answer.

    Runs before the language model and never consults it.
    """
    reasons = []
    kind = decision.kind
    targets = decision.targets

    if not facts.detections:
        reasons.append("the detector returned no animals above the confidence "
                       "floor of {:.2f}".format(facts.confidence_floor))
        return Guard(False, reasons)

    if facts.max_confidence < LOW_CONFIDENCE_FLOOR:
        reasons.append("every detection scored below {:.2f}, which is too weak to "
                       "assert anything about this image".format(LOW_CONFIDENCE_FLOOR))
        return Guard(False, reasons)

    if kind == KIND_COUNT:
        if targets:
            for t in targets:
                reasons += _count_problems(facts.facts_for(t))
        else:
            for cf in facts.classes.values():
                reasons += _count_problems(cf)

    elif kind == KIND_KINDS:
        weak = [cf.label for cf in facts.classes.values() if cf.confident == 0]
        if weak:
            reasons.append("{} appear only as detections below {:.2f}, so it is not clear "
                           "whether {} really present".format(
                               ", ".join(plural(w, 2) for w in weak), LOW_CONFIDENCE_FLOOR,
                               "they are" if len(weak) > 1 else "it is"))
        if facts.conflicts:
            reasons.append("{} animal{} carry two different labels on the same pixels, so "
                           "the number of distinct kinds is uncertain".format(
                               len(facts.conflicts), "s" if len(facts.conflicts) != 1 else ""))

    elif kind == KIND_RANKING:
        ranking = facts.ranking
        if decision.direction == "least":
            ranking = list(reversed(ranking))
        if len(ranking) >= 2:
            (a, na), (b, nb) = ranking[0], ranking[1]
            margin = max(1, math.ceil(RANK_MARGIN_FRACTION * max(na, nb)))
            if abs(na - nb) < margin:
                reasons.append("{} ({}) and {} ({}) are too close to rank with confidence; "
                               "a single missed or duplicated box would change the answer".format(
                                   plural(a, 2), na, plural(b, 2), nb))
            reasons += _count_problems(facts.facts_for(a))
            # The runner-up only matters if an error in its count could flip
            # the order, i.e. when it would need less than doubling to catch up.
            if na < 2 * nb:
                reasons += _count_problems(facts.facts_for(b))
        else:
            reasons += _count_problems(facts.facts_for(ranking[0][0]))

    elif kind == KIND_COMPARISON:
        for t in targets[:2]:
            reasons += _count_problems(facts.facts_for(t))

    elif kind == KIND_PRESENCE:
        for t in targets:
            cf = facts.facts_for(t)
            if cf.count and cf.max_confidence < LOW_CONFIDENCE_FLOOR:
                reasons.append("{} {} detected, but every one scored below {:.2f}, which is "
                               "too weak to confirm".format(
                                   cf.count, plural(t, cf.count), LOW_CONFIDENCE_FLOOR))
            if cf.conflicts:
                reasons.append("the {} candidate{} also labelled as {} on the same pixels, "
                               "so the detector is unsure what it is".format(
                                   t, "s are" if cf.conflicts != 1 else " is",
                                   " or ".join(cf.conflicts_with)))

    return Guard(not reasons, reasons)


# ---------------------------------------------------------------------------
# Answer composition
# ---------------------------------------------------------------------------

def _n(label, n):
    return "{} {}".format(n, plural(label, n))


def _list(parts):
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _deterministic_answer(question: str, decision: Route, facts) -> str:
    kind, targets = decision.kind, decision.targets
    ranking = facts.ranking

    if kind == KIND_COUNT:
        if targets:
            parts = []
            for t in targets:
                cf = facts.facts_for(t)
                parts.append(_n(t, cf.count) if cf.count else "no {}".format(plural(t, 2)))
            text = "I count {} in this image.".format(_list(parts))
            if len(targets) == 1 and not facts.facts_for(targets[0]).count:
                text = "I do not detect any {} in this image above the {:.2f} threshold.".format(
                    plural(targets[0], 2), facts.confidence_floor)
        else:
            text = "I count {} in this image: {}.".format(
                _n("animal", facts.total), _list([_n(k, v) for k, v in ranking]))
        if facts.duplicates_suppressed:
            text += " ({} overlapping duplicate box{} collapsed before counting.)".format(
                facts.duplicates_suppressed, "es" if facts.duplicates_suppressed != 1 else "")
        return text

    if kind == KIND_KINDS:
        return "I can distinguish {} kind{} of animal here: {}.".format(
            facts.kinds, "s" if facts.kinds != 1 else "",
            _list(["{} ({})".format(k, v) for k, v in ranking]))

    if kind == KIND_RANKING:
        if decision.direction == "least":
            label, n = ranking[-1]
            text = "The least common animal detected is the {}, with {} of the {} animals".format(
                label, n, facts.total)
            if len(ranking) > 1:
                text += "; the most common is the {} with {}".format(ranking[0][0], ranking[0][1])
            return text + "."
        label, n = ranking[0]
        text = "The most common animal is the {}: {} of the {} animals detected".format(
            label, n, facts.total)
        if len(ranking) > 1:
            text += ", ahead of {}".format(_n(ranking[1][0], ranking[1][1]))
        return text + "."

    if kind == KIND_COMPARISON:
        a, b = targets[0], targets[1]
        na, nb = facts.facts_for(a).count, facts.facts_for(b).count
        if na == nb:
            return "They are equal: I count {} and {}.".format(_n(a, na), _n(b, nb))
        hi, lo = (a, b) if na > nb else (b, a)
        return "There are more {} than {}: {} versus {}.".format(
            plural(hi, 2), plural(lo, 2), _n(hi, max(na, nb)), _n(lo, min(na, nb)))

    if kind == KIND_PRESENCE:
        if not targets:
            return "Yes. I detect {}: {}.".format(
                _n("animal", facts.total), _list([_n(k, v) for k, v in ranking]))
        yes = [t for t in targets if facts.facts_for(t).count]
        no = [t for t in targets if not facts.facts_for(t).count]
        parts = []
        if yes:
            parts.append("Yes, I detect {} (highest confidence {:.2f})".format(
                _list([_n(t, facts.facts_for(t).count) for t in yes]),
                max(facts.facts_for(t).max_confidence for t in yes)))
        if no:
            parts.append("{} do not detect any {} above the {:.2f} threshold".format(
                "I" if not yes else "but I", _list([plural(t, 2) for t in no]), facts.confidence_floor))
        return ". ".join(p if i == 0 else p[0].upper() + p[1:] for i, p in enumerate(parts)) + "."

    # summary
    text = "This looks like an aquarium scene with {} across {} kind{}: {}.".format(
        _n("animal", facts.total), facts.kinds, "s" if facts.kinds != 1 else "",
        _list([_n(k, v) for k, v in ranking]))
    if len(ranking) > 1:
        text += " The most common is the {}.".format(ranking[0][0])
    return text


SYSTEM_PROMPT = """You answer questions about a photograph taken at a public aquarium.

You cannot see the photograph. You are given the structured output of an
RT-DETR object detector fine-tuned on seven classes: fish, jellyfish, penguin,
puffin, shark, starfish and stingray. A separate deterministic step has already
collapsed duplicate boxes, measured crowding and label conflicts, and decided
that these facts are sufficient to answer the question.

Rules:
- Answer only from the structured facts given. Never infer animals, species,
  attributes or context that are not in them.
- Be direct and plain. Two or three sentences at most.
- Give the numbers that matter and say what they are counts of.
- Never describe your own confidence as a percentage. State what was detected.
- "fish" is a category, not a species: never name a species."""


def _llm_answer(question: str, decision: Route, facts) -> Optional[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import json

        import anthropic
        from pydantic import BaseModel

        class Answer(BaseModel):
            answer: str

        client = anthropic.Anthropic()
        payload = {k: v for k, v in facts.to_dict().items() if k != "detections"}
        response = client.messages.parse(
            model=MODEL_ID,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{
                "role": "user",
                "content": ("Question: {}\nQuestion type: {}\nClasses the question names: {}\n\n"
                            "Detector facts:\n{}".format(
                                question, decision.kind, decision.targets or "none",
                                json.dumps(payload, indent=2, sort_keys=True))),
            }],
            output_format=Answer,
        )
        if response.stop_reason == "refusal":
            log.warning("model declined to answer, falling back to templates")
            return None
        return response.parsed_output.answer.strip()
    except Exception as exc:
        log.warning("language model step failed (%s), falling back to templates", exc)
        return None


def compose(question: str, decision: Route, facts) -> tuple:
    """Return (answer_text, source) where source is 'llm' or 'template'."""
    text = _llm_answer(question, decision, facts)
    if text:
        return text, "llm"
    return _deterministic_answer(question, decision, facts), "template"


def insufficient_message(kind: str, reasons: List[str]) -> str:
    lead = "I do not have enough information to answer that confidently."
    if kind == KIND_NOT_IMAGE:
        lead = ("That question is not about the contents of the image, so I did "
                "not run the detector.")
    elif kind == KIND_OUT_OF_SCOPE:
        lead = ("I cannot answer that. This detector recognises seven kinds of aquarium "
                "animal as categories (fish, jellyfish, penguin, puffin, shark, starfish, "
                "stingray); it cannot identify species, or judge health, size, age, sex, "
                "colour or behaviour.")
    if not reasons:
        return lead
    return lead + " " + " ".join(r[0].upper() + r[1:] + "." for r in reasons)
