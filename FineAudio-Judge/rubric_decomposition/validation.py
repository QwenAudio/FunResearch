from __future__ import annotations

from collections import Counter
import re
from typing import Any

import networkx as nx

from .models import (AtomicTuple, AudioSceneGraph, DependencyRecord,
                     Attribute, Event, GeneratedQuestion, Relation, RubricQuestion)
from .obligations import extract_prompt_obligations, normalize_quoted_content


_ZH_NUMERALS = {
    0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
    6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
}


def _zh_number(value: Any) -> str:
    """Render small integer values naturally without changing their meaning."""
    text = str(value).strip()
    if not text.isdigit():
        return text
    number = int(text)
    if number in _ZH_NUMERALS:
        return _ZH_NUMERALS[number]
    if 10 < number < 20:
        return "十" + _ZH_NUMERALS[number % 10]
    if 20 <= number < 100:
        tail = _ZH_NUMERALS[number % 10] if number % 10 else ""
        return _ZH_NUMERALS[number // 10] + "十" + tail
    return text


def _prompt_language(prompt: str) -> str:
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in prompt)
    latin = sum(ch.isascii() and ch.isalpha() for ch in prompt)
    return "zh" if cjk >= max(4, latin // 2) else "en"


def _natural_language_mismatch(value: Any, language: str) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
        return False
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    latin = sum(ch.isascii() and ch.isalpha() for ch in text)
    if language == "zh":
        # Acronyms may coexist with Chinese, while an all-English generated
        # phrase is a language mismatch.
        return cjk == 0 and latin >= 2
    return cjk > 0


def _verbatim_foreign_content(value: Any, prompt: str, language: str) -> bool:
    """Allow foreign-script content only when it is copied from the prompt.

    An English rubric may legitimately quote Chinese speech, and a Chinese
    rubric may quote an English utterance. The reference validator treated
    either case as a language error even though the question scaffold remained
    in the requested language. This exception never permits newly invented
    foreign text: every foreign-script span must occur verbatim in the prompt.
    """
    if not isinstance(value, str) or not _natural_language_mismatch(value, language):
        return False
    if language == "zh":
        return value.strip().casefold() in prompt.casefold()
    spans = re.findall(r"[\u4e00-\u9fff]+", value)
    return bool(spans) and all(span in prompt for span in spans)


def _duplicates(values: list[str]) -> list[str]:
    return sorted(key for key, count in Counter(values).items() if count > 1)


def _contains_value(text: str, value: Any) -> bool:
    """Conservative lexical guard for facts copied into the wrong layer."""
    if not isinstance(value, str):
        return False
    needle = value.strip().casefold().replace("’", "'").replace("‘", "'")
    haystack = text.strip().casefold().replace("’", "'").replace("‘", "'")
    if len(needle) < 2:
        return False
    if re.search(r"[\u4e00-\u9fff]", needle):
        if needle in haystack:
            return True
        # Natural Chinese yes/no syntax can insert a short copular marker
        # inside an otherwise exact property phrase: ``音调适中`` becomes
        # ``音调是否适中``. Permit only this closed set of grammatical
        # bridges, while preserving the value's character order and both
        # non-empty sides. This does not license paraphrases or extra facts.
        bridges = ("是否", "是不是", "为", "呈现为", "表现为")
        return any(
            f"{needle[:index]}{bridge}{needle[index:]}" in haystack
            for index in range(1, len(needle))
            for bridge in bridges
        )
    if re.search(
        rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])",
        haystack,
    ):
        return True

    def stems(source: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", source)
        output: list[str] = []
        for word in words:
            if len(word) > 5 and word.endswith("ing"):
                word = word[:-3]
                if len(word) > 2 and word[-2:] in {"pp", "tt", "nn", "gg", "mm", "rr"}:
                    word = word[:-1]
            elif len(word) > 4 and word.endswith("sses"):
                word = word[:-2]
            elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
                word = word[:-1]
            if len(word) > 4 and word.endswith("e"):
                word = word[:-1]
            output.append(word)
        return output

    needle_tokens = stems(needle)
    haystack_tokens = stems(haystack)
    if not needle_tokens:
        return False
    cursor = 0
    for token in haystack_tokens:
        if token == needle_tokens[cursor]:
            cursor += 1
            if cursor == len(needle_tokens):
                return True
    return False


def _remove_literal_value(label: str, value: Any) -> str | None:
    """Remove an exact property phrase while keeping a usable identity label."""
    if not isinstance(value, str) or not value.strip():
        return None
    updated = re.sub(re.escape(value.strip()), " ", label, flags=re.IGNORECASE)
    updated = re.sub(r"[_\s]+", " ", updated).strip(" _-–—·,，:：()（）")
    return updated if len(re.findall(r"[A-Za-z\u4e00-\u9fff]", updated)) >= 2 else None


def _canonical_quote_basis(prompt: str, value: Any) -> str | None:
    """Locate a quoted content value in the source without inventing evidence."""
    if not isinstance(value, str) or not value:
        return None
    for left, right in (("“", "”"), ('"', '"'), ("‘", "’"), ("'", "'")):
        candidate = f"{left}{value}{right}"
        if candidate in prompt:
            return candidate
    return value if value in prompt else None


def _quote_equivalent_span(prompt: str, value: str) -> str | None:
    """Return the original prompt span when only quote glyphs/case differ."""
    translation = str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"',
    })
    normalized_prompt = prompt.translate(translation).casefold()
    normalized_value = value.translate(translation).casefold()
    start = normalized_prompt.find(normalized_value)
    if start < 0:
        return None
    # Every normalized glyph above has width one, so indexes remain aligned
    # with the source text and the returned evidence is still verbatim.
    return prompt[start:start + len(value)]


def _normalize_speech_act_colon_relations(
    prompt: str, graph: AudioSceneGraph
) -> None:
    """Ground `X follows a speech-act: Q` as Q-before-X.

    A quotation after the colon is appositional content of the referenced
    speech act even though it is mentioned after X in the text. This is a
    general grammar normalization, independent of any evaluation sample.
    """
    event_map = {event.id: event for event in graph.events}
    content_owner = {
        str(attribute.value).strip(): attribute.owner_id
        for attribute in graph.attributes
        if attribute.type == "speech_content" and isinstance(attribute.value, str)
    }
    pattern = re.compile(
        r"(?P<prefix>[^.!?]{2,320}?)\s+follows\s+"
        r"(?P<speech_act>[^:]{0,100}\b(?:reply|response|answer|utterance|statement))"
        r"\s*:\s*[“\"](?P<quote>[^”\"]+)[”\"]",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(prompt):
        speech_id = content_owner.get(match.group("quote").strip())
        if not speech_id or event_map.get(speech_id, None) is None:
            continue
        prefix = match.group("prefix")
        source_candidates = [
            event for event in graph.events
            if event.id != speech_id and event.type != "speech" and event.basis in prefix
        ]
        if not source_candidates:
            continue
        later_event = max(source_candidates, key=lambda event: len(event.basis))
        pair = {speech_id, later_event.id}
        graph.relations = [
            relation for relation in graph.relations
            if not (
                relation.type in {"before", "after"}
                and {relation.source_id, relation.target_id} == pair
            )
        ]
        graph.relations.append(Relation(
            id=f"REL{len(graph.relations) + 1}",
            type="before",
            source_id=speech_id,
            target_id=later_event.id,
            basis=match.group(0),
        ))


def _normalize_relation_entailments(graph: AudioSceneGraph) -> None:
    """Materialize temporal entailments of directional acoustic relations."""
    masking = [relation for relation in graph.relations if relation.type == "masks"]
    for relation in masking:
        pair = {relation.source_id, relation.target_id}
        graph.relations = [
            item for item in graph.relations
            if not (
                item.type in {"before", "after"}
                and {item.source_id, item.target_id} == pair
            )
        ]
        if not any(
            item.type == "overlap"
            and {item.source_id, item.target_id} == pair
            for item in graph.relations
        ):
            graph.relations.append(Relation(
                id=f"REL{len(graph.relations) + 1}",
                type="overlap",
                source_id=relation.source_id,
                target_id=relation.target_id,
                basis=relation.basis,
            ))


def _normalize_pause_overlap_relations(prompt: str, graph: AudioSceneGraph) -> None:
    """Ground `As <subject> pauses, <event>` as temporal overlap."""
    for pause in (event for event in graph.events if event.type == "silence"):
        if not re.search(r"\bAs\b[^,.!?]{0,80}\bpauses?\b", pause.basis, re.IGNORECASE):
            continue
        pause_position = prompt.find(pause.basis)
        if pause_position < 0:
            continue
        sentence_end_match = re.search(r"[.!?]", prompt[pause_position:])
        sentence_end = (
            pause_position + sentence_end_match.start()
            if sentence_end_match else len(prompt)
        )
        candidates = [
            event for event in graph.events
            if event.id != pause.id
            and pause_position < prompt.find(event.basis) <= sentence_end
        ]
        if not candidates:
            continue
        concurrent = min(candidates, key=lambda event: prompt.find(event.basis))
        pair = {pause.id, concurrent.id}
        graph.relations = [
            item for item in graph.relations
            if not (
                item.type in {"before", "after"}
                and {item.source_id, item.target_id} == pair
            )
        ]
        if not any(
            item.type == "overlap" and {item.source_id, item.target_id} == pair
            for item in graph.relations
        ):
            basis_end = prompt.find(",", pause_position)
            basis = prompt[pause_position:sentence_end]
            if basis_end >= 0 and basis_end < sentence_end:
                basis = prompt[pause_position:sentence_end]
            graph.relations.append(Relation(
                id=f"REL{len(graph.relations) + 1}",
                type="overlap",
                source_id=pause.id,
                target_id=concurrent.id,
                basis=basis.strip(),
            ))


def canonicalize_scene_graph(prompt: str, graph: AudioSceneGraph) -> AudioSceneGraph:
    """Apply deterministic, prompt-grounded repairs before strict validation."""
    language = _prompt_language(prompt)
    if language == "zh":
        # Models occasionally mix neutral English schema scaffolding into an
        # otherwise Chinese identity label (for example, ``主持人_utterance_2``).
        # These words identify the event's role rather than an audible prompt
        # fact, so localize only the closed set of neutral scaffolding tokens.
        scaffolding = {
            "utterance": "话语",
            "speech": "说话",
            "sound": "声音",
            "occurrence": "段",
        }
        for event in graph.events:
            label = event.label
            # Dynamic properties belong to Attributes, not Event identity.
            # Remove English dynamic phrases that occasionally leak into an
            # otherwise Chinese label; the corresponding Chinese Attribute is
            # retained separately by the graph.
            for phrase in ("fade out", "fade in", "fades out", "fades in"):
                if phrase not in prompt.casefold():
                    label = re.sub(
                        rf"(?<![A-Za-z]){re.escape(phrase)}(?![A-Za-z])",
                        "",
                        label,
                        flags=re.IGNORECASE,
                    )
            for source, target in scaffolding.items():
                if source.casefold() not in prompt.casefold():
                    label = re.sub(
                        rf"(?<![A-Za-z]){source}(?![A-Za-z])",
                        target,
                        label,
                        flags=re.IGNORECASE,
                    )
            event.label = re.sub(r"[_\s]+", " ", label).strip()
    event_map = {event.id: event for event in graph.events}
    obligations = extract_prompt_obligations(prompt)
    required_quote_items = [
        item for item in obligations["quotations"] if item["requires_speech_event"]
    ]
    required_quote_groups: dict[str, list[dict[str, Any]]] = {}
    for item in required_quote_items:
        required_quote_groups.setdefault(
            normalize_quoted_content(item["text"]), []
        ).append(item)
    required_quotes = {
        normalized: items[0] for normalized, items in required_quote_groups.items()
    }

    # Quoted linguistic content is an exact, stable evidence anchor. This also
    # prevents a generated reporting verb from becoming part of the basis.
    for attribute in graph.attributes:
        if attribute.type == "speech_content":
            if quote_basis := _canonical_quote_basis(prompt, attribute.value):
                attribute.basis = quote_basis

    # Normalize layer overlap without adding semantic facts. Spatial
    # properties belong to Attributes. A non-linguistic quoted vocalization
    # uses open-vocabulary vocal_content rather than speech_content. When a
    # texture/emotion Attribute merely repeats the audible action of a sound or
    # vocalization Event, the redundant Attribute contributes no extra fact.
    quoted_speech = set(required_quotes)
    normalized_attributes = []
    for attribute in graph.attributes:
        owner = event_map.get(attribute.owner_id)
        if (
            owner is not None
            and owner.type == "silence"
            and attribute.family in {"content", "source", "acoustic", "dynamic"}
        ):
            continue
        if (
            owner is not None
            and attribute.type == "speech_content"
            and owner.type == "vocalization"
            and normalize_quoted_content(str(attribute.value)) not in quoted_speech
        ):
            attribute.type = "vocal_content"

        if (
            owner is not None
            and attribute.type == "speech_content"
            and normalize_quoted_content(str(attribute.value)) in required_quotes
        ):
            owner.type = "speech"

        overlap = owner is not None and _contains_value(owner.label, attribute.value)
        if overlap:
            if reduced := _remove_literal_value(owner.label, attribute.value):
                owner.label = reduced
                overlap = _contains_value(owner.label, attribute.value)
            elif attribute.type == "speech_content" and owner is not None:
                owner.label = "说话" if language == "zh" else "spoken utterance"
                overlap = False

        redundant_action = (
            overlap
            and owner is not None
            and (
                (owner.type == "vocalization" and attribute.type == "emotion")
                or (owner.type in {"sound", "ambience"} and attribute.type == "texture")
            )
        )
        if redundant_action or (overlap and attribute.type != "speech_content"):
            continue
        normalized_attributes.append(attribute)
    graph.attributes = normalized_attributes

    # Repeated identical quotations are represented once per explicit prompt
    # occurrence. Cap only values covered by the input obligation contract;
    # this removes model-created duplicate owners without discarding unrelated
    # speech content.
    retained_quote_counts: Counter[str] = Counter()
    deduplicated_attributes: list[Attribute] = []
    for attribute in graph.attributes:
        if attribute.type != "speech_content":
            deduplicated_attributes.append(attribute)
            continue
        normalized = normalize_quoted_content(str(attribute.value))
        allowed = len(required_quote_groups.get(normalized, []))
        if allowed and retained_quote_counts[normalized] >= allowed:
            continue
        retained_quote_counts[normalized] += 1
        deduplicated_attributes.append(attribute)
    graph.attributes = deduplicated_attributes

    # Fulfil explicit utterance obligations locally when the generator omitted
    # only the content node. Prefer an existing nearby speech Event; otherwise
    # add a neutral, prompt-language Event anchored by the exact quotation.
    covered_counts = Counter(
        normalize_quoted_content(str(attribute.value))
        for attribute in graph.attributes if attribute.type == "speech_content"
    )
    repair_index = 0
    for normalized, quote_items in required_quote_groups.items():
        missing = len(quote_items) - covered_counts[normalized]
        for obligation in quote_items[-missing:] if missing > 0 else []:
            repair_index += 1
            quote_position = obligation["start"]
            owners_with_content = {
                attribute.owner_id for attribute in graph.attributes
                if attribute.type == "speech_content"
            }
            candidates = [
                event for event in graph.events
                if event.type == "speech" and event.id not in owners_with_content
            ]
            owner = min(
                candidates,
                key=lambda event: abs(prompt.find(event.basis) - quote_position),
                default=None,
            )
            if owner is None or abs(prompt.find(owner.basis) - quote_position) > 160:
                chinese = _prompt_language(prompt) == "zh"
                owner = Event(
                    id=f"EV_REPAIR_QUOTE_{repair_index}",
                    type="speech",
                    label=(f"第{_zh_number(repair_index)}段说话" if chinese
                           else f"spoken utterance {repair_index}"),
                    basis=obligation["raw"],
                )
                graph.events.append(owner)
            graph.attributes.append(Attribute(
                id=f"AT_REPAIR_QUOTE_{repair_index}", owner_id=owner.id,
                family="content", type="speech_content", value=obligation["text"],
                basis=obligation["raw"],
            ))

    # Silence obligations are explicit audible-structure requirements and can
    # be materialized losslessly from their exact prompt spans.
    covered_silence_spans = {
        prompt.find(event.basis) for event in graph.events if event.type == "silence"
    }
    for silence_index, obligation in enumerate(obligations["silences"], 1):
        if any(abs(position - obligation["start"]) <= 24 for position in covered_silence_spans):
            continue
        chinese = _prompt_language(prompt) == "zh"
        graph.events.append(Event(
            id=f"EV_REPAIR_SILENCE_{silence_index}", type="silence",
            label=(f"第{_zh_number(silence_index)}段静默" if chinese
                   else f"silence interval {silence_index}"),
            basis=obligation["text"],
        ))
        covered_silence_spans.add(obligation["start"])

    _normalize_speech_act_colon_relations(prompt, graph)
    _normalize_relation_entailments(graph)
    _normalize_pause_overlap_relations(prompt, graph)

    for item in graph.events + graph.attributes + graph.relations:
        if item.basis in prompt:
            continue
        start = prompt.casefold().find(item.basis.casefold())
        if start >= 0:
            item.basis = prompt[start:start + len(item.basis)]
            continue
        if equivalent := _quote_equivalent_span(prompt, item.basis):
            item.basis = equivalent
            continue
        if item.basis.endswith(('"', "”", "’", "'")):
            for punctuation in (",", ".", "!", "?"):
                candidate = item.basis[:-1] + punctuation + item.basis[-1]
                if candidate in prompt:
                    item.basis = candidate
                    break
            if item.basis in prompt:
                continue
        fragments = [
            fragment.strip(" ,，。.!?！？")
            for fragment in re.split(r"(?:\.{3}|…+|[。.!?！？])", item.basis)
        ]
        grounded = [fragment for fragment in fragments if len(fragment) >= 2 and fragment in prompt]
        if grounded:
            item.basis = max(grounded, key=len)

    # Optional attributes and relations with no continuous source span cannot
    # yield auditable rubrics. Drop that local record rather than rejecting an
    # otherwise grounded scene graph. Event grounding remains a hard error.
    graph.attributes = [item for item in graph.attributes if item.basis in prompt]
    graph.relations = [
        item for item in graph.relations
        if item.basis in prompt and item.source_id != item.target_id
    ]

    # A broad speech Event can cover several utterances interleaved with sound
    # Events. In that case an otherwise grounded relation audit may emit both
    # speech -> sound and sound -> later-speech edges against the same Event,
    # making the collapsed event graph cyclic. Preserve relation mention order
    # and retain the maximal prefix-compatible acyclic subset. This is a
    # deterministic structural repair: it never invents or reverses an edge.
    before_graph = nx.DiGraph()
    before_graph.add_nodes_from(event.id for event in graph.events)
    acyclic_relations: list[Relation] = []
    for relation in graph.relations:
        if relation.type != "before":
            acyclic_relations.append(relation)
            continue
        before_graph.add_edge(relation.source_id, relation.target_id)
        if nx.is_directed_acyclic_graph(before_graph):
            acyclic_relations.append(relation)
        else:
            before_graph.remove_edge(relation.source_id, relation.target_id)
    graph.relations = acyclic_relations

    # A speech-content span is a more precise position anchor than a broad
    # speaker-introduction basis. Use it to restore prompt order, then rewrite
    # Event references deterministically.
    speech_basis = {
        attribute.owner_id: attribute.basis
        for attribute in graph.attributes if attribute.type == "speech_content"
    }
    for event in graph.events:
        if event.id in speech_basis:
            event.basis = speech_basis[event.id]
    old_order = {event.id: index for index, event in enumerate(graph.events)}
    graph.events.sort(key=lambda event: (
        prompt.find(event.basis), old_order[event.id]
    ))

    # Textually identical labels make downstream presence questions ambiguous.
    # Add neutral occurrence ordinals without introducing prompt facts.
    counts = Counter(event.label.casefold() for event in graph.events)
    seen: Counter[str] = Counter()
    chinese = language == "zh"
    for event in graph.events:
        key = event.label.casefold()
        if counts[key] <= 1:
            continue
        seen[key] += 1
        suffix = (f"（第{_zh_number(seen[key])}段）" if chinese
                  else f" (occurrence {seen[key]})")
        event.label += suffix
    event_id_map: dict[str, str] = {}
    for index, event in enumerate(graph.events, 1):
        event_id_map[event.id] = f"EV{index}"
    for event in graph.events:
        event.id = event_id_map[event.id]
    for attribute in graph.attributes:
        if attribute.owner_id in event_id_map:
            attribute.owner_id = event_id_map[attribute.owner_id]
    for relation in graph.relations:
        relation.source_id = event_id_map.get(relation.source_id, relation.source_id)
        relation.target_id = event_id_map.get(relation.target_id, relation.target_id)

    for index, attribute in enumerate(graph.attributes, 1):
        attribute.id = f"AT{index}"
    for index, relation in enumerate(graph.relations, 1):
        relation.id = f"REL{index}"
    return graph


def _quoted_speech_spans(prompt: str) -> list[str]:
    """Return only quotations classified as audible spoken utterances."""
    obligations = extract_prompt_obligations(prompt)
    return [
        item["text"] for item in obligations["quotations"]
        if item["requires_speech_event"]
    ]


def _explicit_silence_spans(prompt: str) -> list[str]:
    """Find explicit silence obligations through the shared input contract."""
    return [item["text"] for item in extract_prompt_obligations(prompt)["silences"]]


def validate_scene_graph(prompt: str, graph: AudioSceneGraph) -> list[str]:
    errors: list[str] = []
    nodes = [item.id for item in graph.events]
    ids = nodes + [item.id for item in graph.attributes + graph.relations]
    if dup := _duplicates(ids): errors.append("duplicate_ids:" + ",".join(dup))
    event_set = set(nodes)
    if dup := _duplicates([item.label.casefold() for item in graph.events]):
        errors.append("duplicate_event_labels:" + ",".join(dup))
    explicit_silences = _explicit_silence_spans(prompt)
    silence_events = [item for item in graph.events if item.type == "silence"]
    if len(silence_events) < len(explicit_silences):
        errors.append(
            f"explicit_silence_coverage:expected={len(explicit_silences)}:"
            f"actual={len(silence_events)}"
        )
    node_set = event_set | {"SCENE"}
    for attribute in graph.attributes:
        if attribute.owner_id not in node_set:
            errors.append(f"unknown_attribute_owner:{attribute.id}:{attribute.owner_id}")
    for relation in graph.relations:
        if relation.source_id not in event_set or relation.target_id not in event_set:
            errors.append(f"unknown_relation_endpoint:{relation.id}")
    for item in graph.events + graph.attributes + graph.relations:
        if item.basis not in prompt:
            errors.append(f"basis_not_verbatim:{item.id}")
    language = _prompt_language(prompt)
    for event in graph.events:
        if (_natural_language_mismatch(event.label, language)
                and not _verbatim_foreign_content(event.label, prompt, language)):
            errors.append(f"event_label_language_mismatch:{event.id}:{language}")
    for attribute in graph.attributes:
        if (_natural_language_mismatch(attribute.value, language)
                and not _verbatim_foreign_content(attribute.value, prompt, language)):
            errors.append(f"attribute_value_language_mismatch:{attribute.id}:{language}")
    event_map = {item.id: item for item in graph.events}
    for attribute in graph.attributes:
        owner = event_map.get(attribute.owner_id)
        if owner and _contains_value(owner.label, attribute.value):
            errors.append(f"event_label_attribute_overlap:{owner.id}:{attribute.id}")
        if attribute.type == "speech_content" and (owner is None or owner.type != "speech"):
            errors.append(f"speech_content_owner_not_speech:{attribute.id}:{attribute.owner_id}")

    quoted_spans = Counter(
        normalize_quoted_content(value) for value in _quoted_speech_spans(prompt)
    )
    speech_values = Counter(
        normalize_quoted_content(str(item.value)) for item in graph.attributes
        if item.type == "speech_content" and isinstance(item.value, str)
    )
    for index, (quote, required_count) in enumerate(quoted_spans.items(), 1):
        actual_count = speech_values[quote]
        if actual_count < required_count:
            # Keep the public error code stable for the upstream validators.
            errors.append(f"quoted_speech_missing:{index}")
        elif actual_count > required_count:
            errors.append(f"quoted_speech_duplicate:{index}")

    for relation in graph.relations:
        if relation.source_id == relation.target_id:
            errors.append(f"self_relation:{relation.id}")

    temporal_graph = nx.DiGraph()
    temporal_graph.add_nodes_from(event_set)
    temporal_graph.add_edges_from(
        (relation.source_id, relation.target_id)
        for relation in graph.relations if relation.type == "before"
    )
    if not nx.is_directed_acyclic_graph(temporal_graph):
        errors.append("before_relation_cycle")

    last_position = -1
    for event in graph.events:
        position = prompt.find(event.basis)
        if position >= 0 and position < last_position:
            errors.append(f"event_prompt_order_mismatch:{event.id}")
        last_position = max(last_position, position)

    return errors


def _dependency_requirements(
    item: AtomicTuple, presence_by_node: dict[str, str]
) -> tuple[list[str], list[str]]:
    if item.predicate_type == "presence":
        return [], []
    if item.predicate_type == "attribute":
        if len(item.subject_ids) != 1:
            return [], list(item.subject_ids)
        owner = item.subject_ids[0]
        if owner == "SCENE":
            return [], []
        parent = presence_by_node.get(owner)
        return ([parent], []) if parent else ([], [owner])
    if len(item.subject_ids) != 2:
        return [], list(item.subject_ids)
    parents = [presence_by_node[node_id] for node_id in item.subject_ids if node_id in presence_by_node]
    missing = [node_id for node_id in item.subject_ids if node_id not in presence_by_node]
    return list(dict.fromkeys(parents)), missing


def build_dependencies(tuples: list[AtomicTuple]) -> list[DependencyRecord]:
    """Build the exact two-level prerequisite DAG from tuple ownership/endpoints."""
    presence_by_node: dict[str, str] = {}
    for item in tuples:
        if item.predicate_type == "presence":
            for node_id in item.subject_ids:
                presence_by_node.setdefault(node_id, item.tuple_id)

    reasons = {
        "presence": "event presence root",
        "attribute_event": "event attribute requires owner event presence",
        "attribute_scene": "scene attribute root",
        "relation": "relation requires both endpoint event presences",
    }
    output: list[DependencyRecord] = []
    for item in tuples:
        parents, missing = _dependency_requirements(item, presence_by_node)
        if missing:
            raise ValueError(f"Missing presence tuple for {item.tuple_id}: {','.join(missing)}")
        if item.predicate_type == "presence":
            reason = reasons["presence"]
        elif item.predicate_type == "attribute":
            reason = reasons["attribute_scene" if item.subject_ids == ["SCENE"] else "attribute_event"]
        else:
            reason = reasons["relation"]
        output.append(DependencyRecord(tuple_id=item.tuple_id, parent_tuple_ids=parents, reason=reason))
    return output


def validate_dependencies(tuples: list[AtomicTuple], dependencies: list[DependencyRecord]) -> list[str]:
    errors: list[str] = []
    tuple_ids = [item.tuple_id for item in tuples]
    dep_ids = [item.tuple_id for item in dependencies]
    if dup := _duplicates(tuple_ids): errors.append("duplicate_tuple_ids:" + ",".join(dup))
    if dup := _duplicates(dep_ids): errors.append("duplicate_dependency_ids:" + ",".join(dup))
    if set(dep_ids) != set(tuple_ids) or len(dep_ids) != len(tuple_ids):
        errors.append("dependency_coverage_mismatch")
    presence_by_node: dict[str, str] = {}
    for item in tuples:
        if item.predicate_type == "presence":
            for node_id in item.subject_ids:
                presence_by_node.setdefault(node_id, item.tuple_id)
    tuple_map = {item.tuple_id: item for item in tuples}
    graph = nx.DiGraph()
    graph.add_nodes_from(tuple_ids)
    for item in dependencies:
        if dup := _duplicates(item.parent_tuple_ids):
            errors.append(f"duplicate_parents:{item.tuple_id}:" + ",".join(dup))
        for parent in item.parent_tuple_ids:
            if parent not in tuple_ids:
                errors.append(f"unknown_parent:{item.tuple_id}:{parent}")
            else:
                graph.add_edge(parent, item.tuple_id)
        source = tuple_map.get(item.tuple_id)
        if source:
            expected, missing = _dependency_requirements(source, presence_by_node)
            if missing:
                errors.append(f"missing_presence:{item.tuple_id}:" + ",".join(missing))
            if item.parent_tuple_ids != expected:
                errors.append(
                    f"dependency_parent_mismatch:{item.tuple_id}:"
                    f"expected={','.join(expected)}:actual={','.join(item.parent_tuple_ids)}"
                )
    if not nx.is_directed_acyclic_graph(graph): errors.append("dependency_cycle")
    return errors


def canonicalize_questions(tuples: list[AtomicTuple], dependencies: list[DependencyRecord],
                           model_questions: list[GeneratedQuestion],
                           prompt: str | None = None,
                           language_override: str | None = None) -> list[RubricQuestion]:
    """Preserve model wording while making graph metadata deterministic."""
    document_language = (
        "zh" if language_override.casefold().startswith("zh") else "en"
    ) if language_override else (_prompt_language(prompt) if prompt is not None else None)

    def use_chinese(value: str) -> bool:
        if document_language is not None:
            return document_language == "zh"
        return any("\u4e00" <= ch <= "\u9fff" for ch in value)

    by_tuple = {item.tuple_id: item for item in model_questions}
    rubric_by_tuple = {item.tuple_id: f"R{index:03d}" for index, item in enumerate(tuples, 1)}
    dep_map = {item.tuple_id: item for item in dependencies}
    output: list[RubricQuestion] = []
    event_labels = {
        item.subject_ids[0]: str(item.object_value)
        for item in tuples
        if item.predicate_type == "presence" and len(item.subject_ids) == 1
    }
    for item in tuples:
        proposed = by_tuple.get(item.tuple_id)
        if proposed is None:
            raise ValueError(f"Question generation omitted {item.tuple_id}")
        wording = proposed.question.strip()
        terminal = ""
        if wording.endswith(("?", "？")):
            terminal, wording = wording[-1], wording[:-1]
        if wording.count("“") == wording.count("”") + 1:
            wording += "”"
        if wording.count('"') % 2:
            wording += '"'
        wording += terminal or ("？" if use_chinese(wording) else "?")
        if item.predicate_type == "presence":
            label = event_labels.get(item.subject_ids[0], str(item.object_value))
            if use_chinese(label):
                wording = f"是否能听到{label}？"
            else:
                # Presence questions must not inherit acoustic attributes from
                # free-form model wording (for example "audible", "distant",
                # or "loud").  Use one neutral deterministic English template,
                # just as the Chinese branch above already does.
                wording = f"Can {label} be heard?"
        elif item.predicate_type == "attribute" and item.predicate == "clarity":
            label = event_labels.get(item.subject_ids[0], "声音")
            value = str(item.object_value)
            if use_chinese(label + value):
                wording = f"{label}的声音是否{value}？"
        elif item.predicate_type == "attribute" and item.predicate == "distance":
            label = event_labels.get(item.subject_ids[0], "声音")
            value = str(item.object_value)
            if use_chinese(label + value):
                wording = f"{label}是否来自{value}？"
        elif item.predicate_type == "attribute" and item.predicate == "start_time":
            label = event_labels.get(
                item.subject_ids[0],
                "完整音频场景" if use_chinese(item.basis) else "complete audio scene",
            )
            value = str(item.object_value)
            unit = item.object_unit or ""
            if use_chinese(label + value + unit):
                rendered = f"第{_zh_number(value)}{unit}" if unit else value
                wording = f"{label}的开始时间是否为{rendered}？"
            else:
                rendered = f"{value} {unit}".strip()
                wording = f"Does {label} start at {rendered}?"
        elif item.predicate_type == "attribute" and item.predicate == "occurrence_count":
            label = event_labels.get(item.subject_ids[0], "声音")
            value = str(item.object_value)
            unit = str(item.object_unit or "")
            if use_chinese(label + value + unit + item.basis):
                wording = f"{label}是否发生{_zh_number(value)}{unit or '次'}？"
            else:
                wording = f"Does {label} occur {value} {unit or 'times'}?"
        elif (item.predicate_type == "attribute"
              and item.predicate in {"loudness", "loudness_change"}):
            label = event_labels.get(item.subject_ids[0], "sound")
            value = str(item.object_value)
            if use_chinese(label + value):
                if value in {"提高", "降低", "升高", "减弱"}:
                    wording = f"{label}的音量是否{value}？"
            elif value.casefold().endswith("ly"):
                subject = label
                if not re.match(r"^(?:a|an|the)\b", subject, re.IGNORECASE):
                    subject = "the " + subject
                wording = f"Is {subject} {value} audible?"
        elif item.predicate_type == "attribute" and item.predicate == "texture":
            label = event_labels.get(item.subject_ids[0], "声音")
            value = str(item.object_value)
            if use_chinese(label + value):
                wording = f"{label}是否发出“{value}”声？"
        elif item.predicate_type == "attribute" and item.predicate == "repetition_pattern":
            label = event_labels.get(item.subject_ids[0], "声音")
            value = str(item.object_value)
            if value == "偶尔":
                wording = f"{label}是否偶尔响起？"
        elif item.predicate_type == "relation":
            source_label = event_labels.get(item.subject_ids[0], "source event")
            target_label = event_labels.get(item.subject_ids[1], "target event")
            chinese = use_chinese(source_label + target_label)
            if chinese:
                templates = {
                    "before": f"{source_label}是否发生在{target_label}之前？",
                    "after": f"{source_label}是否发生在{target_label}之后？",
                    "overlap": f"{source_label}是否与{target_label}重叠？",
                    "same_source": f"{source_label}与{target_label}是否来自同一声源？",
                    "different_source": f"{source_label}与{target_label}是否来自不同声源？",
                    "same_speaker": f"{source_label}与{target_label}是否由同一说话人发出？",
                    "different_speaker": f"{source_label}与{target_label}是否由不同说话人发出？",
                    "louder_than": f"{source_label}是否比{target_label}更响？",
                    "masks": f"{source_label}是否掩盖{target_label}？",
                    "background_of": f"{source_label}是否是{target_label}的背景？",
                    "foreground_of": f"{source_label}是否是{target_label}的前景？",
                    "near": f"{source_label}是否比{target_label}更近？",
                    "far": f"{source_label}是否比{target_label}更远？",
                    "left_of": f"{source_label}是否位于{target_label}左侧？",
                    "right_of": f"{source_label}是否位于{target_label}右侧？",
                }
                wording = templates[item.predicate]
            else:
                templates = {
                    "before": f"Does {source_label} occur before {target_label}?",
                    "after": f"Does {source_label} occur after {target_label}?",
                    "overlap": f"Does {source_label} overlap {target_label}?",
                    "same_source": f"Do {source_label} and {target_label} have the same source?",
                    "different_source": f"Do {source_label} and {target_label} have different sources?",
                    "same_speaker": f"Are {source_label} and {target_label} produced by the same speaker?",
                    "different_speaker": f"Are {source_label} and {target_label} produced by different speakers?",
                    "louder_than": f"Is {source_label} louder than {target_label}?",
                    "masks": f"Does {source_label} mask {target_label}?",
                    "background_of": f"Is {source_label} the background of {target_label}?",
                    "foreground_of": f"Is {source_label} the foreground of {target_label}?",
                    "near": f"Is {source_label} nearer than {target_label}?",
                    "far": f"Is {source_label} farther away than {target_label}?",
                    "left_of": f"Is {source_label} to the left of {target_label}?",
                    "right_of": f"Is {source_label} to the right of {target_label}?",
                }
                wording = templates[item.predicate]
        if item.predicate_type == "attribute" and len(item.subject_ids) == 1:
            # Model wording must never leak internal schema identifiers, and a
            # quantitative proposition must retain its unit. Repair either
            # condition with one conservative localized proposition template.
            exposes_type = "_" in item.predicate and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(item.predicate)}(?![A-Za-z0-9_])",
                wording,
                flags=re.IGNORECASE,
            )
            unit = str(item.object_unit or "")
            missing_unit = bool(unit and unit.casefold() not in wording.casefold())
            if exposes_type or missing_unit:
                default_label = (
                    "完整音频场景" if use_chinese(item.basis)
                    else "complete audio scene"
                )
                label = event_labels.get(item.subject_ids[0], default_label)
                value = str(item.object_value)
                rendered = f"{value} {unit}".strip()
                chinese = use_chinese(label + value + unit + item.basis)
                wording = (
                    f"{label}是否具有“{rendered}”这一可听属性？"
                    if chinese else f'Does {label} have the audible property "{rendered}"?'
                )
        output.append(RubricQuestion(
            rubric_id=rubric_by_tuple[item.tuple_id],
            tuple_id=item.tuple_id,
            axis=item.axis,
            scope=item.scope,
            domain=item.domain,
            subtype=item.predicate,
            question=wording,
            basis=item.basis,
            target_ids=item.subject_ids,
            parent_ids=[rubric_by_tuple[parent] for parent in dep_map[item.tuple_id].parent_tuple_ids],
            expected_answer="yes",
        ))
    return output


def validate_questions(prompt: str, tuples: list[AtomicTuple], dependencies: list[DependencyRecord],
                       questions: list[RubricQuestion],
                       language_override: str | None = None) -> list[str]:
    errors: list[str] = []
    if len(questions) != len(tuples): errors.append("question_count_mismatch")
    tuple_map = {item.tuple_id: item for item in tuples}
    dep_map = {item.tuple_id: item for item in dependencies}
    question_by_tuple = {item.tuple_id: item for item in questions}
    event_labels = {
        item.subject_ids[0]: str(item.object_value)
        for item in tuples
        if item.predicate_type == "presence" and len(item.subject_ids) == 1
    }
    if set(question_by_tuple) != set(tuple_map): errors.append("question_tuple_coverage_mismatch")
    expected_rubric = {tuple_id: f"R{index:03d}" for index, tuple_id in enumerate(tuple_map, 1)}
    language = (
        "zh" if language_override.casefold().startswith("zh") else "en"
    ) if language_override else _prompt_language(prompt)
    attributes_by_owner: dict[str, list[AtomicTuple]] = {}
    for item in tuples:
        if item.predicate_type == "attribute" and len(item.subject_ids) == 1:
            attributes_by_owner.setdefault(item.subject_ids[0], []).append(item)
    for tuple_id, question in question_by_tuple.items():
        source = tuple_map[tuple_id]
        if question.rubric_id != expected_rubric[tuple_id]: errors.append(f"rubric_id_mismatch:{tuple_id}")
        if question.axis != source.axis or question.scope != source.scope or question.domain != source.domain:
            errors.append(f"tuple_metadata_mismatch:{tuple_id}")
        if question.basis != source.basis or question.basis not in prompt:
            errors.append(f"question_basis_mismatch:{tuple_id}")
        if question.target_ids != source.subject_ids: errors.append(f"target_ids_mismatch:{tuple_id}")
        expected_parents = [expected_rubric[parent] for parent in dep_map.get(tuple_id, DependencyRecord(tuple_id=tuple_id, reason="")).parent_tuple_ids]
        if question.parent_ids != expected_parents: errors.append(f"question_parents_mismatch:{tuple_id}")
        if not question.question.strip().endswith(("?", "？")): errors.append(f"not_a_question:{tuple_id}")
        if (_natural_language_mismatch(question.question, language)
                and not _verbatim_foreign_content(question.question, prompt, language)):
            errors.append(f"question_language_mismatch:{tuple_id}:{language}")
        if language == "zh":
            foreign_words = [
                word for word in re.findall(r"[A-Za-z]{3,}", question.question)
                if word.casefold() not in prompt.casefold() and not word.isupper()
            ]
            if foreign_words:
                errors.append(
                    f"question_mixed_language:{tuple_id}:" + ",".join(foreign_words)
                )
        if question.question.count("“") != question.question.count("”"):
            errors.append(f"unbalanced_curly_quotes:{tuple_id}")
        if question.question.count('"') % 2:
            errors.append(f"unbalanced_double_quotes:{tuple_id}")
        graph_like_tokens = re.findall(
            r"\b(?:EV|AT|REL)\d+\b", question.question, flags=re.IGNORECASE
        )
        if any(token.casefold() not in prompt.casefold() for token in graph_like_tokens):
            errors.append(f"question_exposes_graph_id:{tuple_id}")
        if source.predicate_type == "presence" and len(source.subject_ids) == 1:
            if not _contains_value(question.question, source.object_value):
                errors.append(f"presence_question_missing_event_label:{tuple_id}")
            for attribute in attributes_by_owner.get(source.subject_ids[0], []):
                if _contains_value(question.question, attribute.object_value):
                    errors.append(
                        f"presence_question_attribute_overlap:{tuple_id}:{attribute.tuple_id}"
                    )
        elif source.predicate_type == "attribute" and len(source.subject_ids) == 1:
            if "_" in source.predicate and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(source.predicate)}(?![A-Za-z0-9_])",
                question.question,
                flags=re.IGNORECASE,
            ):
                errors.append(f"question_exposes_attribute_type:{tuple_id}")
            if isinstance(source.object_value, str) and len(source.object_value.strip()) >= 2:
                if not _contains_value(question.question, source.object_value):
                    errors.append(f"attribute_question_missing_value:{tuple_id}")
            if (
                source.object_unit
                and str(source.object_unit).casefold() not in question.question.casefold()
            ):
                errors.append(f"attribute_question_missing_unit:{tuple_id}")
            for sibling in attributes_by_owner.get(source.subject_ids[0], []):
                sibling_is_part_of_target = _contains_value(
                    str(source.object_value), sibling.object_value
                )
                if (
                    sibling.tuple_id != tuple_id
                    and not sibling_is_part_of_target
                    and _contains_value(question.question, sibling.object_value)
                ):
                    errors.append(
                        f"attribute_question_sibling_overlap:{tuple_id}:{sibling.tuple_id}"
                    )
        elif source.predicate_type == "relation":
            endpoint_labels = [
                event_labels.get(endpoint, "") for endpoint in source.subject_ids
            ]
            for endpoint in source.subject_ids:
                for attribute in attributes_by_owner.get(endpoint, []):
                    identity_fact = any(
                        _contains_value(label, attribute.object_value)
                        for label in endpoint_labels
                    )
                    identity_fact = identity_fact or _contains_value(
                        " ".join(endpoint_labels), attribute.object_value
                    )
                    role_value = str(attribute.object_value).strip().casefold()
                    relation_requires_role_word = (
                        source.predicate == "background_of"
                        and role_value in {"background", "背景", "背景声", "背景音"}
                        or source.predicate == "foreground_of"
                        and role_value in {"foreground", "前景", "前景声", "前景音"}
                        or source.predicate in {"before", "after"}
                        and attribute.predicate in {"temporal_position", "position"}
                        and role_value == source.predicate
                    )
                    if (
                        not identity_fact
                        and not relation_requires_role_word
                        and _contains_value(question.question, attribute.object_value)
                    ):
                        errors.append(
                            f"relation_question_attribute_overlap:{tuple_id}:{attribute.tuple_id}"
                        )
    return errors
