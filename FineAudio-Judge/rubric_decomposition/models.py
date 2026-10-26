from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EventType = Literal["speech", "vocalization", "sound", "music", "ambience", "silence"]
Axis = Literal["content_fidelity", "temporal_structural_fidelity", "acoustic_presentation_fidelity"]
Scope = Literal["local", "relational", "global"]


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: EventType
    label: str
    basis: str


class Attribute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    owner_id: str
    family: Literal[
        "content", "source", "quantity", "temporal", "acoustic", "dynamic",
        "spatial", "domain_specific", "scene",
    ]
    type: str
    value: Any
    unit: str | None = None
    basis: str


class Relation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal[
        "before", "after", "overlap", "same_source", "different_source",
        "same_speaker", "different_speaker", "louder_than", "masks",
        "foreground_of", "background_of", "near", "far", "left_of", "right_of"
    ]
    source_id: str
    target_id: str
    basis: str


class AudioSceneGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[Event]
    attributes: list[Attribute] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class SceneGraphResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_graph: AudioSceneGraph


class RelationReviewResponse(BaseModel):
    """Relation-only audit output; Events and Attributes stay immutable."""

    model_config = ConfigDict(extra="forbid")
    relations: list[Relation] = Field(default_factory=list)


class RelationDecision(BaseModel):
    # Explanatory evidence may be emitted despite the minimal response schema;
    # only the auditable candidate ID and closed-set verdict are trusted.
    model_config = ConfigDict(extra="ignore")
    candidate_id: str
    verdict: Literal[
        "source_before_target", "target_before_source", "overlap",
        "keep_non_temporal", "reverse_non_temporal", "unsupported",
    ]


class RelationVerificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[RelationDecision]


class AtomicTuple(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tuple_id: str
    predicate_type: Literal["presence", "attribute", "relation"]
    predicate: str
    subject_ids: list[str]
    object_value: Any | None = None
    object_unit: str | None = None
    basis: str
    axis: Axis
    scope: Scope
    domain: list[str]


class DependencyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tuple_id: str
    parent_tuple_ids: list[str] = Field(default_factory=list)
    reason: str


class RubricQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rubric_id: str
    tuple_id: str
    axis: Axis
    scope: Scope
    domain: list[str]
    subtype: str
    question: str
    basis: str
    target_ids: list[str]
    parent_ids: list[str]
    expected_answer: Literal["yes"] = "yes"


class GeneratedQuestion(BaseModel):
    """The only fields the question-writing LLM is allowed to generate."""

    # Some JSON-only model backends occasionally add explanatory keys despite
    # an explicit minimal schema. They are ignored; only wording is trusted.
    model_config = ConfigDict(extra="ignore")
    tuple_id: str
    question: str


class QuestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[GeneratedQuestion]
