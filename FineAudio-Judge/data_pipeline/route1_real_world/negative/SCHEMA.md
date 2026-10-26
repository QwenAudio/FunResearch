# Input schemas

## Positive manifest

Required fields:

- `id` or `sample_id`: sample identifier;
- `audio_path`: readable local audio path;
- `audio_caption` (preferred) or `caption`: accepted positive caption.

Optional fields used by this pipeline:

- `dataset`: dataset namespace for the identifier;
- `audio_duration` or `audio_duration_seconds`: duration metadata.

Other positive-pipeline provenance fields are preserved in the source manifest
but are not required by negative construction.

## Rubric manifest

Required row fields:

- `id` or `sample_id`: identifier matching the positive row;
- `questions` or `rubrics`: non-empty list of rubric objects.

Recommended row fields:

- `dataset`: required when IDs are not globally unique;
- `prompt`: the decomposed caption, checked for exact equality with the positive
  `audio_caption` when present.

Each rubric object requires:

- `rubric_id`;
- `scope`;
- `subtype`;
- `question`;
- `basis`.

The public decomposition output also supplies `axis`, `domain`, `basis_start`,
`basis_end`, `target_ids`, and `parent_ids`. These fields are strongly
recommended because controlled target selection and dependency propagation use
them. Label or answer fields are rejected throughout negative construction.

## Join and safety rules

- `(dataset, id)` is the preferred join key.
- An ID-only rubric row is accepted only when that ID identifies one positive
  row unambiguously.
- Every rubric row must match a positive row; positive rows without rubrics are
  skipped.
- A provided rubric `prompt` must exactly equal the positive caption.
- Audio exists locally and is hashed during `prepare`; the final audit checks
  that its bytes are unchanged.

## Generated relation endpoint contract

Every relation patch reports two exact `relation_endpoints` found in the edited
caption. For a relation-only negative, both strings must also occur unchanged in
the positive caption. For an `all_three` negative, an endpoint that is directly
replaced by the event edit reports its replacement anchor instead; every other
endpoint must still occur unchanged in both captions. This exception is limited
by the target IDs shared between the direct event and relation rubrics.
