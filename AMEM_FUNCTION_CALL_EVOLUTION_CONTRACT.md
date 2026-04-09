# A-MEM Function-Call Evolution Contract

## Goal

This document defines the tool contract for collapsing A-MEM's two current evolution steps:

- `LinkStrengtheningEvolution`
- `NeighborContextUpdateEvolution`

into one `LLMFunctionCallEvolution`-based controller.

The target is repo-consistent A-MEM behavior, not the broadest possible paper wording. In particular:

- the LLM should act only over the current note plus retrieved neighbor candidates
- neighbor updates are limited to `context` and `tags`
- link strengthening must preserve existing links, dedupe them, and truncate to `max_links_per_record`
- neighbor updates do not require immediate embedding refresh

## Intended Evolution Shape

One evolution module should run after the new A-MEM note has already been appended to the graph-note layer.

That single evolution controller should:

1. identify the current just-written note record
2. retrieve candidate neighbor records
3. expose only those records as the tool-visible candidate set
4. let the LLM issue one or more tool calls
5. apply graph-link strengthening and neighbor note updates through tool executors

This makes the LLM interaction look like the upstream repo's "JSON / tool-like action selection", while keeping write semantics inside executor code instead of prompt-only discipline.

## Visible Record Boundary

The controller must enforce a hard visible-record boundary.

- `selected_records` should contain the current just-written note record
- prompt-recall retrieval may add candidate neighbors
- the final `visible_records` set should be:
  - current record
  - retrieved neighbor candidates
- tools must reject any `record_id` not present in `visible_records`

This boundary is part of the contract, not a prompt hint.

## Required Tools

The collapsed A-MEM evolution should use two custom write tools.

### 1. `AMEM_STRENGTHEN_LINKS`

Purpose:
strengthen outgoing graph links on the current note and optionally patch the current note's tags.

This is the function-call replacement for the current `LinkStrengtheningEvolution`.

#### Allowed target

- only the current selected A-MEM record
- must not modify any other record

#### Parameters

```json
{
  "type": "object",
  "properties": {
    "record_id": { "type": "string" },
    "neighbor_record_ids": {
      "type": "array",
      "items": { "type": "string" }
    },
    "tags": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["record_id", "neighbor_record_ids"],
  "additionalProperties": false
}
```

#### Executor preconditions

- `record_id` must be the current selected record
- every id in `neighbor_record_ids` must be in `visible_records`
- `neighbor_record_ids` must not include `record_id` itself
- target layer must be a graph layer

#### Executor behavior

The executor must:

1. load the target record
2. read existing `graph.links`
3. merge existing links with `neighbor_record_ids`
4. dedupe while preserving stable order
5. truncate to `max_links_per_record`
6. update:
   - `metadata["graph"]["links"]`
   - `metadata["graph"]["link_count"]`
7. if `tags` is provided:
   - normalize and dedupe tags
   - patch only the A-MEM note payload tags on the current record
8. preserve all unrelated metadata
9. avoid immediate embedding regeneration for this update path

#### Important semantics

- existing links are preserved unless excluded by truncation
- repeated neighbor ids are ignored after dedupe
- truncation is executor-enforced, not prompt-enforced
- `tags` is optional; links are the primary effect

#### Suggested effect payload

```json
{
  "action": "amem_strengthen_links",
  "effect_type": "link_strengthening",
  "record_id": "rec-current",
  "layer": "knowledge_graph",
  "status": "applied",
  "previous_links": ["rec-old-1"],
  "requested_neighbor_record_ids": ["rec-a", "rec-b", "rec-a"],
  "current_links": ["rec-old-1", "rec-a", "rec-b"],
  "strengthened_links": ["rec-a", "rec-b"],
  "truncated": false,
  "updated_tags": ["habit", "focus"]
}
```

### 2. `AMEM_UPDATE_NEIGHBOR`

Purpose:
update one neighbor note's `context` and/or `tags` from the perspective of the current note.

This is the function-call replacement for the current `NeighborContextUpdateEvolution`.

#### Allowed target

- any non-current record inside `visible_records`
- intended for retrieved or linked neighbor notes only

#### Parameters

```json
{
  "type": "object",
  "properties": {
    "record_id": { "type": "string" },
    "context": { "type": "string" },
    "tags": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["record_id"],
  "additionalProperties": false
}
```

#### Executor preconditions

- `record_id` must be in `visible_records`
- `record_id` must not be the current selected record
- target record must be an A-MEM note-bearing record

#### Executor behavior

The executor must:

1. load the neighbor record
2. read the current A-MEM note payload from metadata
3. patch only:
   - `context` when provided and non-empty
   - `tags` when provided
4. preserve:
   - `content`
   - `note_text`
   - `keywords`
   - `category`
   - `attributes`
   - graph metadata
   - unrelated metadata
5. normalize and dedupe `tags`
6. avoid immediate embedding regeneration for this update path

#### Important semantics

- this tool must not edit `content`
- this tool must not edit `keywords`
- this tool must not edit graph links
- no-op calls are allowed but should preferably be avoided by prompt design

#### Suggested effect payload

```json
{
  "action": "amem_update_neighbor",
  "effect_type": "neighbor_context_update",
  "record_id": "rec-neighbor",
  "layer": "knowledge_graph",
  "status": "applied",
  "updated_fields": ["context", "tags"]
}
```

## Why Two Tools Instead of One

One tool per mutation family keeps the contract clearer.

- link strengthening has graph-specific invariants and current-record-only targeting
- neighbor update has note-payload-only invariants and neighbor-only targeting

Trying to combine both into a single polymorphic tool would make validation, prompting, and testing weaker.

## Prompt-Side Expectations

The prompt should explain that the LLM is acting as an A-MEM evolution controller over a bounded candidate set.

The prompt should include:

- the current note record id
- the current note content, context, and tags
- the visible neighbor candidate list with record ids
- the available tools and their exact argument names
- the rule that only tool calls change memory
- the rule that if no update is needed, it should make no tool call

The prompt should prefer this action order:

1. call `AMEM_STRENGTHEN_LINKS` zero or one time for the current record
2. call `AMEM_UPDATE_NEIGHBOR` zero or more times for neighbors that need reinterpretation

That action order is a preference, not a hard runtime requirement.

## Module-Level Contract

The collapsed evolution module should satisfy the following high-level contract.

### Inputs

- one current unit and record pair to evolve
- graph-note target layer
- retrieval pipeline or template-driven recall path that yields candidate neighbors
- `max_links_per_record`
- `note_namespace`

### Requires

- graph layer topology
- note payload contract on visible records
- hard visible-record restriction at tool execution time

### Produces

- graph link updates on the current record
- note `context` and `tags` updates on neighbor records
- trace entries for tool calls and resulting effects

### Non-goals

- no requirement to update all four paper-wording note fields
- no immediate embedding refresh for neighbor updates
- no unrestricted whole-layer mutation

## Mapping From Old Modules

### Old `LinkStrengtheningEvolution`

Becomes:

- candidate retrieval for visible set
- one optional `AMEM_STRENGTHEN_LINKS` call

### Old `NeighborContextUpdateEvolution`

Becomes:

- zero or more `AMEM_UPDATE_NEIGHBOR` calls over visible neighbors

## Recommended Trace Fields

The merged evolution trace should make the new controller easy to inspect.

Recommended fields:

- `module`
- `tool_names`
- `selected_record_ids`
- `visible_record_ids`
- `visible_record_source`
- `effects`
- `tool_calls`
- `prompt_trace`
- `max_links_per_record`
- `note_namespace`

## Validation Checklist

Before treating the merged version as a valid replacement, it should pass at least these cases:

1. tool calls cannot mutate records outside `visible_records`
2. `AMEM_STRENGTHEN_LINKS` preserves existing links
3. duplicate requested links are deduped
4. links are truncated to `max_links_per_record`
5. self-link requests are rejected
6. `AMEM_UPDATE_NEIGHBOR` can update `context` only
7. `AMEM_UPDATE_NEIGHBOR` can update `tags` only
8. neighbor updates do not alter `content` or `keywords`
9. neighbor updates preserve graph metadata
10. no immediate embedding refresh is required for neighbor updates

## Bottom-Line Recommendation

The preferred replacement shape is:

- one `LLMFunctionCallEvolution`
- one bounded visible-record set
- two A-MEM-specific custom tools:
  - `AMEM_STRENGTHEN_LINKS`
  - `AMEM_UPDATE_NEIGHBOR`

This is the cleanest way to preserve current A-MEM behavior while aligning the implementation style with the upstream repo's function-like memory evolution flow.
