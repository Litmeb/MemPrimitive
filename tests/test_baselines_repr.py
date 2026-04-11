from __future__ import annotations

import json
from typing import Any
import pytest

from memprimitive.contracts import UNIT_EMBEDDING_CONTRACT
from memprimitive.core import (
    MemoryStore,
    Observation,
    Packet,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.utils import _runtime

from baselines_test_helpers import (
    _FakeAMEMRuntime,
    _seed_layer,
)


def test_unit_formation_returns_one_unit_with_provenance() -> None:
    from memprimitive.baselines import PassThroughUnitFormation

    module = PassThroughUnitFormation()
    packet = Packet(observation=Observation(text="Alice likes tea.", source="dialogue"))

    packet_out, _ = module.run(packet, MemoryStore())

    assert packet_out.units is not None
    assert len(packet_out.units) == 1
    assert packet_out.units[0].text == "Alice likes tea."
    assert packet_out.units[0].metadata["provenance"]["observation_id"] == packet.observation.observation_id


def test_unit_formation_requires_observation() -> None:
    from memprimitive.baselines import PassThroughUnitFormation

    module = PassThroughUnitFormation()

    with pytest.raises(ValueError, match="packet.observation"):
        module.run(Packet(), MemoryStore())


def test_representation_preserves_identity_and_adds_normalized_text() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="  Alice Likes Tea  ", source="dialogue")),
        MemoryStore(),
    )

    packet_out, _ = BasicRepresentation().run(unit_packet, store)

    assert packet_out.units is not None
    assert len(packet_out.units) == 1
    assert packet_out.units[0].unit_id == unit_packet.units[0].unit_id
    assert packet_out.units[0].text == "Alice Likes Tea"
    assert packet_out.units[0].normalized_text == "alice likes tea"
    assert packet_out.units[0].embedding is not None
    assert len(packet_out.units[0].embedding) > 0
    assert packet_out.units[0].representation_elements == ("embedding", "text")
    assert packet_out.trace["representation"]["elements"] == ["text", "embedding"]
    assert packet_out.trace["representation"]["per_unit"][0]["elements"] == ["embedding", "text"]
    assert packet_out.units[0].metadata["representation"]["text"] == "Alice Likes Tea"
    assert packet_out.units[0].metadata["representation"]["normalized_text"] == "alice likes tea"
    assert packet_out.units[0].metadata["representation"]["embedding"]["dim"] == len(packet_out.units[0].embedding)


def test_basic_representation_rejects_legacy_triple_element() -> None:
    from memprimitive.baselines import BasicRepresentation

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "triple"))


def test_configurable_embedding_representation_defaults_to_unit_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import ConfigurableEmbeddingRepresentation, PassThroughUnitFormation

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        MemoryStore(),
    )

    out, _ = ConfigurableEmbeddingRepresentation().run(packet, store)

    unit = out.units[0]
    assert unit.text == "Alice likes tea."
    assert unit.normalized_text is None
    assert unit.description is None
    assert unit.tags == []
    assert unit.metadata["representation"]["embedding_input_text"] == "Alice likes tea."
    assert unit.metadata["representation"]["embedding_input_preview"] == "Alice likes tea."
    assert unit.metadata["representation"]["embedding_version"]
    assert unit.metadata["representation"]["embedding"]["dim"] == len(unit.embedding)
    assert unit.embedding == _runtime._DEFAULT_RUNTIME.embed("Alice likes tea.")
    assert "enhanced_embedding_text" not in unit.metadata["representation"]
    assert out.trace["representation"]["prompt_is_template"] is True
    assert out.trace["representation"]["per_unit"][0]["embedding_input_text"] == "Alice likes tea."


def test_configurable_embedding_representation_supports_literal_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import ConfigurableEmbeddingRepresentation, PassThroughUnitFormation

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        MemoryStore(),
    )

    out, _ = ConfigurableEmbeddingRepresentation(embedding_text="focus only").run(packet, store)

    assert out.units[0].metadata["representation"]["embedding_input_text"] == "focus only"
    assert out.units[0].embedding == _runtime._DEFAULT_RUNTIME.embed("focus only")
    assert out.trace["representation"]["prompt_is_template"] is False


def test_configurable_embedding_representation_supports_template_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import ConfigurableEmbeddingRepresentation, PassThroughUnitFormation

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice likes tea.",
                source="notes",
                metadata={
                    "amem": {
                        "content": "Alice likes tea.",
                        "context": "Tea supports focus.",
                        "keywords": ["alice", "tea"],
                        "tags": ["preference", "habit"],
                    }
                },
            )
        ),
        MemoryStore(),
    )

    out, _ = ConfigurableEmbeddingRepresentation(
        embedding_text=(
            "{{ unit.metadata.amem.content }} | "
            "context: {{ unit.metadata.amem.context }} | "
            "keywords: {{ unit.metadata.amem.keywords | join(', ') }}"
        )
    ).run(packet, store)

    assert out.units[0].metadata["representation"]["embedding_input_text"] == (
        "Alice likes tea. | context: Tea supports focus. | keywords: alice, tea"
    )
    assert out.trace["representation"]["per_unit"][0]["missing_variables"] == []


def test_configurable_embedding_representation_rejects_empty_rendered_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import ConfigurableEmbeddingRepresentation, PassThroughUnitFormation

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        MemoryStore(),
    )

    with pytest.raises(ValueError, match="non-empty embedding text"):
        ConfigurableEmbeddingRepresentation(embedding_text="   ").run(packet, store)


def test_configurable_embedding_representation_contracts_are_embedding_only() -> None:
    from memprimitive.baselines import ConfigurableEmbeddingRepresentation

    module = ConfigurableEmbeddingRepresentation()

    assert module.get_requires_contracts() == frozenset()
    assert module.get_produces_contracts() == frozenset({UNIT_EMBEDDING_CONTRACT})


def test_triple_representation_direct_uses_real_llm(require_real_runtime: None) -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice works at OpenAI in San Francisco and collaborates with Bob on graph memory systems.",
                source="notes",
            )
        ),
        MemoryStore(),
    )

    packet_out, _ = TripleRepresentation(method="direct").run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.triples
    assert unit.metadata["representation"]["triples"] == unit.triples
    assert unit.entities
    assert len(unit.entities) >= 2
    assert "triple" in unit.representation_elements
    assert "entities" in unit.representation_elements
    assert all(len(triple) == 3 for triple in unit.triples)
    flattened = " ".join(" ".join(part for part in triple) for triple in unit.triples).casefold()
    assert "alice" in flattened or "openai" in flattened or "bob" in flattened


def test_triple_representation_two_stage_uses_real_llm(require_real_runtime: None) -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice mentors Bob at OpenAI, and Bob researches retrieval graphs with Carol.",
                source="notes",
            )
        ),
        MemoryStore(),
    )

    packet_out, _ = TripleRepresentation(method="two_stage").run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.triples
    assert unit.entities
    assert unit.metadata["representation"]["triples"] == unit.triples
    entity_set = {entity.casefold() for entity in unit.entities}
    assert any(subject.casefold() in entity_set for subject, _, _ in unit.triples)
    assert any(obj.casefold() in entity_set for _, _, obj in unit.triples)
    assert all(subject and predicate and obj for subject, predicate, obj in unit.triples)


def test_triple_representation_custom_string_prompt_is_passed_to_llm_and_traced() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def require_llm(self, *, capability: str) -> None:
            self.calls.append({"capability": capability})

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            payload = json.loads(input_text)
            self.calls.append(
                {
                    "name": name,
                    "instructions": instructions,
                    "payload": payload,
                    "temperature": temperature,
                    "max_turns": max_turns,
                    "output_type": output_type,
                }
            )
            return {
                "entities": ["Alice", "jasmine tea"],
                "relationships": [{"subject": "Alice", "predicate": "likes", "object": "jasmine tea"}],
            }

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes jasmine tea.", source="notes")),
        MemoryStore(),
    )
    fake_runtime = _FakeRuntime()
    rep = TripleRepresentation(method="direct", prompt="Keep only durable preference triples.")
    rep._runtime = lambda: fake_runtime  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    llm_call = next(call for call in fake_runtime.calls if call.get("name") == "MemPrimitiveTripleDirectAgent")
    assert llm_call["payload"]["prompt"] == "Keep only durable preference triples."
    assert llm_call["payload"]["text"] == "Alice likes jasmine tea."
    assert packet_out.trace["representation"]["prompt_is_template"] is False
    assert packet_out.trace["representation"]["per_unit"][0]["rendered_prompt"] == "Keep only durable preference triples."
    assert packet_out.trace["representation"]["per_unit"][0]["missing_variables"] == []
    assert packet_out.units[0].triples == [("Alice", "likes", "jasmine tea")]


def test_triple_representation_prompt_template_renders_unit_context_and_trace() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def require_llm(self, *, capability: str) -> None:
            self.calls.append({"capability": capability})

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            payload = json.loads(input_text)
            self.calls.append({"name": name, "payload": payload})
            return {
                "entities": ["Alice", "tea"],
                "relationships": [{"subject": "Alice", "predicate": "likes", "object": "tea"}],
            }

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice likes tea.",
                source="notes",
                metadata={"session_id": "sess-1"},
            )
        ),
        MemoryStore(),
    )
    fake_runtime = _FakeRuntime()
    rep = TripleRepresentation(
        method="direct",
        prompt="Extract triples for {{ unit.text }} in {{ unit.metadata.session_id | default('none') }}.",
    )
    rep._runtime = lambda: fake_runtime  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    llm_call = next(call for call in fake_runtime.calls if call.get("name") == "MemPrimitiveTripleDirectAgent")
    assert llm_call["payload"]["prompt"] == "Extract triples for Alice likes tea. in sess-1."
    assert packet_out.trace["representation"]["prompt_is_template"] is True
    assert packet_out.trace["representation"]["per_unit"][0]["rendered_prompt"] == "Extract triples for Alice likes tea. in sess-1."
    assert packet_out.trace["representation"]["per_unit"][0]["missing_variables"] == []


def test_triple_representation_prompt_template_missing_variables_do_not_crash() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def require_llm(self, *, capability: str) -> None:
            return None

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            payload = json.loads(input_text)
            assert payload["prompt"] == "Extract  from Alice likes tea."
            return {"entities": ["Alice", "tea"], "relationships": [{"subject": "Alice", "predicate": "likes", "object": "tea"}]}

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        MemoryStore(),
    )
    rep = TripleRepresentation(method="direct", prompt="Extract {{ unit.metadata.unknown_key }} from {{ unit.text }}")
    rep._runtime = lambda: _FakeRuntime()  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    assert "unit.metadata.unknown_key" in packet_out.trace["representation"]["per_unit"][0]["missing_variables"]


def test_triple_representation_without_prompt_keeps_default_llm_payload_shape() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def require_llm(self, *, capability: str) -> None:
            return None

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            payload = json.loads(input_text)
            self.payloads.append(payload)
            return {"entities": ["Alice", "tea"], "relationships": [{"subject": "Alice", "predicate": "likes", "object": "tea"}]}

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        MemoryStore(),
    )
    fake_runtime = _FakeRuntime()
    rep = TripleRepresentation(method="direct")
    rep._runtime = lambda: fake_runtime  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    assert fake_runtime.payloads == [{"text": "Alice likes tea."}]
    assert packet_out.trace["representation"]["prompt_is_template"] is False
    assert "rendered_prompt" not in packet_out.trace["representation"]["per_unit"][0]


def test_triple_representation_embed_extracted_from_metadata_hints() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def embed(self, text: str) -> list[float]:
            return [float(len(text)), 1.0, 2.0]

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice likes tea.",
                source="notes",
                metadata={
                    "entities": ["Alice", "tea"],
                    "triples": [("Alice", "likes", "tea")],
                },
            )
        ),
        MemoryStore(),
    )
    rep = TripleRepresentation(method="direct", embed_extracted=True)
    rep._runtime = lambda: _FakeRuntime()  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.embedding == [54.0, 1.0, 2.0]
    assert "embedding" in unit.representation_elements
    assert unit.metadata["representation"]["embedding"] == {"dim": 3}
    assert packet_out.trace["representation"]["embed_extracted"] is True


def test_triple_representation_embed_extracted_direct_llm_path() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def require_llm(self, *, capability: str) -> None:
            return None

        def embed(self, text: str) -> list[float]:
            return [9.0, float(text.count("Alice")), float(text.count("tea"))]

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            return {
                "entities": ["Alice", "tea"],
                "relationships": [{"subject": "Alice", "predicate": "likes", "object": "tea"}],
            }

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        MemoryStore(),
    )
    rep = TripleRepresentation(method="direct", embed_extracted=True)
    rep._runtime = lambda: _FakeRuntime()  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.embedding == [9.0, 2.0, 2.0]
    assert unit.metadata["representation"]["embedding"] == {"dim": 3}


def test_triple_representation_embed_extracted_two_stage_llm_path() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def require_llm(self, *, capability: str) -> None:
            return None

        def embed(self, text: str) -> list[float]:
            return [7.0, float(text.count("Bob")), float(text.count("graphs"))]

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            if name == "MemPrimitiveTripleEntityAgent":
                return {"entities": ["Bob", "graphs"]}
            return {"relationships": [{"subject": "Bob", "predicate": "studies", "object": "graphs"}]}

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Bob studies graphs.", source="notes")),
        MemoryStore(),
    )
    rep = TripleRepresentation(method="two_stage", embed_extracted=True)
    rep._runtime = lambda: _FakeRuntime()  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.embedding == [7.0, 2.0, 2.0]
    assert unit.metadata["representation"]["embedding"] == {"dim": 3}


def test_triple_representation_embed_extracted_skips_empty_graph_payload() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="No structured graph fact.", source="notes")),
        MemoryStore(),
    )
    rep = TripleRepresentation(method="direct", embed_extracted=True)

    represented = rep._replace_unit(
        unit_packet.units[0],
        "No structured graph fact.",
        "no structured graph fact.",
        [],
        [],
    )

    assert represented.embedding is None
    assert "embedding" not in represented.representation_elements


def test_triple_representation_embed_entities_from_metadata_hints() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def embed(self, text: str) -> list[float]:
            return [float(len(text)), float(text.count("a")), float(text.count("e"))]

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice likes tea.",
                source="notes",
                metadata={
                    "entities": ["Alice", "tea"],
                    "triples": [("Alice", "likes", "tea")],
                },
            )
        ),
        MemoryStore(),
    )
    rep = TripleRepresentation(method="direct", embed_entities=True)
    rep._runtime = lambda: _FakeRuntime()  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.embedding is None
    assert unit.metadata["representation"]["entity_embeddings"] == {
        "Alice": [5.0, 0.0, 1.0],
        "tea": [3.0, 1.0, 1.0],
    }
    assert packet_out.trace["representation"]["embed_entities"] is True


def test_triple_representation_embed_entities_direct_llm_path_can_coexist_with_graph_embedding() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def require_llm(self, *, capability: str) -> None:
            return None

        def embed(self, text: str) -> list[float]:
            lowered = text.casefold()
            return [float(len(text)), 10.0 if lowered == "alice" else 0.0, 5.0 if lowered == "tea" else 0.0]

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            return {
                "entities": ["Alice", "tea"],
                "relationships": [{"subject": "Alice", "predicate": "likes", "object": "tea"}],
            }

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        MemoryStore(),
    )
    rep = TripleRepresentation(method="direct", embed_extracted=True, embed_entities=True)
    rep._runtime = lambda: _FakeRuntime()  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.embedding == [54.0, 0.0, 0.0]
    assert unit.metadata["representation"]["embedding"] == {"dim": 3}
    assert unit.metadata["representation"]["entity_embeddings"] == {
        "Alice": [5.0, 10.0, 0.0],
        "tea": [3.0, 0.0, 5.0],
    }


def test_triple_representation_embed_entities_two_stage_llm_path() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def require_llm(self, *, capability: str) -> None:
            return None

        def embed(self, text: str) -> list[float]:
            return [7.0, float(text.count("Bob")), float(text.count("graphs"))]

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            if name == "MemPrimitiveTripleEntityAgent":
                return {"entities": ["Bob", "graphs"]}
            return {"relationships": [{"subject": "Bob", "predicate": "studies", "object": "graphs"}]}

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Bob studies graphs.", source="notes")),
        MemoryStore(),
    )
    rep = TripleRepresentation(method="two_stage", embed_entities=True)
    rep._runtime = lambda: _FakeRuntime()  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.embedding is None
    assert unit.metadata["representation"]["entity_embeddings"] == {
        "Bob": [7.0, 1.0, 0.0],
        "graphs": [7.0, 0.0, 1.0],
    }


def test_triple_representation_two_stage_reuses_one_prompt_for_both_calls() -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def require_llm(self, *, capability: str) -> None:
            return None

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            payload = json.loads(input_text)
            self.calls.append({"name": name, "payload": payload, "instructions": instructions})
            if name == "MemPrimitiveTripleEntityAgent":
                return {"entities": ["Alice", "Bob"]}
            return {"relationships": [{"subject": "Alice", "predicate": "mentors", "object": "Bob"}]}

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice mentors Bob.", source="notes")),
        MemoryStore(),
    )
    fake_runtime = _FakeRuntime()
    rep = TripleRepresentation(method="two_stage", prompt="Keep only durable people and relationship triples.")
    rep._runtime = lambda: fake_runtime  # type: ignore[method-assign]

    packet_out, _ = rep.run(unit_packet, store)

    entity_call = next(call for call in fake_runtime.calls if call["name"] == "MemPrimitiveTripleEntityAgent")
    relation_call = next(call for call in fake_runtime.calls if call["name"] == "MemPrimitiveTripleRelationAgent")
    assert entity_call["payload"]["prompt"] == "Keep only durable people and relationship triples."
    assert relation_call["payload"]["prompt"] == "Keep only durable people and relationship triples."
    assert relation_call["payload"]["entities"] == ["Alice", "Bob"]
    assert packet_out.units[0].entities == ["Alice", "Bob"]
    assert packet_out.units[0].triples == [("Alice", "mentors", "Bob")]


def test_mem0g_example_builds_with_prompt_driven_triple_representation() -> None:
    from memprimitive.baselines import TripleRepresentation
    from memprimitive.example.classics.mem0g_memory import build_mem0g_memory_system

    system = build_mem0g_memory_system()
    pipeline = system["mem0g_write_pipeline"]
    representation = pipeline.representation

    assert isinstance(representation, tuple)
    assert isinstance(representation[0], TripleRepresentation)
    assert representation[0].prompt is not None
    assert representation[0].embed_extracted is True


def test_basic_representation_rejects_summary_and_description_elements() -> None:
    from memprimitive.baselines import BasicRepresentation

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "summary"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "description"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "kv"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "entities"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "tags"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "relation_tags"))


def test_llm_representation_requires_openai_config() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(
        field="summary",
        prompt="Extract a one-sentence summary.",
        api_key="",
        base_url="",
        model="",
    )
    with pytest.raises(ValueError, match="LLMRepresentation field 'summary'.*MEMPRIMITIVE"):
        rep.run(unit_packet, store)


def test_llm_representation_writes_known_list_field_to_unit() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(field="tags", prompt="Extract retrieval tags.")

    def _fake_llm_json(*, user: str) -> Any:
        payload = json.loads(user)
        assert payload["field"] == "tags"
        assert payload["prompt"] == "Extract retrieval tags."
        assert payload["unit"]["text"] == "Alice studies graph memory systems."
        return ["graph-memory", "research"]

    rep._llm_json = _fake_llm_json  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.tags == ["graph-memory", "research"]
    assert "tags" in unit.representation_elements
    assert unit.metadata["representation"]["tags"] == ["graph-memory", "research"]
    assert packet_out.trace["representation"]["field"] == "tags"
    assert packet_out.trace["representation"]["per_unit"][0]["kind"] == "list"


def test_llm_representation_writes_summary_and_custom_fields_into_representation_metadata() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    summary_rep = LLMRepresentation(field="summary", prompt="Extract a one-sentence summary.")
    summary_rep._llm_text = lambda *, user: "Alice studies graph memory systems."  # type: ignore[method-assign]
    packet_out, _ = summary_rep.run(unit_packet, store)

    summary_unit = packet_out.units[0]
    assert summary_unit.metadata["representation"]["summary"] == "Alice studies graph memory systems."
    assert "summary" in summary_unit.representation_elements

    custom_rep = LLMRepresentation(field="custom_topic", prompt="Extract the main topic.")
    custom_rep._llm_text = lambda *, user: "graph memory"  # type: ignore[method-assign]
    packet_out, _ = custom_rep.run(packet_out, store)

    custom_unit = packet_out.units[0]
    assert custom_unit.metadata["representation"]["custom_topic"] == "graph memory"
    assert "custom_topic" in custom_unit.representation_elements


def test_llm_representation_custom_field_explicit_str_type_matches_text_path() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(field="custom_topic", prompt="Extract the main topic.", value_type=str)
    rep._llm_text = lambda *, user: "graph memory"  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.metadata["representation"]["custom_topic"] == "graph memory"
    assert packet_out.trace["representation"]["declared_value_type"] == "str"
    assert packet_out.trace["representation"]["per_unit"][0]["kind"] == "text"
    assert packet_out.trace["representation"]["per_unit"][0]["declared_value_type"] == "str"


def test_llm_representation_custom_field_explicit_list_type_writes_json_list() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(field="custom_topics", prompt="Extract topics.", value_type=list[str])
    rep._llm_json = lambda *, user: ["graph-memory", "research"]  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.metadata["representation"]["custom_topics"] == ["graph-memory", "research"]
    assert packet_out.trace["representation"]["declared_value_type"] == "list[str]"
    assert packet_out.trace["representation"]["per_unit"][0]["kind"] == "list"


def test_llm_representation_custom_field_explicit_dict_type_writes_json_dict() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(field="custom_slots", prompt="Extract slots.", value_type=dict[str, str])
    rep._llm_json = lambda *, user: {"topic": "graph memory", "goal": "research"}  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.metadata["representation"]["custom_slots"] == {"topic": "graph memory", "goal": "research"}
    assert packet_out.trace["representation"]["declared_value_type"] == "dict[str, str]"
    assert packet_out.trace["representation"]["per_unit"][0]["kind"] == "dict"


def test_llm_representation_custom_field_explicit_dict_list_type_writes_json_dict_list() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(field="custom_thoughts", prompt="Extract thoughts.", value_type=list[dict[str, str]])
    rep._llm_json = lambda *, user: [  # type: ignore[method-assign]
        {"thought": " Graph memory helps retrieval. ", "topic": " research "},
        {"thought": "Graph memory helps retrieval.", "topic": "research"},
        {"thought": "Structured traces help debugging.", "topic": 7},
        {"thought": "", "topic": "ignored"},
        {"thought": "ignored", "": "blank-key"},
        {},
        "not-a-dict",
    ]
    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.metadata["representation"]["custom_thoughts"] == [
        {"thought": "Graph memory helps retrieval.", "topic": "research"},
        {"thought": "Structured traces help debugging.", "topic": "7"},
        {"topic": "ignored"},
        {"thought": "ignored"},
    ]
    assert packet_out.trace["representation"]["declared_value_type"] == "list[dict[str, str]]"
    assert packet_out.trace["representation"]["per_unit"][0]["kind"] == "dict_list"


def test_llm_representation_known_field_explicit_list_type_preserves_unit_writeback() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(field="tags", prompt="Extract retrieval tags.", value_type=list[str])
    rep._llm_json = lambda *, user: ["graph-memory", "research"]  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.tags == ["graph-memory", "research"]
    assert unit.metadata["representation"]["tags"] == ["graph-memory", "research"]
    assert packet_out.trace["representation"]["per_unit"][0]["kind"] == "list"


def test_llm_representation_rejects_unsupported_value_types() -> None:
    from memprimitive.baselines import LLMRepresentation

    with pytest.raises(ValueError, match="value_type only supports str, list\\[str\\], dict\\[str, str\\], or list\\[dict\\[str, str\\]\\]"):
        LLMRepresentation(field="custom_score", prompt="Extract score.", value_type=int)

    with pytest.raises(ValueError, match="value_type only supports str, list\\[str\\], dict\\[str, str\\], or list\\[dict\\[str, str\\]\\]"):
        LLMRepresentation(field="custom_scores", prompt="Extract scores.", value_type=list[int])

    with pytest.raises(ValueError, match="value_type only supports str, list\\[str\\], dict\\[str, str\\], or list\\[dict\\[str, str\\]\\]"):
        LLMRepresentation(field="custom_items", prompt="Extract items.", value_type=list)


def test_llm_representation_prompt_template_renders_unit_context_and_trace() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice studies graph memory systems.",
                source="notes",
                metadata={"session_id": "sess-1"},
            )
        ),
        MemoryStore(),
    )
    rep = LLMRepresentation(
        field="summary",
        prompt="Extract {{ field }} for {{ unit.unit_type }}: {{ unit.text }} / {{ unit.metadata.session_id | default('none') }}",
    )

    def _fake_llm_text(*, user: str) -> str:
        payload = json.loads(user)
        assert payload["prompt"] == "Extract summary for observation: Alice studies graph memory systems. / sess-1"
        return "templated summary"

    rep._llm_text = _fake_llm_text  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    assert packet_out.units[0].metadata["representation"]["summary"] == "templated summary"
    assert packet_out.trace["representation"]["prompt_is_template"] is True
    assert packet_out.trace["representation"]["per_unit"][0]["rendered_prompt"].startswith("Extract summary")
    assert packet_out.trace["representation"]["per_unit"][0]["missing_variables"] == []


def test_llm_representation_prompt_template_missing_variables_do_not_crash() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(field="summary", prompt="Extract {{ unit.metadata.unknown_key }} from {{ unit.text }}")

    def _fake_llm_text(*, user: str) -> str:
        payload = json.loads(user)
        assert payload["prompt"] == "Extract  from Alice studies graph memory systems."
        return "summary with missing field"

    rep._llm_text = _fake_llm_text  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    assert "unit.metadata.unknown_key" in packet_out.trace["representation"]["per_unit"][0]["missing_variables"]


def test_llm_representation_prompt_template_supports_recalled_prompt_from_current_store() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMRepresentation, PassThroughUnitFormation, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import text_prompt

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice is preparing a reply.", source="notes")),
        MemoryStore(),
    )
    _seed_layer(store, "default", ["CURRENT STORE PROFILE"])

    pipeline_store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")]))
    _seed_layer(pipeline_store, "default", ["WRONG PIPELINE STORE PROFILE"])
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="default"),
        readout=ConcatenateReadout(),
        store=pipeline_store,
    )

    rep = LLMRepresentation(
        field="summary",
        prompt=text_prompt(
            "Use {{ recalled_prompt }} while summarizing {{ unit.text }}",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"profile for {context['unit']['text']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
    )

    def _fake_llm_text(*, user: str) -> str:
        payload = json.loads(user)
        assert payload["prompt"] == "Use CURRENT STORE PROFILE while summarizing Alice is preparing a reply."
        return "summary with recalled prompt"

    rep._llm_text = _fake_llm_text  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    prompt_trace = packet_out.trace["representation"]["per_unit"][0]
    assert prompt_trace["recall_prompt"]["enabled"] is True
    assert prompt_trace["recall_prompt"]["rendered_recall_query"] == "profile for Alice is preparing a reply."
    assert prompt_trace["recalled_prompt"] == "CURRENT STORE PROFILE"


def test_llm_representation_prompt_template_empty_recalled_prompt_falls_back_to_empty_string() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMRepresentation, PassThroughUnitFormation, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import text_prompt

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice is preparing a reply.", source="notes")),
        MemoryStore(),
    )
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="default"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    rep = LLMRepresentation(
        field="summary",
        prompt=text_prompt(
            "prefix {{ recalled_prompt }} suffix",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"profile for {context['unit']['text']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
    )

    def _fake_llm_text(*, user: str) -> str:
        payload = json.loads(user)
        assert payload["prompt"] == "prefix  suffix"
        return "summary without recalled prompt"

    rep._llm_text = _fake_llm_text  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    prompt_trace = packet_out.trace["representation"]["per_unit"][0]
    assert prompt_trace["recall_prompt"]["enabled"] is True
    assert prompt_trace["recall_prompt"]["matched"] is False
    assert prompt_trace["recalled_prompt"] == ""


def test_memory_store_delete_record_removes_expected_record() -> None:
    store = MemoryStore()
    _seed_layer(store, "default", ["first", "second"])

    removed = store.delete_record("default", "rec-1")

    assert removed.record_id == "rec-1"
    assert [record.record_id for record in store.iter_records("default")] == ["rec-2"]


def test_memory_store_delete_record_rejects_unknown_record() -> None:
    store = MemoryStore()
    _seed_layer(store, "default", ["first"])

    with pytest.raises(KeyError, match="not found"):
        store.delete_record("default", "rec-missing")
