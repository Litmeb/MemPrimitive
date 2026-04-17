from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from memprimitive import MemoryPipeline, Observation, Query, StoreTopology
from memprimitive.baselines import LLMRepresentation, TemplateReadout
from memprimitive.config import (
    ConfigLoaderError,
    ConfigResolutionError,
    load_object_from_dict,
    load_pipeline_from_dict,
    load_pipeline_from_yaml,
)


def _simple_pipeline_config() -> dict[str, object]:
    return {
        "version": 1,
        "root": "pipeline",
        "objects": {
            "shared_store": {
                "$call": "memprimitive.core.MemoryStore",
            },
            "pipeline": {
                "$call": "memprimitive.pipeline.MemoryPipeline",
                "kwargs": {
                    "unit_formation": "PassThroughUnitFormation",
                    "representation": {
                        "$call": "BasicRepresentation",
                        "kwargs": {
                            "elements": ["text"],
                        },
                    },
                    "write_trigger": "AlwaysTrigger",
                    "organization": {
                        "$call": "AppendOrganization",
                        "kwargs": {
                            "target_layer": "default",
                        },
                    },
                    "evolution_trigger": "NeverTrigger",
                    "memory_evolution": "AppendOnlyEvolution",
                    "retrieval": {
                        "$call": "RecencyRetrieval",
                        "kwargs": {
                            "top_k": 2,
                            "layer": "default",
                        },
                    },
                    "readout": "ConcatenateReadout",
                    "store": {
                        "$ref": "shared_store",
                    },
                },
            },
        },
    }


def _nested_template_pipeline_config() -> dict[str, object]:
    return {
        "version": 1,
        "root": "pipeline",
        "objects": {
            "shared_store": {
                "$call": "memprimitive.core.MemoryStore",
            },
            "sub_recall_pipeline": {
                "$call": "memprimitive.pipeline.MemoryPipeline",
                "kwargs": {
                    "retrieval": {
                        "$call": "RecencyRetrieval",
                        "kwargs": {
                            "top_k": 1,
                            "layer": "default",
                        },
                    },
                    "readout": "ConcatenateReadout",
                    "store": {
                        "$ref": "shared_store",
                    },
                },
            },
            "pipeline": {
                "$call": "memprimitive.pipeline.MemoryPipeline",
                "kwargs": {
                    "representation": {
                        "$call": "BasicRepresentation",
                        "kwargs": {
                            "elements": ["text"],
                        },
                    },
                    "organization": {
                        "$call": "AppendOrganization",
                        "kwargs": {
                            "target_layer": "default",
                        },
                    },
                    "retrieval": {
                        "$call": "RecencyRetrieval",
                        "kwargs": {
                            "top_k": 1,
                            "layer": "default",
                        },
                    },
                    "readout": {
                        "$call": "TemplateReadout",
                        "kwargs": {
                            "prompt": {
                                "$call": "memprimitive.utils._template.text_prompt",
                                "kwargs": {
                                    "template": (
                                        "Primary retrieval:\n{{ retrieved.items | join_text }}\n\n"
                                        "Shared-store sub recall:\n{{ recalled_prompt }}"
                                    ),
                                    "recall_plan": {
                                        "$call": "memprimitive.utils._template.text_prompt",
                                        "kwargs": {
                                            "template": "{{ retrieved.items | join_text }}",
                                            "metadata_mode": "readout",
                                        },
                                    },
                                    "recall_query_builder": {
                                        "$import": "memprimitive.example.demonstration.recalled_prompt_template.build_recalled_readout_query",
                                    },
                                    "sub_recall_pipeline": {
                                        "$ref": "sub_recall_pipeline",
                                    },
                                },
                            },
                        },
                    },
                    "store": {
                        "$ref": "shared_store",
                    },
                },
            },
        },
    }


def _nested_llm_representation_config() -> dict[str, object]:
    return {
        "version": 1,
        "root": "pipeline",
        "objects": {
            "shared_store": {
                "$call": "memprimitive.core.MemoryStore",
            },
            "sub_recall_pipeline": {
                "$call": "memprimitive.pipeline.MemoryPipeline",
                "kwargs": {
                    "retrieval": {
                        "$call": "RecencyRetrieval",
                        "kwargs": {
                            "top_k": 1,
                            "layer": "default",
                        },
                    },
                    "readout": "ConcatenateReadout",
                    "store": {
                        "$ref": "shared_store",
                    },
                },
            },
            "pipeline": {
                "$call": "memprimitive.pipeline.MemoryPipeline",
                "kwargs": {
                    "unit_formation": "PassThroughUnitFormation",
                    "representation": {
                        "$call": "LLMRepresentation",
                        "kwargs": {
                            "field": "summary",
                            "prompt": {
                                "$call": "memprimitive.utils._template.text_prompt",
                                "kwargs": {
                                    "template": (
                                        "Summarize the memory.\n"
                                        "Recalled context:\n{{ recalled_prompt }}\n\n"
                                        "Current memory:\n{{ unit.text }}"
                                    ),
                                    "recall_plan": {
                                        "$call": "memprimitive.utils._template.text_prompt",
                                        "kwargs": {
                                            "template": "{{ retrieved.items | join_text }}",
                                            "metadata_mode": "readout",
                                        },
                                    },
                                    "recall_query_builder": {
                                        "$import": "memprimitive.example.demonstration.recalled_prompt_template.build_recalled_prompt_query",
                                    },
                                    "sub_recall_pipeline": {
                                        "$ref": "sub_recall_pipeline",
                                    },
                                },
                            },
                        },
                    },
                    "organization": {
                        "$call": "AppendOrganization",
                        "kwargs": {
                            "target_layer": "default",
                        },
                    },
                    "store": {
                        "$ref": "shared_store",
                    },
                },
            },
        },
    }


@pytest.mark.parametrize(
    ("payload", "expected_path"),
    [
        ({}, "version"),
        ({"version": 1}, "root"),
        ({"version": 1, "root": "pipeline"}, "objects"),
        ({"version": "1", "root": "pipeline", "objects": {}}, "version"),
        ({"version": 1, "root": "", "objects": {}}, "root"),
        ({"version": 1, "root": "pipeline", "objects": []}, "objects"),
    ],
)
def test_load_object_from_dict_rejects_invalid_top_level_shape(
    payload: dict[str, object],
    expected_path: str,
) -> None:
    with pytest.raises(ConfigResolutionError, match=expected_path):
        load_object_from_dict(payload)


def test_load_object_from_dict_supports_call_args_and_forward_refs() -> None:
    config = {
        "version": 1,
        "root": "topology",
        "objects": {
            "topology": {
                "$call": "memprimitive.core.StoreTopology.from_layers",
                "args": [
                    [
                        {
                            "$ref": "profile_layer",
                        }
                    ]
                ],
            },
            "profile_layer": {
                "$call": "memprimitive.core.StoreLayerSpec",
                "kwargs": {
                    "name": "profile",
                    "theme": "semantic",
                    "indices": ["temporal"],
                },
            },
        },
    }

    topology = load_object_from_dict(config)

    assert isinstance(topology, StoreTopology)
    assert topology.layer_names == ("profile",)


def test_load_object_from_dict_detects_circular_refs() -> None:
    config = {
        "version": 1,
        "root": "a",
        "objects": {
            "a": {"$ref": "b"},
            "b": {"$ref": "a"},
        },
    }

    with pytest.raises(ConfigResolutionError, match=r"objects\.b: circular \$ref detected: a -> b -> a\."):
        load_object_from_dict(config)


def test_load_object_from_dict_imports_named_symbol() -> None:
    config = {
        "version": 1,
        "root": "helper",
        "objects": {
            "helper": {
                "$import": "memprimitive.example.demonstration.recalled_prompt_template.build_recalled_prompt_query",
            },
        },
    }

    helper = load_object_from_dict(config)

    assert callable(helper)
    assert helper.__name__ == "build_recalled_prompt_query"


def test_load_object_from_dict_reports_import_failure_with_path() -> None:
    config = {
        "version": 1,
        "root": "broken",
        "objects": {
            "broken": {
                "$import": "memprimitive.example.demonstration.recalled_prompt_template.not_real",
            },
        },
    }

    with pytest.raises(ConfigResolutionError, match=r"objects\.broken: failed to import"):
        load_object_from_dict(config)


def test_load_object_from_dict_reports_call_failure_with_path() -> None:
    config = {
        "version": 1,
        "root": "broken",
        "objects": {
            "broken": {
                "$call": "RecencyRetrieval",
                "kwargs": {
                    "top_k": 0,
                },
            },
        },
    }

    with pytest.raises(ConfigResolutionError, match=r"objects\.broken: failed to call .*top_k"):
        load_object_from_dict(config)


def test_load_pipeline_from_dict_rejects_non_pipeline_root() -> None:
    config = {
        "version": 1,
        "root": "store",
        "objects": {
            "store": {
                "$call": "memprimitive.core.MemoryStore",
            },
        },
    }

    with pytest.raises(ConfigResolutionError, match=r"root: resolved root object must be MemoryPipeline, got MemoryStore\."):
        load_pipeline_from_dict(config)


def test_load_pipeline_from_dict_auto_infers_trigger_slots_for_explicit_calls() -> None:
    config = {
        "version": 1,
        "root": "pipeline",
        "objects": {
            "pipeline": {
                "$call": "memprimitive.pipeline.MemoryPipeline",
                "kwargs": {
                    "write_trigger": {
                        "$call": "AlwaysTrigger",
                    },
                    "evolution_trigger": {
                        "$call": "NeverTrigger",
                    },
                },
            },
        },
    }

    pipeline = load_pipeline_from_dict(config)

    assert pipeline.write_trigger.spec.slot == "write_trigger"
    assert pipeline.evolution_trigger.spec.slot == "evolution_trigger"


def test_load_pipeline_from_dict_builds_simple_pipeline_and_runs_round_trip() -> None:
    pipeline = load_pipeline_from_dict(_simple_pipeline_config())

    assert isinstance(pipeline, MemoryPipeline)

    pipeline.ingest(Observation(text="Alice likes graph retrieval.", source="dialogue"))
    readout = pipeline.recall(Query(text="Alice"))

    assert "Alice likes graph retrieval." in readout.text
    assert readout.source_ids


def test_load_pipeline_from_dict_builds_nested_llm_representation_prompt_graph() -> None:
    pipeline = load_pipeline_from_dict(_nested_llm_representation_config())

    assert isinstance(pipeline.representation, LLMRepresentation)
    assert callable(pipeline.representation.prompt.recall_query_builder)
    assert isinstance(pipeline.representation.prompt.sub_recall_pipeline, MemoryPipeline)
    assert pipeline.representation.prompt.sub_recall_pipeline.store is pipeline.store


def test_load_pipeline_from_dict_supports_shared_store_with_nested_sub_recall_runtime() -> None:
    pipeline = load_pipeline_from_dict(_nested_template_pipeline_config())

    assert isinstance(pipeline.readout, TemplateReadout)
    assert isinstance(pipeline.readout.prompt.sub_recall_pipeline, MemoryPipeline)
    assert pipeline.readout.prompt.sub_recall_pipeline.store is pipeline.store

    pipeline.ingest(Observation(text="Alice keeps one shared memory store.", source="dialogue"))
    readout = pipeline.recall(Query(text="Alice"))

    assert "Primary retrieval:" in readout.text
    assert "Shared-store sub recall:" in readout.text
    assert "Alice keeps one shared memory store." in readout.text


def test_load_pipeline_from_yaml_reads_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "simple_pipeline.yml"
    config_path.write_text(yaml.safe_dump(_simple_pipeline_config(), sort_keys=False), encoding="utf-8")

    pipeline = load_pipeline_from_yaml(config_path)

    assert isinstance(pipeline, MemoryPipeline)


def test_cli_validate_reports_success_and_failure(tmp_path: Path) -> None:
    good_path = tmp_path / "good.yml"
    bad_path = tmp_path / "bad.yml"
    good_path.write_text(yaml.safe_dump(_simple_pipeline_config(), sort_keys=False), encoding="utf-8")
    bad_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "root": "broken",
                "objects": {
                    "broken": {
                        "$import": "memprimitive.example.demonstration.recalled_prompt_template.not_real",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    good_result = subprocess.run(
        [sys.executable, "-m", "memprimitive.config", "validate", str(good_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    bad_result = subprocess.run(
        [sys.executable, "-m", "memprimitive.config", "validate", str(bad_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert good_result.returncode == 0
    assert "Validation succeeded: MemoryPipeline(" in good_result.stdout

    assert bad_result.returncode == 1
    assert "Validation failed:" in bad_result.stderr
    assert "objects.broken" in bad_result.stderr
