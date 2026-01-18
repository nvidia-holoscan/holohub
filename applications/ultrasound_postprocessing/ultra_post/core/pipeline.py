# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import yaml
from ultra_post.filters.registry import DEFAULT_PARAMS, FILTERS

CONFIG_VERSION = "1"

Tensor = Any  # CuPy ndarray at runtime; kept loose to avoid import-time dependency.
PipelineNode = dict[str, Any]
Pipeline = list[PipelineNode]


def create_node(
    op: str, params: Mapping[str, Any] | None = None, *, enabled: bool = True
) -> PipelineNode:
    """Merge defaults with params and produce a plain pipeline node dict."""

    base = dict(DEFAULT_PARAMS.get(op, {}))
    base.update(dict(params or {}))
    enable_flag = bool(base.pop("enable", enabled))
    return {"op": op, "params": base, "enabled": enable_flag}


def run_pipeline(
    pipeline: Pipeline, tensor: Tensor, *, filters: Mapping[str, Any] | None = None
) -> Tensor:
    """Apply enabled filters in sequence.

    :note: This function does not rely on Holoscan SDK.
    """

    registry = filters if filters is not None else FILTERS
    result = tensor
    for node in pipeline:
        if not node.get("enabled", True):
            continue
        filter_name = node.get("op")
        if not isinstance(filter_name, str):
            raise ValueError("Pipeline node missing 'op' string.")
        func = registry.get(filter_name)
        if func is None:
            raise KeyError(f"Filter '{filter_name}' not found.")
        params = node.get("params") or {}
        kwargs = _params_to_kwargs(params)

        # Instantiate stateful filters stored as classes in the registry.
        # Cache the instance on the node to preserve state across frames.
        if isinstance(func, type):
            inst_key = "__instance__"
            inst = node.get(inst_key)
            if not isinstance(inst, func):
                inst = func()
                node[inst_key] = inst
            func = inst

        result = func(result, **kwargs)
    return result


def pipeline_to_dict(
    pipeline: Iterable[PipelineNode], *, display: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Convert a pipeline into a serializable dictionary."""

    graph = [_node_to_config(node) for node in pipeline]
    data: dict[str, Any] = {"version": CONFIG_VERSION, "graph": graph}
    if display:
        data["display"] = display
    return data


def pipeline_from_dict(
    data: Mapping[str, Any], *, filters: Mapping[str, Any] | None = None
) -> Pipeline:
    """Create a pipeline from a parsed configuration dictionary."""

    if not isinstance(data, Mapping):
        raise ValueError("Pipeline configuration must be a mapping.")

    version = str(data.get("version", CONFIG_VERSION))
    if version != CONFIG_VERSION:
        raise ValueError(f"Unsupported pipeline config version '{version}'.")

    graph = data.get("graph")
    if not isinstance(graph, Iterable):
        raise ValueError("Pipeline configuration missing 'graph' list.")

    nodes: Pipeline = []
    registry = filters if filters is not None else FILTERS
    for node in graph:
        if not isinstance(node, Mapping):
            raise ValueError("Each pipeline node must be a mapping.")
        op_name = node.get("op")
        if not isinstance(op_name, str):
            raise ValueError("Pipeline node missing 'op' string.")
        params = node.get("params") or {}
        if not isinstance(params, Mapping):
            raise ValueError("Pipeline node 'params' must be a mapping.")
        params = dict(params)
        enabled = bool(node.get("enabled", params.pop("enable", True)))

        if op_name not in registry:
            raise KeyError(f"Filter '{op_name}' is not available.")

        nodes.append(create_node(op_name, params, enabled=enabled))

    return nodes


def dump_pipeline_config(
    path: str | Path, pipeline: Iterable[PipelineNode], *, display: Optional[dict[str, Any]] = None
) -> None:
    """Serialize the pipeline to a YAML file."""

    data = pipeline_to_dict(list(pipeline), display=display)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def load_pipeline_config(path: str | Path, *, filters: Mapping[str, Any] | None = None) -> Pipeline:
    """Load a pipeline from a YAML file."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pipeline config not found: {resolved}")

    text = resolved.read_text(encoding="utf-8")
    return pipeline_from_yaml(text, filters=filters)


def pipeline_to_yaml(
    pipeline: Iterable[PipelineNode], *, display: Optional[dict[str, Any]] = None
) -> str:
    """Return a YAML string representing the pipeline."""

    return yaml.safe_dump(pipeline_to_dict(list(pipeline), display=display), sort_keys=False)


def pipeline_from_yaml(
    data: str | bytes, *, filters: Mapping[str, Any] | None = None, include_config: bool = False
) -> Union[Pipeline, Tuple[Pipeline, Mapping[str, Any]]]:
    """Parse a pipeline YAML string into a pipeline list.

    Set ``include_config=True`` to receive a tuple
    ``(pipeline, original_config_mapping)``.
    """

    if isinstance(data, bytes):
        data = data.decode("utf-8")

    parsed: Optional[Any] = yaml.safe_load(data)
    if parsed is None:
        parsed = {"version": CONFIG_VERSION, "graph": []}

    if not isinstance(parsed, Mapping):
        raise ValueError("Pipeline YAML must describe a mapping.")

    pipeline = pipeline_from_dict(parsed, filters=filters)
    result: Union[Pipeline, Tuple[Pipeline, Mapping[str, Any]]] = pipeline
    if include_config:
        result = (pipeline, dict(parsed))
    return result


def _node_to_config(node: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"op": node.get("op"), "params": _normalize(node.get("params") or {})}
    if not node.get("enabled", True):
        payload["enabled"] = False
    return payload


def _normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare params for YAML serialization (handle Enums and tuples)."""

    normalized: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, Enum):
            normalized[k] = v.value
        elif isinstance(v, tuple):
            normalized[k] = list(v)
        else:
            normalized[k] = v
    return normalized


def _params_to_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(params, "model_dump"):
        return dict(params.model_dump())
    return dict(params)


__all__ = [
    "CONFIG_VERSION",
    "Pipeline",
    "PipelineNode",
    "create_node",
    "dump_pipeline_config",
    "load_pipeline_config",
    "pipeline_from_dict",
    "pipeline_from_yaml",
    "pipeline_to_dict",
    "pipeline_to_yaml",
    "run_pipeline",
]
