"""

The circulatory system of Octavius, managing how user-configured stages are run and determining what
raw data can be safely dropped between analysis stages.

"""

# type checking (semantic)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .conventions import OctaviusConfig
    from .snapshot_readers import SnapshotReader
    from .data_structures import ParticleStore

# default libraries
from dataclasses import dataclass
from yaml import safe_load
from pathlib import Path
from itertools import product

from ..log import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """
    Object containing metadata about an Octavius analysis stage.
    """

    name: str
    label: str
    requires: frozenset[str]
    applies_to: frozenset[str]
    cross_group_requirements: dict[str, frozenset[str]]
    needs_particle_columns: dict[str, frozenset[str]]
    optional_particle_columns: dict[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class Internals:
    """
    Internal pipeline management and naming dictionaries for data writing.
    """

    stages: dict[str, PipelineStage]
    baryonic_ptypes: frozenset[str]
    group_types: dict[str, dict]
    output_columns: dict[str, OutputColumnMetadata]
    membership_columns: dict[str, dict[str, OutputColumnMetadata]]
    header_fields: dict[str, OutputColumnMetadata]


@dataclass(frozen=True, slots=True)
class OutputColumnMetadata:
    """
    For stamping attributes on HDF5 datasets: dtype/unit/brief description of each column as specified in internals.yaml.
    """

    dtype: str
    unit: str
    a_exp: int
    description: str
    label: str


def load_internals(internals_filepath: Path, config: OctaviusConfig) -> Internals:
    """
    Loads stage definitions from internals.yaml, validates output columns, and
    returns the Internals dataclass which contains resolved metadata/ordering from internals.yaml
    """
    with open(internals_filepath, "r") as f:
        internals: dict[str, Any] = safe_load(f)

    raw_output_columns = internals["output_columns"]
    output_columns = {}

    for (
        template,
        meta,
    ) in (
        raw_output_columns.items()
    ):  # your IDE will grey this out because safe_load returns Any as type check (fear not it is a dict)
        if "over" in meta:
            resolved = resolve_over(meta["over"], config)

            for key, values in resolved.items():
                assert values, (
                    f"Output column template {template!r}: 'over' key {key!r} cannot find anything to iterate over."
                )

            expanded = expand_column_templates([template], resolved)

            for col_name in expanded:
                output_columns[col_name] = meta

        else:
            output_columns[template] = meta

    stages: dict[str, PipelineStage] = {}

    for stage_name, stage_config in internals["stages"].items():
        raw_needed = stage_config.get("needs_particle_columns", {})
        needed = {k: frozenset(v) for k, v in raw_needed.items()}

        raw_optional = stage_config.get("optional_particle_columns", {})
        optional = {k: frozenset(v) for k, v in raw_optional.items()}

        raw_cross_group = stage_config.get("cross_group_requirements", {})
        cross_groups = {k: frozenset(v) for k, v in raw_cross_group.items()}

        stages[stage_name] = PipelineStage(
            name=stage_name,
            label=stage_config["label"],
            requires=frozenset(stage_config.get("requires", [])),
            cross_group_requirements=cross_groups,
            applies_to=frozenset(stage_config["applies_to"]),
            needs_particle_columns=needed,
            optional_particle_columns=optional,
        )

    all_stage_outputs: set[str] = set()
    expanded_output_columns: dict[str, OutputColumnMetadata] = {}

    for stage_name, stage_config in internals["stages"].items():
        stage_label = stages[stage_name].label

        for sub_block in stage_config.get("outputs", []):
            templates = sub_block["columns"]

            if "over" in sub_block:
                resolved_over = resolve_over(sub_block["over"], config)

                for key, values in resolved_over.items():
                    assert len(values) > 0, (
                        f"Stage {stage_name!r}: 'over' key {key!r} cannot find anything to iterate over!"
                        f"Please check the from_config: reference."
                    )

                expanded = expand_column_templates(templates, resolved_over)

            else:
                expanded = list(templates)

            for col_name in expanded:
                assert col_name in output_columns, (
                    f"Stage {stage_name!r} says it outputs {col_name!r}. However, this is not found in output_columns."
                )
                assert col_name not in all_stage_outputs, f"Output {col_name!r} is being computed by multiple stages."
                all_stage_outputs.add(col_name)

                meta = output_columns[col_name]
                expanded_output_columns[col_name] = OutputColumnMetadata(
                    dtype=meta["dtype"],
                    unit=meta["unit"],
                    a_exp=meta["a_exp"],
                    description=meta.get("description", ""),
                    label=stage_label,
                )

    # validate all output columns are claimed by a stage
    unclaimed = set(output_columns.keys()) - all_stage_outputs
    assert not unclaimed, f"output_columns not claimed by any stage: {unclaimed}"

    membership_columns: dict[str, dict[str, OutputColumnMetadata]] = {}
    for group_name, columns in internals.get("membership_columns", {}).items():
        membership_columns[group_name] = {
            column_name: OutputColumnMetadata(
                dtype=meta["dtype"],
                unit=meta.get("unit", ""),
                a_exp=meta.get("a_exp", ""),
                description=meta.get("description", ""),
                label="membership",
            )
            for column_name, meta in columns.items()
        }

    header_fields: dict[str, OutputColumnMetadata] = {}
    for field_name, meta in internals.get("header_fields", {}).items():
        header_fields[field_name] = OutputColumnMetadata(
            dtype="float64",
            unit=meta["unit"],
            description=meta["description"],
            a_exp=meta["a_exp"],
            label="header",
        )

    return Internals(
        stages=stages,
        baryonic_ptypes=frozenset(internals["baryonic_ptypes"]),
        group_types=internals["group_types"],
        output_columns=expanded_output_columns,
        membership_columns=membership_columns,
        header_fields=header_fields,
    )


def load_stage_columns(
    particles: dict[str, ParticleStore],
    reader: SnapshotReader,
    stage: PipelineStage,
) -> None:
    """
    Automatically loads data into the ParticleStores depending on what was declared
    for the stage in internals.yaml.
    """
    _load_columns(particles=particles, reader=reader, spec=stage.needs_particle_columns, optional=False)
    _load_columns(particles=particles, reader=reader, spec=stage.optional_particle_columns, optional=True)


def release_stage_columns(
    particles: dict[str, ParticleStore],
    current_idx: int,
    ordered_stages: list[PipelineStage],
) -> None:
    """
    Releases data from ParticleStores automatically depending on the resolution of the
    stage dependency graph to free up memory.
    """
    current_needs = ordered_stages[current_idx].needs_particle_columns
    future_needs: dict[str, set[str]] = {}

    # check the stages ahead of this one and determine what data they need
    for stage in ordered_stages[current_idx + 1 :]:
        for ptype, columns in stage.needs_particle_columns.items():
            future_needs.setdefault(ptype, set()).update(columns)  # have to use update due to frozenset
        for ptype, columns in stage.optional_particle_columns.items():
            future_needs.setdefault(ptype, set()).update(columns)

    # determine what can be released, then drop it
    for ptype, columns in current_needs.items():
        target_ptypes = list(particles.keys()) if ptype == "all" else [ptype]

        for target in target_ptypes:
            if target not in particles:
                continue

            future_all = future_needs.get("all", set()) | future_needs.get(
                target, set()
            )  # set notation simplifies this
            releasable = columns - future_all

            for col in releasable:
                if col in particles[target]:
                    particles[target].release(col)


def validate_stage_requirements(
    ordered_stages: list[PipelineStage],
    available_ptypes: set[str],
) -> list[PipelineStage]:
    """
    Validates the ptypes enabled in the config and the stages the user requested are self-consistent:
    skips stages for which data does not exist. Returns:

    - validated: a list of stages which can run in the pipeline given the available data
    """
    dropped: set[str] = set()
    validated: list[PipelineStage] = []

    # NOTE: by this point the stages are topologically sorted
    for stage in ordered_stages:

        # dependency check
        missing_deps = stage.requires & dropped  # the sort means a requisite will be dropped before we arrive at its dependent
        if missing_deps:
            logger.warning(
                f"Skipping '{stage.name}': depends on dropped stages {", ".join(sorted(missing_deps))}."  # sort for identical log output between ranks
            )
            dropped.add(stage.name)
            continue

        # ptype check: this skips a stage if all are missing; stages should handle one missing internally
        specific_ptypes = {
            ptype for ptype in stage.needs_particle_columns if ptype != "all"
        }
        if specific_ptypes:  
            viable_ptypes = specific_ptypes & available_ptypes
            if not viable_ptypes:
                logger.warning(
                    f"Skipping '{stage.name}': requires ptypes {", ".join(sorted(specific_ptypes))}, "
                    f"but none are available."
                )
                dropped.add(stage.name)
                continue

        validated.append(stage)

    return validated


def _load_columns(
    particles: dict[str, ParticleStore],
    reader: SnapshotReader,
    spec: dict[str, frozenset[str]],
    optional: bool,
) -> None:
    """
    Loads requested columns from the snapshot.
    """
    for ptype, columns in spec.items():  # what to load
        target_ptypes = (
            list(particles.keys()) if ptype == "all" else [ptype]
        )  # particles.keys() only contains available ptypes

        for target in target_ptypes:
            if target not in particles:  # if users are disabling things in the config
                continue

            for column in sorted(
                columns
            ):  # sorted() is essential otherwise ranks desynchronise and crash (read different datasets)
                if column in particles[target]:  # if already present on the ParticleStore
                    continue
                if optional and not reader.has_dataset(target, column):
                    continue

                particles[target][column] = reader.read_dataset(ptype=target, dataset=column)


def resolve_over(over: dict[str, list | str], config: OctaviusConfig) -> dict[str, list[str]]:
    """
    Expands the "over:" field in internals.yaml.
    """
    resolved: dict[str, list[str]] = {}

    for key, val in over.items():
        if isinstance(val, str) and val.startswith("from_config:"):
            config_key = val.split(":", maxsplit=1)[1]
            config_value = getattr(config, config_key)

            if isinstance(config_value, dict):
                config_value = list(config_value.keys())  # needed for the radial quantiles dict in config

            resolved[key] = [str(v) for v in config_value]

        else:
            resolved[key] = [str(v) for v in val]

    return resolved


def expand_column_templates(outputs: list[str], over: dict[str, list[str]]) -> list[str]:
    """
    Expands an output column template in internals.yaml if they have ptype or
    quantity-specific fields (e.g. mass_star_30kpc), returning a list.
    """
    expanded: list[str] = []

    for template in outputs:
        matched_keys = [
            key for key in over if f"{{{key}}}" in template
        ]  # triple bracket is needed to parse the .yaml {}

        if not matched_keys:
            expanded.append(template)
            continue

        value_lists = [over[key] for key in matched_keys]

        for combo in product(*value_lists):
            result = template

            for key, val in zip(matched_keys, combo):
                result = result.replace(f"{{{key}}}", str(val))

            expanded.append(result)

    return expanded


def resolve_dependencies(stages: dict[str, PipelineStage], requested: list[str]) -> list[PipelineStage]:
    """
    Resolves user-requested stage dependencies.

    Returns a list of Stages in the correct order of execution.
    """
    needed = set()
    stack = list(requested)

    while stack:
        stage_name = stack.pop()  # final element
        assert stage_name in stages, f"{stage_name} is not in the supported stages; check typo?"

        if stage_name in needed:
            continue

        needed.add(stage_name)
        stack.extend(stages[stage_name].requires)  # see PipelineStage dataclass

    # NOTE: Kahn's algorithm for rigorous dependency resolution
    depth = {n: 0 for n in needed}
    dependents = {n: [] for n in needed}

    for n in needed:
        for req in stages[n].requires:
            if req in needed:
                depth[n] += 1
                dependents[req].append(n)

    # NOTE: need to sort stages such that all stages are executed in the same order across ranks to avoid MPI desync
    ready = sorted(n for n in needed if depth[n] == 0)  # top-level nodes
    ordered_stages = []

    while ready:
        ready.sort()  # sort each time a stage is added (see above comment ^)
        n = ready.pop()
        ordered_stages.append(stages[n])

        for m in dependents[n]:
            depth[m] -= 1

            if depth[m] == 0:
                ready.append(m)

    assert len(ordered_stages) == len(needed), "Stage dependencies are muddled; order unresolvable."

    return ordered_stages
