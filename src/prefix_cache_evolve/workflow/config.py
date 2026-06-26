"""Validated configuration models and providers for Levi workflows."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


class _StrictConfigModel(BaseModel):
    """Base class for validated repository configuration sections."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class LLMConfig(_StrictConfigModel):
    """LLM settings accepted by the workflow YAML."""

    default_provider: str = "openai"
    primary_model: str | None = None
    secondary_model: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    max_tokens: PositiveInt | None = None


class EvaluatorRuntimeConfig(_StrictConfigModel):
    """Evaluator process settings accepted by the workflow YAML."""

    timeout: PositiveFloat | None = None
    cascade_evaluation: bool | None = None
    parallel_evaluations: PositiveInt | None = None


class PromptConfig(_StrictConfigModel):
    """Prompt settings accepted by the workflow YAML."""

    system_message: str | None = None
    objectives: list[str] = Field(default_factory=list)


class ProblemConfig(_StrictConfigModel):
    """Problem description and problem-owned settings."""

    description: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class SearchConfig(_StrictConfigModel):
    """Search notes accepted by the workflow YAML."""

    seed: NonNegativeInt = 0
    notes: str | None = None


class WorkflowFileConfig(_StrictConfigModel):
    """Validated top-level schema for repository workflow YAML files."""

    max_iterations: PositiveInt | None = None
    function_signature: str | None = None
    description: str | None = None
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    evaluator: EvaluatorRuntimeConfig = Field(default_factory=EvaluatorRuntimeConfig)
    problem: ProblemConfig = Field(default_factory=ProblemConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    init: dict[str, Any] = Field(default_factory=dict)
    cvt: dict[str, Any] = Field(default_factory=dict)
    behavior: dict[str, Any] = Field(default_factory=dict)
    meta_advice: dict[str, Any] = Field(default_factory=dict)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    punctuated_equilibrium: dict[str, Any] = Field(default_factory=dict)
    prompt_overrides: dict[str, Any] = Field(default_factory=dict)
    cascade: dict[str, Any] = Field(default_factory=dict)
    run_cost: dict[str, Any] = Field(default_factory=dict)
    budget_dollars: NonNegativeFloat | None = None
    budget_seconds: NonNegativeFloat | None = None
    output_dir: str | None = None


def load_yaml_document(path: Path) -> object:
    """Safely load one YAML document, normalizing an empty document to a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {} if data is None else data


def yaml_documents_equal(left: Path, right: Path) -> bool:
    """Return whether two YAML files contain equal parsed documents."""
    try:
        return load_yaml_document(left) == load_yaml_document(right)
    except yaml.YAMLError:
        return False


def normalize_model_name(
    model: str | None,
    *,
    default_provider: str = "openai",
) -> str | None:
    """Return a provider-qualified LiteLLM model identifier."""
    if not model:
        return None
    normalized_model = model.strip()
    normalized_provider = default_provider.strip().rstrip("/")
    if not normalized_model:
        return None
    if "/" in normalized_model:
        return normalized_model
    if not normalized_provider:
        raise ValueError("llm.default_provider must not be empty")
    return f"{normalized_provider}/{normalized_model}"


class LeviRunConfig(BaseModel):
    """Resolved Levi configuration for one evolution run."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, validate_default=True)

    max_iterations: PositiveInt
    problem_description: str
    function_signature: str
    search_seed: NonNegativeInt = 0
    model: str | None = None
    paradigm_model: str | None = None
    mutation_model: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    budget_dollars: NonNegativeFloat | None = None
    budget_seconds: NonNegativeFloat | None = None
    output_dir: str | None = None
    pipeline: dict[str, Any] = Field(default_factory=dict)
    behavior: dict[str, Any] = Field(default_factory=dict)
    cvt: dict[str, Any] = Field(default_factory=dict)
    init: dict[str, Any] = Field(default_factory=dict)
    meta_advice: dict[str, Any] = Field(default_factory=dict)
    punctuated_equilibrium: dict[str, Any] = Field(default_factory=dict)
    prompt_overrides: dict[str, Any] = Field(default_factory=dict)
    cascade: dict[str, Any] = Field(default_factory=dict)
    run_cost: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    def evolve_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by Levi's evolution API."""
        kwargs: dict[str, Any] = {
            "budget_evals": self.max_iterations,
        }
        if self.model:
            kwargs["model"] = self.model
        else:
            kwargs["paradigm_model"] = self.paradigm_model
            kwargs["mutation_model"] = self.mutation_model
        if self.budget_dollars is not None:
            kwargs["budget_dollars"] = self.budget_dollars
        if self.budget_seconds is not None:
            kwargs["budget_seconds"] = self.budget_seconds
        if self.output_dir:
            kwargs["output_dir"] = self.output_dir
        if self.pipeline:
            kwargs["pipeline"] = self.pipeline
        if self.behavior:
            kwargs["behavior"] = self.behavior
        if self.cvt:
            kwargs["cvt"] = self.cvt
        if self.init:
            kwargs["init"] = self.init
        if self.meta_advice:
            kwargs["meta_advice"] = self.meta_advice
        if self.punctuated_equilibrium:
            kwargs["punctuated_equilibrium"] = self.punctuated_equilibrium
        if self.prompt_overrides:
            kwargs["prompt_overrides"] = self.prompt_overrides
        if self.cascade:
            kwargs["cascade"] = self.cascade
        return kwargs


class ConfigLoader:
    """Loads repo YAML files into Levi run configuration objects."""

    def load(self, path: Path) -> LeviRunConfig:
        """Load and validate a workflow YAML file."""
        raw_data = load_yaml_document(path)
        return self.from_dict(raw_data)

    def from_dict(self, data: object) -> LeviRunConfig:
        """Validate and normalize a workflow configuration mapping."""
        data = WorkflowFileConfig.model_validate(data).model_dump(
            exclude_none=True,
            exclude_unset=True,
        )
        llm = data.get("llm", {}) or {}
        evaluator = data.get("evaluator", {}) or {}
        pipeline: dict[str, Any] = dict(data.get("pipeline", {}) or {})

        if llm.get("temperature") is not None:
            pipeline["temperature"] = llm["temperature"]
        if llm.get("max_tokens") is not None:
            pipeline["max_tokens"] = llm["max_tokens"]
        if evaluator.get("parallel_evaluations") is not None:
            pipeline["n_eval_processes"] = evaluator["parallel_evaluations"]
        if evaluator.get("timeout") is not None:
            pipeline["eval_timeout"] = evaluator["timeout"]
        cascade: dict[str, Any] = dict(data.get("cascade", {}) or {})
        if evaluator.get("cascade_evaluation") is not None:
            cascade["enabled"] = evaluator["cascade_evaluation"]

        model_override = os.environ.get("LEVI_MODEL")
        default_provider = str(llm.get("default_provider") or "openai")
        primary_model = normalize_model_name(
            model_override or llm.get("primary_model"),
            default_provider=default_provider,
        )
        secondary_model = normalize_model_name(
            model_override or llm.get("secondary_model"),
            default_provider=default_provider,
        )
        default_model = normalize_model_name(
            os.environ.get("LEVI_MODEL", "gpt-4o-mini"),
            default_provider=default_provider,
        )
        search = data.get("search", {}) or {}

        problem = data.get("problem", {}) or {}
        description = _compose_problem_description(data, problem)
        if not description:
            description = (
                "Optimize the candidate_factory implementation for the configured evaluator."
            )

        return LeviRunConfig(
            max_iterations=int(data.get("max_iterations") or 1),
            problem_description=description,
            function_signature=str(
                data.get("function_signature") or "def candidate_factory(*args, **kwargs):"
            ),
            search_seed=int(search.get("seed", 0)),
            paradigm_model=secondary_model or primary_model or default_model,
            mutation_model=primary_model or secondary_model or default_model,
            api_base=llm.get("api_base"),
            api_key_env=llm.get("api_key_env"),
            budget_dollars=data.get("budget_dollars"),
            budget_seconds=data.get("budget_seconds"),
            output_dir=data.get("output_dir"),
            pipeline=pipeline,
            behavior=data.get("behavior", {}) or {"score_keys": ["combined_score"]},
            cvt=data.get("cvt", {}) or {},
            init=data.get("init", {}) or {},
            meta_advice=data.get("meta_advice", {}) or {},
            punctuated_equilibrium=data.get("punctuated_equilibrium", {}) or {},
            prompt_overrides=data.get("prompt_overrides", {}) or {},
            cascade=cascade,
            run_cost=data.get("run_cost", {}) or {},
            raw=data,
        )


def _compose_problem_description(
    data: dict[str, Any],
    problem: dict[str, Any],
) -> str:
    """Build the full Levi prompt from YAML problem and prompt sections."""
    sections: list[str] = []
    base = str(problem.get("description") or data.get("description") or "").strip()
    if base:
        sections.append(base)

    prompt = data.get("prompt", {}) or {}
    system_message = str(prompt.get("system_message") or "").strip()
    if system_message:
        sections.append(system_message)

    objectives = prompt.get("objectives") or ()
    if objectives:
        objective_lines = "\n".join(f"- {objective}" for objective in objectives)
        sections.append(f"Objectives:\n{objective_lines}")

    search = data.get("search", {}) or {}
    notes = str(search.get("notes") or "").strip()
    if notes:
        sections.append(f"Search notes:\n{notes}")

    return "\n\n".join(sections)


class ConfigProvider(Protocol):
    """Abstracts the origin of Levi configuration objects."""

    def load(self, iterations: int) -> LeviRunConfig:
        """Return a resolved configuration for the requested iteration budget."""
        ...

    def describe(self) -> str:
        """Return a human-readable configuration source label."""
        ...


@dataclass(slots=True)
class YamlConfigProvider(ConfigProvider):
    """Loads configuration from YAML using the shared loader."""

    path: Path
    loader: ConfigLoader
    model: str | None = None
    primary_model: str | None = None
    secondary_model: str | None = None
    search_seed: int | None = None
    api_base: str | None = None
    api_key_env: str | None = None

    def load(self, iterations: int) -> LeviRunConfig:
        """Load YAML configuration and apply command-line overrides."""
        config = self.loader.load(self.path)
        config.max_iterations = iterations
        default_provider = str(config.raw.get("llm", {}).get("default_provider") or "openai")
        if self.model:
            model = normalize_model_name(self.model, default_provider=default_provider)
            config.model = model
            config.paradigm_model = None
            config.mutation_model = None
        else:
            if self.primary_model:
                config.mutation_model = normalize_model_name(
                    self.primary_model,
                    default_provider=default_provider,
                )
            if self.secondary_model:
                config.paradigm_model = normalize_model_name(
                    self.secondary_model,
                    default_provider=default_provider,
                )
        if self.search_seed is not None:
            config.search_seed = self.search_seed
        if self.api_base is not None:
            config.api_base = self.api_base
        if self.api_key_env is not None:
            config.api_key_env = self.api_key_env
        return config

    def describe(self) -> str:
        """Return the YAML path used by this provider."""
        return str(self.path)


class MinimalConfigProvider(ConfigProvider):
    """Produces an in-memory Levi configuration without external files."""

    def __init__(
        self,
        problem_description: str = "Optimize the candidate_factory implementation.",
        function_signature: str = "def candidate_factory(*args, **kwargs):",
        model: str | None = None,
        search_seed: int = 0,
        api_base: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self._problem_description = problem_description
        self._function_signature = function_signature
        self._model = normalize_model_name(model or os.environ.get("LEVI_MODEL", "gpt-4o-mini"))
        self._search_seed = search_seed
        self._api_base = api_base
        self._api_key_env = api_key_env

    def load(self, iterations: int) -> LeviRunConfig:
        """Build an in-memory minimal workflow configuration."""
        return LeviRunConfig(
            max_iterations=iterations,
            problem_description=self._problem_description,
            function_signature=self._function_signature,
            search_seed=self._search_seed,
            model=self._model,
            api_base=self._api_base,
            api_key_env=self._api_key_env,
            behavior={"score_keys": ["combined_score"]},
        )

    def describe(self) -> str:
        """Return a label for the in-memory configuration."""
        return "Inline minimal Levi configuration"
