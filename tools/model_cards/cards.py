"""Scaffold and check the model cards in ``frontend/public/models/``.

``scaffold`` writes an empty card for each model in ``predictions.json`` that has none.
It is offline: it only fills in what the eval ID gives away, and leaves every other
field null for a person or agent to fill in (see README.md next to this file).

``check`` validates every card against the frontend's schema and checks that each
model in ``predictions.json`` matches exactly one card. The frontend build silently
drops a card that fails its schema, so run this before publishing.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated
from typing import Literal
from typing import Optional

import typer
import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "frontend" / "public" / "models"
PREDICTIONS = REPO_ROOT / "frontend" / "public" / "data" / "predictions.json"

app = typer.Typer(
    help="Scaffold and check the model cards in frontend/public/models/. See README.md.",
    no_args_is_help=True,
)


# Mirrors ModelSchema in frontend/src/lib/data.ts; keep the two in sync.
class _Strict(BaseModel):
    model_config = ConfigDict(strict=True)


Availability = Literal["open-source", "open-weights", "saas-chat", "api"]
Modality = Literal["text", "image", "video", "audio"]
Capability = Literal["reasoning", "coding", "multilingual", "math", "tools"]


class Modalities(_Strict):
    input: Optional[list[Modality]] = None
    output: Optional[list[Modality]] = None


class Specs(_Strict):
    architecture: Optional[str] = None
    knowledge_cutoff: Optional[str] = None
    total_params: Optional[float] = None
    active_params: Optional[float] = None
    context: Optional[float] = None
    modalities: Optional[Modalities] = None
    capabilities: Optional[list[Capability]] = None


class Source(_Strict):
    title: Optional[str] = None
    url: str
    type: Optional[str] = None


class Training(_Strict):
    tokens: Optional[float] = None
    compute_flops: Optional[float] = None
    energy_kwh: Optional[float] = None
    emissions_co2: Optional[float] = None
    training_dataset: Optional[str] = None
    base_model: Optional[str] = None


class Stats(_Strict):
    inference_efficiency_tokens_per_kwh: Optional[float] = None
    throughput_tokens_per_second: Optional[float] = None
    latency_ms_per_token: Optional[float] = None


class Region(_Strict):
    region_deployable: Optional[list[Optional[str]]] = None
    region_developed: Optional[list[Optional[str]]] = None


class ModelCard(_Strict):
    model_config = ConfigDict(strict=True, extra="allow")

    name: str
    release_date: Optional[str] = None
    open_weights: Optional[bool] = None
    organization: Optional[str] = None
    license: Optional[str] = None
    availability: Optional[list[Availability]] = None
    description: Optional[str] = None
    specs: Optional[Specs] = None
    sources: Optional[list[Source]] = None
    training: Optional[Training] = None
    stats: Optional[Stats] = None
    region: Optional[Region] = None
    evals: Optional[list[str]] = None


# Fields a finished card should have; scaffolded cards leave them null. (license may stay null for closed models.)
REQUIRED_TO_PUBLISH = (
    "organization",
    "release_date",
    "open_weights",
    "availability",
    "description",
)
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
GROUPED_NUMBER = re.compile(r"^-?\d{1,3}(_\d{3})+(\.\d+)?$")


class _Yaml12Loader(yaml.SafeLoader):
    """Reads booleans and floats the way js-yaml (YAML 1.2) does in the frontend.

    PyYAML follows YAML 1.1, which reads ``4e23`` as a string and ``yes`` as true.
    """


_Yaml12Loader.yaml_implicit_resolvers = {
    first: [
        (tag, regex)
        for tag, regex in resolvers
        if tag not in {"tag:yaml.org,2002:bool", "tag:yaml.org,2002:float"}
    ]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9][0-9_]*(?:\.[0-9_]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def _normalize_grouped_numbers(value):
    """Read ``128_000`` as a number, as the frontend does."""
    if isinstance(value, str) and GROUPED_NUMBER.match(value):
        return float(value.replace("_", ""))
    if isinstance(value, list):
        return [_normalize_grouped_numbers(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_grouped_numbers(v) for k, v in value.items()}
    return value


def _prediction_ids(predictions: Path) -> list[str]:
    return list(json.loads(predictions.read_text())["models"])


def _load_cards(models_dir: Path) -> dict[str, object]:
    """Slug -> parsed YAML (or the parse error message)."""
    cards: dict[str, object] = {}
    for path in sorted(models_dir.glob("*.yaml")):
        try:
            cards[path.stem] = yaml.load(path.read_text(), Loader=_Yaml12Loader)  # noqa: S506 (safe loader subclass)
        except yaml.YAMLError as error:
            cards[path.stem] = f"YAML parse error: {error}"
    return cards


def _owners(cards: dict[str, object]) -> dict[str, list[str]]:
    """Eval ID -> slugs of the cards that list it."""
    owners: dict[str, list[str]] = defaultdict(list)
    for slug, card in cards.items():
        if isinstance(card, dict):
            for eval_id in card.get("evals") or []:
                owners[eval_id].append(slug)
    return owners


def default_slug(eval_id: str) -> str:
    """``openrouter/tencent/hy3`` -> ``hy3``."""
    return re.sub(r"[^a-z0-9.]+", "-", eval_id.rsplit("/", 1)[-1].lower()).strip("-.")


def _sources(eval_id: str) -> list[tuple[str, str, str]]:
    """Source links that follow from the eval ID alone. Check them when filling in the card."""
    parts = eval_id.split("/")
    if parts[0] in {"openrouter", "openrouter-batch"} and len(parts) >= 3:
        model = "/".join(parts[1:]).split(":", 1)[0]
        return [
            (
                "OpenRouter Model Page",
                f"https://openrouter.ai/{model}",
                "model-homepage",
            )
        ]
    if parts[:2] == ["openai-api", "vllm"] and len(parts) == 4:
        return [
            (
                "Hugging Face Model Card",
                f"https://huggingface.co/{parts[2]}/{parts[3]}",
                "model-weights",
            )
        ]
    return []


def render_card(eval_id: str) -> str:
    """An empty card in the house format: every unknown value is null."""
    q = json.dumps  # a JSON string is a valid double-quoted YAML string
    sources = "".join(
        f"  - title: {q(title)}\n    url: {q(url)}\n    type: {q(kind)}\n"
        for title, url, kind in _sources(eval_id)
    )
    return f"""name: {q(eval_id.rsplit("/", 1)[-1])}
release_date: null
open_weights: null
organization: null
license: null

availability: null

description: null

specs:
  architecture: null
  knowledge_cutoff: null
  total_params: null
  active_params: null
  context: null

  modalities:
    input:
      - "text"
    output:
      - "text"

  capabilities: null

sources:{" []" if not sources else ""}
{sources}
training:
  tokens: null
  compute_flops: null
  energy_kwh: null
  emissions_co2: null
  training_dataset: null
  base_model: null

stats:
  inference_efficiency_tokens_per_kwh: null
  throughput_tokens_per_second: null
  latency_ms_per_token: null

region:
  region_deployable:
    - null
  region_developed: null

evals:
  - {q(eval_id)}
"""


def _display(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


@app.command()
def scaffold(
    eval_ids: Annotated[
        Optional[list[str]],
        typer.Argument(
            help="Eval IDs as in predictions.json. Default: every model there without a card."
        ),
    ] = None,
    slug: Annotated[
        Optional[str],
        typer.Option(
            help="File name (without .yaml) when scaffolding a single eval ID."
        ),
    ] = None,
    models_dir: Annotated[
        Path, typer.Option(help="Folder of model cards.")
    ] = MODELS_DIR,
    predictions: Annotated[
        Path, typer.Option(help="predictions.json written by export-web.")
    ] = PREDICTIONS,
):
    """Write an empty card for each model that has none. Never overwrites a file."""
    owners = _owners(_load_cards(models_dir))
    if not eval_ids:
        eval_ids = [m for m in _prediction_ids(predictions) if m not in owners]
        if not eval_ids:
            print("Every model in predictions.json already has a card.")
            return
    if slug and len(eval_ids) != 1:
        raise typer.BadParameter("--slug needs exactly one eval ID.")

    failed = False
    for eval_id in eval_ids:
        if eval_id in owners:
            print(f"skip   {eval_id}: already in {', '.join(owners[eval_id])}.yaml")
            continue
        name = slug or default_slug(eval_id)
        path = models_dir / f"{name}.yaml"
        if not SLUG_PATTERN.match(name):
            print(
                f"error  {eval_id}: '{name}' is not a valid slug (lowercase letters, digits, '.', '-')."
            )
            failed = True
        elif path.exists():
            print(
                f"error  {eval_id}: {path.name} exists for another model. Pick a name with --slug."
            )
            failed = True
        else:
            path.write_text(render_card(eval_id))
            print(f"wrote  {_display(path)}")
    if failed:
        raise typer.Exit(1)


@app.command()
def check(
    models_dir: Annotated[
        Path, typer.Option(help="Folder of model cards.")
    ] = MODELS_DIR,
    predictions: Annotated[
        Path, typer.Option(help="predictions.json written by export-web.")
    ] = PREDICTIONS,
):
    """Validate every card and its match to predictions.json. Exits 1 on any error."""
    cards = _load_cards(models_dir)
    model_ids = _prediction_ids(predictions)
    errors: list[str] = []
    warnings: list[str] = []

    for slug, card in cards.items():
        if not SLUG_PATTERN.match(slug):
            errors.append(
                f"{slug}: file name must be lowercase letters, digits, '.' and '-' (it is the page URL)"
            )
        if isinstance(card, str):
            errors.append(f"{slug}: {card}")
            continue
        if not isinstance(card, dict):
            errors.append(f"{slug}: not a YAML mapping")
            continue
        try:
            parsed = ModelCard.model_validate(_normalize_grouped_numbers(card))
        except ValidationError as error:
            for issue in error.errors():
                is_date = hasattr(issue["input"], "isoformat")
                hint = " (put dates in quotes)" if is_date else ""
                errors.append(
                    f"{slug}: [{'.'.join(map(str, issue['loc']))}] {issue['msg']}{hint}"
                )
            continue

        missing = [
            field for field in REQUIRED_TO_PUBLISH if getattr(parsed, field) is None
        ]
        if missing:
            warnings.append(f"{slug}: not filled in yet ({', '.join(missing)})")
        availability = parsed.availability or []
        if parsed.open_weights is not None and availability:
            if parsed.open_weights != ("open-weights" in availability):
                warnings.append(
                    f"{slug}: open_weights is {parsed.open_weights} but availability disagrees"
                )
        if "open-source" in availability and "open-weights" not in availability:
            warnings.append(f"{slug}: open-source implies open-weights; list both")

    owners = _owners(cards)
    for model_id in model_ids:
        slugs = owners.get(model_id, [])
        if not slugs:
            errors.append(f"{model_id}: no card lists it (run `scaffold`)")
        elif len(slugs) > 1:
            errors.append(f"{model_id}: listed by several cards ({', '.join(slugs)})")
    for eval_id in sorted(set(owners) - set(model_ids)):
        warnings.append(
            f"{', '.join(owners[eval_id])}: eval '{eval_id}' is not in predictions.json"
        )

    for warning in warnings:
        print(f"warning  {warning}")
    for error in errors:
        print(f"error    {error}")
    print(
        f"{len(cards)} cards, {len(model_ids)} models in predictions.json: "
        f"{len(errors)} errors, {len(warnings)} warnings"
    )
    if errors:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
