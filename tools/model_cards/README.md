# Model cards

Each model on the website has a YAML card in `frontend/public/models/<slug>.yaml`. The slug (file name) is the page URL, `/models/<slug>`. A card is joined to its results through `evals`: every model ID in `frontend/public/data/predictions.json` must be listed by exactly one card. The schema is `ModelSchema` in `frontend/src/lib/data.ts`; `cards.py` mirrors it, so change both together.

The frontend build does not fail on a bad card. It logs `❌ Zod Validation Error` and shows the model without metadata. Run `check` instead of relying on the build.

## Workflow

After `complai export-web` adds new models:

```bash
.venv/bin/python tools/model_cards/cards.py scaffold      # empty card for every model without one
# fill in the new cards (by hand, or ask an agent to follow the instructions below)
.venv/bin/python tools/model_cards/cards.py check         # exits 1 on errors
cd frontend && npm run build                               # then confirm no "Zod Validation Error"
```

- `scaffold` is offline. It writes the eval ID, a name taken from the ID, text-in/text-out modalities and, for OpenRouter or vLLM IDs, the OpenRouter or Hugging Face page as the first source. Everything else is null. It never overwrites a file and skips eval IDs that a card already lists.
- `scaffold ID [ID ...]` scaffolds only those IDs; `--slug NAME` picks the file name for a single ID. The default slug is the last part of the ID, lowercased (`openrouter/mistralai/mistral-large-4-0` → `mistral-large-4-0`); rename it when the ID carries a provider suffix or version noise.
- To add another run of a model that already has a card (for example a vLLM run next to an OpenRouter one), add its ID to that card's `evals` instead of scaffolding.
- `check` errors: a card that does not parse or fails the schema (wrong type, unknown enum value, unquoted date), a bad file name, a model in `predictions.json` with no card, or a model listed by several cards. Warnings: a card with required fields still null, `open_weights` disagreeing with `availability`, `open-source` without `open-weights`, and card evals that are not in `predictions.json` (harmless, but often a typo).

## Instructions for filling in a card

These are written for an agent with web access, but apply equally to a person. Fill one card at a time.

### Ground rules

- **Never guess.** Leave a field null unless a source states it. A null is shown as missing; a wrong number is shown as fact.
- **Report what the developer has announced, not what you can download today.** If the developer announced open weights or a license for a date that has not come yet, record it and say so in the description. Example: Mistral Large 4 is `open_weights: true` with `license: null` because the weights and license were announced for a later date.
- **Prefer primary sources**, in this order: the developer's announcement, tech report or paper; the official Hugging Face model card; the developer's docs or pricing page; the OpenRouter model page (`https://openrouter.ai/<org>/<model>`, or `https://openrouter.ai/api/v1/models` for context length and modalities). Use news articles only to find a primary source.
- **Make sure the card describes the exact model that was evaluated.** The eval ID is the source of truth. A `-preview`, `-flash`, `-think` or dated suffix is a different model from the one without it.
- Keep the field order and layout of the scaffold. Quote strings with double quotes. Write numbers as full integers (`52000000000`, not `52B` or `52e9`).

### Fields

| Field | What to write |
|---|---|
| `name` | The display name the developer uses, e.g. `"Mistral Large 4"`. |
| `release_date` | `"YYYY-MM-DD"`, in quotes. The first day the model was publicly usable (API, chat app or weights), including a public preview. |
| `open_weights` | `true` if the developer released downloadable weights for this model, or officially announced them. Restrictive or non-commercial licenses still count. A model you can only reach through an API is `false`. |
| `organization` | The developer, not the hosting provider (`"Tencent"`, not `"OpenRouter"`). |
| `license` | The weights license as a lowercase slug: the SPDX ID where one exists (`"mit"`, `"apache-2.0"`, `"openmdw-1.1"`), otherwise a short slug of the license name (`"qwen-community-1.0"`, `"gemma"`, `"llama3.3"`). Use `"mit-modified"` for MIT with extra terms. Null for closed models and for licenses not yet published. Add the license text as a `licence-url` source. |
| `availability` | A list in this order, keeping only what applies: `open-source`, `open-weights`, `saas-chat`, `api`. Typical: closed model `[api, saas-chat]`; open model `[open-weights, api]`, plus `saas-chat` if the developer runs a chat app with it. |
| `description` | Two or three sentences of plain English for a non-specialist: who made it, what kind of model it is, its size, what it is meant for, and anything that affects the results (preview status, announced but unreleased weights). No marketing language, no unexplained jargon or abbreviations, no benchmark claims. |
| `specs.architecture` | `"dense decoder-only transformer"` or `"Mixture-of-Experts (MoE)"`; null if not disclosed. |
| `specs.knowledge_cutoff` | `"YYYY-MM"` as the developer states it; null otherwise. |
| `specs.total_params` / `active_params` | Full integers. For an MoE model, total parameters and the parameters active per token. For a dense model, `total_params` only and `active_params: null`. Null for closed models unless the developer states it. |
| `specs.context` | The maximum context window in tokens, as a full integer from the developer or the provider page (`131072`, `1048576`). When the developer says "128K" or "1M" without an exact number, write `128000` or `1000000`. |
| `specs.modalities` | `input` and `output` lists from `text`, `image`, `video`, `audio`. List only what the model itself accepts or produces. |
| `specs.capabilities` | From `reasoning`, `coding`, `multilingual`, `math`, `tools`; only what the developer claims. `reasoning` means a thinking or reasoning mode. `tools` means function or tool calling. |
| `sources` | Every page you used, each with a `title`, `url` and `type`: `model-homepage` (OpenRouter or the developer's model page), `model-weights` (Hugging Face card), `announcement` (blog post or press release), `paper-url` (tech report or paper), `licence-url` (license text). Check that the scaffolded link works. |
| `training`, `stats` | Only figures the developer publishes (usually in a tech report): training tokens, compute in FLOPs, energy in kWh, CO₂ in kg. Otherwise leave them null. |
| `region.region_developed` | ISO 3166-1 alpha-2 code(s) of the developer's headquarters, e.g. `["CH"]`. Use straight quotes. |
| `region.region_deployable` | Leave as `[null]`. |
| `evals` | Leave the scaffolded ID. Add other eval IDs only if they are runs of this exact model. |

### Open source vs open weights

`open-source` follows the [OSI Open Source AI Definition](https://opensource.org/ai/open-source-ai-definition): the weights are under an open license **and** the training code is public **and** the training data is public or described in enough detail to rebuild it (examples: Apertus, OLMo). Publishing weights alone is `open-weights`. A model listed as `open-source` must also list `open-weights` and have `open_weights: true`.

### When you are done

1. `.venv/bin/python tools/model_cards/cards.py check`: no errors, and no "not filled in yet" warning for the cards you filled.
2. `cd frontend && npm run build` and confirm the output has no `Zod Validation Error`.
3. Report, per card, which fields you left null and why, and anything you were unsure about (conflicting sources, announced but unreleased weights).
