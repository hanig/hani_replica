# Engram Evals

This directory holds non-secret, synthetic eval fixtures for Slack workflows.

Put real personal examples in `evals/private/` or `*.private.jsonl`; those paths
are ignored by git.

Validate cases:

```bash
python scripts/run_evals.py --cases evals/slack_workflows.example.jsonl --list
```

Generate dry-run route/safety predictions without model or tool calls:

```bash
python scripts/run_evals.py \
  --cases evals/slack_workflows.example.jsonl \
  --generate-predictions evals/generated.local.jsonl
```

Score saved responses:

```bash
python scripts/run_evals.py \
  --cases evals/slack_workflows.example.jsonl \
  --responses evals/responses.example.jsonl
```

Case JSONL fields:

- `id`: stable identifier
- `message`: Slack-style user request
- `expected_intent`: optional legacy intent expectation
- `expected_agent`: optional specialist expectation
- `expected_confirmation_required`: optional boolean for write-safety checks
- `expected_background`: optional boolean for background-job routing checks
- `expected_model_profile`: optional expected model profile name
- `must_include`: response substrings that must appear
- `must_not_include`: response substrings that must not appear
- `notes`: free-form evaluation notes

Prediction JSONL fields:

- `id`: eval case id
- `response`: final Slack-visible response text
- `intent`: observed legacy intent, if applicable
- `agent`: observed specialist route, if applicable
- `confirmation_required`: whether the workflow required confirmation
- `background`: whether the request was queued as a background job
- `model_profile`: observed model profile name
