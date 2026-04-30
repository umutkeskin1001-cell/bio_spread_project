# BioSpread Problem Statement

## Decision Objective
Rank plasmid backbones by near-term geographic spread risk to guide expert monitoring queues.
Operationally, BioSpread predicts whether a plasmid backbone observed up to
`split_year` will appear in at least two previously unseen countries within
`horizon_years`.

## Primary Users
- Research analysts curating high-risk plasmid watchlists.
- Public-health surveillance teams triaging backlog investigations.

## Model Output Contract
- `risk_probability`: estimated spread risk in the configured horizon.
- `confidence_tier`: `high`, `medium`, or `review` for human triage.
- `explanation`: compact feature-signal summary for review context.

## Decision Policy
- `high`: candidate can be prioritized for immediate review.
- `medium`: candidate stays in standard review queue.
- `review`: insufficient confidence; requires manual domain assessment.

## Non-Goals
- No clinical diagnosis.
- No patient-level decisions.
- No real-time outbreak declaration.
- No autonomous intervention decisions.
- No causal inference claims about transmission mechanisms.

## Claim Boundary
The packaged GeoSpread run is a retrospective benchmark over packaged feature
surfaces. It demonstrates reproducible validation behavior inside this
standalone project; it is not proof of field deployment performance.

## Error Cost Framing
- False negative cost: missed early spread signal.
- False positive cost: analyst time and unnecessary escalation.
- Threshold updates should be tied to explicit operational capacity.
