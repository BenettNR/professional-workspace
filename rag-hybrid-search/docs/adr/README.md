# Architecture Decision Records

This directory captures the non-obvious decisions that shape this service. Each ADR follows the [Michael Nygard template](https://github.com/joelparkerhenderson/architecture-decision-record/blob/main/locales/en/templates/decision-record-template-by-michael-nygard/index.md): **Context → Decision → Consequences → Alternatives**.

Why ADRs over code comments: the *why* of a non-trivial decision belongs somewhere a reviewer can find without grepping. Comments rot when code moves; ADRs are stable artifacts that survive refactors.

## Index

| ADR | Title | Status |
|---|---|---|
| [001](001-hybrid-retrieval.md) | Hybrid Retrieval (Dense + Sparse + Rerank) | Accepted |
| [002](002-citation-verification.md) | LLM-as-Judge for Citation Verification | Accepted |
| [003](003-provider-abstraction.md) | Provider Abstraction and Offline Demo Mode | Accepted |

## When to write a new ADR

- A reviewer's likely first question would be *"why this way and not the obvious alternative?"*
- The decision touches the public API, the deployment story, or a system invariant.
- You ruled out one or more alternatives that future-you might re-propose without remembering they were considered.

## When *not* to write an ADR

- Trivial implementation choices that a code reader can infer from the code itself.
- Decisions that follow directly from an existing ADR.
- Style or formatting preferences (those belong in `pyproject.toml`, not prose).
