# Contributing Guidelines Plan
**Created:** 2025-11-27 09:00:00
**Last Modified:** 2025-11-27 09:00:00
**By:** gemini-3-pro-preview

## Goal
Add concise contribution guidelines to the repository to encourage community involvement in presets, filters, bug fixes, and performance improvements.

## Implementation Steps
1.  Create `CONTRIBUTING.md` in the repository root.
2.  Draft content focusing on:
    -   **Presets:** Add YAML files to `presets/`.
    -   **Filters:** Add concise filter implementations to `ultra_post/ops/`.
    -   **General:** Bug fixes and performance improvements are welcome.
    -   **Style:** Emphasize concise, "academic code golf" style.

## Proposed Content (Draft)

```markdown
# Contributing

We welcome contributions that keep this codebase concise and expressive.

## Ways to Contribute

1.  **Presets**: Create a processing pipeline preset (YAML) and save it to `presets/`.
2.  **Filters**:
    -   Add your filter implementation to `ultra_post/ops/`.
    -   Register it in `ultra_post/ops/registry.py` (add to `OPS` and `DEFAULT_PARAMS`).
3.  **Improvements**: Bug fixes and performance optimizations are highly encouraged.
    -   Our goal is to have code that is easy to understand and highly performant.

## Code Style

-   **Concise & Approachable**: We prioritize readability so new users can quickly understand the post-processing logic.
-   **Performant**: We aim for code that is both expressive and performant, enabling real-time processing.
```

