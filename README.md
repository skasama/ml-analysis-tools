# ML Analysis Tools

Analysis tools for machine-learning-based event reconstruction studies.

This branch contains a new implementation based on a simple, modular
architecture. The existing implementation is kept as a reference and is not
modified or migrated directly.

## Design principles

- Separate data I/O, data handling, transformations, metrics, and plotting.
- Use NumPy arrays as the primary interface between modules.
- Keep task-specific analyses separate.
- Share only genuinely duplicated functionality.
- Prefer simple functions over premature abstractions.
- Use the same library API from scripts, the CLI, and notebooks.
- Keep analysis scripts as reproducible records of individual studies.

## Package structure

```text
src/analysis_tools/
├── io/
├── data/
├── transforms/
├── metrics/
└── plotting/

Additional entry points are organized as follows:
scripts/    Reproducible analysis workflows
tests/      Unit and integration tests
configs/    Dataset and normalization settings

## Plotting API
Plotting functions accept NumPy arrays directly.

They create a new Matplotlib Figure and Axes when ax is not provided, or add data to an existing Axes when ax is provided.

```Python
fig, ax = plot_vertex_error(
    gat_errors,
    label="GAT",
    xlabel="3D vertex error [cm]",
    xlim=(0, 500),
    bins=50,
)

fig, ax = plot_vertex_error(
    cherp_errors,
    ax=ax,
    label="CheRP",
    bins=50,
)

ax.legend()
```

Plotting functions return both the Figure and Axes. Saving, displaying, and closing figures are handled by the calling script, CLI, or notebook.

## Development
'''Shell
uv sync
uv run ruff check .
uv run ruff format .
uv run pytest
'''

The package is under active redesign. Modules and APIs will be added incrementally.

This README.md is written by Copilot to summarize development policy.