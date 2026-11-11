# README diagrams

The root README embeds a checked-in PNG so it does not depend on a client's
Mermaid support or a third-party rendering service. A scalable SVG is linked
beside it. Both images are generated from the same editable Graphviz DOT source.

From the repository root, with Graphviz and the DejaVu Sans font installed:

```sh
dot -Tsvg docs/diagrams/knowledge-to-backlog.dot -o docs/diagrams/knowledge-to-backlog.svg
dot -Tpng -Gdpi=192 docs/diagrams/knowledge-to-backlog.dot -o docs/diagrams/knowledge-to-backlog.png
```

Regenerate both images after editing the source. Inspect the PNG at README width
and the SVG at full size, then check the PR's rendered README before merging.
Graphviz or font-version changes can alter layout, so a successful command alone
is not a visual check. The SVG must remain self-contained, without scripts,
external resources or HTML labels.

Solid arrows show current engine and task-store interactions. Dashed arrows and
the dashed development box show external development work, not an autonomous
coding service supplied by rumen. The task-store write boundary always requires
explicit reviewer approval; mining and acceptance alone do not promote tasks.
