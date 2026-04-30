# Full Parent-Project Input Bundle

This directory is a copied data bundle from the parent `plasmid-priority`
workspace. It is included so BioSpread can be extracted with the raw/source
material and audit context that produced the geographic-spread branch.

Included subdirectories:

- `raw/`: parent raw input tables and metadata dumps.
- `external/`: external reference resources used by the parent workflow.
- `geo_spread/`: parent geo-spread branch inputs, outputs, inventory, and raw
  archive.
- `silver/`: parent prepared backbone record table.
- `scores/`: parent scored backbone surface.

The standalone BioSpread product workflow does not use the prepared scored
surface as its primary input. It computes its own features and labels from
`data/raw/plasmid_backbones.tsv` plus `data/raw/amr.tsv`.
