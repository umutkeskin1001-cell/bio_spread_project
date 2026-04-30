# Packaged Raw Data

This directory contains the data needed to run BioSpread independently from the
parent `plasmid-priority` repository.

Files:

- `plasmid_backbones.tsv`: raw/backbone observation records packaged with the
  project. BioSpread derives its own modeling features and spread labels from
  this file.
- `amr.tsv`: raw AMR gene evidence. BioSpread joins this by sequence accession
  and computes AMR burden features itself.

The original parent repository also contains `data/geo_spread/geo_spread_raw.tar`,
but that archive is about 20 GB and is not required for this standalone product
pipeline. The standalone project packages the raw tabular inputs it needs to
derive features, train, evaluate, select a primary model, and generate reports.
