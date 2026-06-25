# Tennis datasets — archive of Jeff Sackmann's data

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This repository is an **archival mirror of the public tennis datasets compiled by Jeff Sackmann**.
It exists so the data remains available and citable. It contains **only data and its documentation** —
no models, analysis, or derived code of any kind.

The same archive is mirrored on the Hugging Face Hub:
**https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive**

## Contents

| Folder | What it is | Files | Coverage |
| --- | --- | --- | --- |
| `slam_pointbypoint/` | Point-by-point logs for the four Grand Slams (one `*-points.csv` and one `*-matches.csv` per slam-year), plus `data_dictionary.txt` | 166 CSVs | 2011–2024, Australian Open / Roland Garros / Wimbledon / US Open |
| `atp/` | Men's tour: match results, rankings, and player tables going back decades, plus `matches_data_dictionary.txt` | 174 CSVs | through 2026 |
| `wta/` | Women's tour: match results, rankings, and player tables | 125 CSVs | through 2026 |

Each folder also keeps Jeff Sackmann's original repository README as `UPSTREAM_README.md` for provenance.

## Original sources

- Grand Slam point-by-point — https://github.com/JeffSackmann/tennis_slam_pointbypoint
- ATP — https://github.com/JeffSackmann/tennis_atp
- WTA — https://github.com/JeffSackmann/tennis_wta

The `slam_pointbypoint` snapshot was taken from upstream commit `6febb77` (October 2024); the `atp`
and `wta` snapshots from upstream commits made in June 2026.

## Attribution and license

All data here was collected and compiled by **Jeff Sackmann** and is redistributed under the same
license he releases it under: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)**. See `LICENSE`.

If you use this data, attribute Jeff Sackmann and link back to the original repositories above.
Per the license, use is **non-commercial**, and any redistribution must carry the same license.
This mirror claims no ownership over the data and adds no rights to it.
