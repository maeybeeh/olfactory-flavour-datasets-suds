OpenPOM Olfactory–Flavour Dataset Analysis
A research‑driven repository exploring preprocessing, descriptor normalization, activity‑cliff detection, and OTDD (Optimal 
Transport Dataset Distance) analysis across three major olfactory datasets: OpenPOM, Goodscent‑Leffingwell, and FlavorDB.

 Overview
This repository contains all preprocessing scripts, notebooks, processed datasets, and analytical outputs used to evaluate the 
structure, consistency, and cross‑dataset alignment of OpenPOM.

The project focuses on:

cleaning and standardizing molecular SMILES

normalizing odor descriptors

detecting activity cliffs

computing OTDD distances between datasets

producing reproducible figures and a full datacard

All results are stored in a structured, research‑ready format.

Repository Structure

olfactory-flavour-datasets-suds/
│
├── data/
│   ├── raw/                  
│   ├── processed/   set
│   └── external/    ces
│
├── notebooks/               
│
├── src/
│   ├── preprocessingning
│   ├── analysis/    pts
│   └── visualizationies
│
├── reports/
│   ├── figures/  res
│   └── datacard.mdion
│
└── README.md

 Processed Datasets
OpenPOM (processed)
Located at:
data/processed/openpom_processed.csv
Includes:

canonicalized SMILES

cleaned + normalized descriptors

deduplicated molecules

descriptor grouping (citrus, woody, floral, etc.)

External Analysis Outputs
Stored in:
data/external/
Includes:

openpom_activity_cliffs.csv

full_otdd_matrix.csv

Activity Cliffs
Activity cliffs highlight molecules that are structurally similar but have different odor descriptors.

Computed using:

Tanimoto similarity

Jaccard similarity

descriptor disagreement

OTDD (Optimal Transport Dataset Distance)
OTDD quantifies how different two datasets are in terms of:

molecular distribution

descriptor distribution

representation geometry

Your OTDD matrix compares:

OpenPOM

Goodscent‑Leffingwell

FlavorDB

Figures
All figures are stored in:

reports/figures/

Datacard
Full dataset documentation is available in:

reports/datacard.md
 
️ Reproducibility

To reproduce results:

Install dependencies

Run preprocessing scripts

Execute notebooks

Generate figures

Review datacard

 Citation
If you use this repository, cite:

OpenPOM

Leffingwell

FlavorDB

OTDD methodology (Alvarez‑Melis & Fusi, 2020)
 

