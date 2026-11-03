# Recorded activation intervals of ambulance crews, Lublin Voivodeship, 2020 to 2022

Analysis code for: Lupa M, Chuchro M, Zawadzki M, Jadczyk T. *Routine dispatch data for monitoring ambulance crew readiness during epidemics: lessons from the COVID-19 pandemic in a predominantly rural Polish region.* Frontiers in Public Health (under review).

## Data
The record level dispatch orders (SWD PRM exports for 2020, 2021 and 2022) are held by the Lublin Voivodeship Office and are not redistributed here. They can be requested from the Office under a data agreement.
`data_public/pl_cases.csv` contains daily notified COVID-19 cases in Poland from Our World in Data (CC BY 4.0).
Station sites and the team table are published as Supplementary Data S1 of the article.

## Requirements
Python 3.13, pandas, numpy, pyarrow, statsmodels 0.15, scipy, patsy, matplotlib, openpyxl.

## Run order
1. `build.py <out.parquet> 2020.xlsx 2021.xlsx 2022.xlsx` reads the minimal set of columns from the exports
2. `join.py <missions.csv>` adds incident county from the mission file
3. `prep.py` computes the activation interval, exclusions and fleet occupancy
4. `teammap.py` and `prep2.py` link team codes before and after 26 March 2021 and assign station counties and former dispatch areas
5. `desc.py` descriptive statistics
6. `its.py` daily series and initial time series models, `its_final.py` final models with end of phase contrasts
7. `callmodel_week.py` and `callmodel.py` models for individual orders and median regression
8. `boot.py` clustering and moving block bootstrap, `sens.py` wave threshold sensitivity
9. `merger.py` dispatch centre merger analyses, `rev1.py` priority code, start date and exclusion analyses
10. `figs.py` figures, `supp_rev1.py` Supplementary Tables S1 to S14

## Licence
MIT
