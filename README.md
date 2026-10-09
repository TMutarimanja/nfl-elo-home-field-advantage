# NFL Home-Field Advantage Is Real — and Then It Collapsed

**Logistic regression on 5,805 NFL games (2000–2021) using FiveThirtyEight Elo ratings.**

`R` · `dplyr` · `ggplot2` · `glm` (logistic regression) · `pROC` · R Markdown

![Home-field advantage by season](images/home-field-advantage-by-season.png)

## The question

Does playing at home still help an NFL team win once you control for how good each team is — and has that edge changed over time?

## Key findings

| | Result |
|---|---|
| **Team strength dominates** | A 100-point Elo edge multiplies the odds of winning by **1.70×** |
| **Home field is real** | Between evenly rated teams, the home side won **57.1%** of games (≈54 Elo points, p < 10⁻²³) |
| **…until it wasn't** | Held near **58%** through 2018, then dropped to **51.6% (2019)**, **49.3% (2020, empty stadiums)**, **50.7% (2021)** — likelihood-ratio test p = 0.0018 |
| **Model generalises** | Trained on 2000–2016, tested on 2017–2021: out-of-sample **AUC 0.690**, beating the "always pick the home team" baseline by **9.1 pp** accuracy |
| **Drift shows up as miscalibration** | On held-out seasons the model over-predicts home wins by **3.8 pp** — a symptom of the collapse, not a bug |

**So what?** Any forecasting or betting model calibrated on pre-2019 data will systematically overrate home teams. Home advantage should be re-estimated per season, not treated as a constant.

## What makes this analysis careful

- **Caught and corrected our own specification error.** Our first draft modelled home advantage as a `HomeGame` dummy — but the only comparison group was 61 neutral-site games, so the coefficient didn't measure home advantage at all. The corrected model estimates it from the intercept. The report shows both, side by side.
- **Time-based train/test split** (not random) so the model never sees the future.
- **Checked assumptions**: empirical log-odds are linear in Elo difference.
- **Didn't trust in-sample calibration** — it is guaranteed to look perfect for a logistic model with an intercept; held-out calibration told the real story.

## Read the analysis

📄 **[Full report with code and charts](reports/analysis.md)**

## Repository structure

```
├── nfl-elo-analysis.Rmd     # source: all code + narrative
├── data/nfl_elo.csv         # FiveThirtyEight NFL Elo snapshot (1920–2021)
├── reports/analysis.md      # rendered report (GitHub-readable)
└── images/                  # headline figures
```

## Reproduce

```r
install.packages(c("dplyr","tidyr","readr","ggplot2","scales","pROC",
                   "patchwork","knitr","kableExtra","rmarkdown"))
rmarkdown::render("nfl-elo-analysis.Rmd",
  output_format = rmarkdown::github_document(), output_dir = "reports")
```

## Data

[FiveThirtyEight NFL Elo](https://github.com/fivethirtyeight/data/tree/master/nfl-elo) (CC BY 4.0). FiveThirtyEight stopped updating the series after the 2021 season.

## Authors

Tapiwanashe Mutarimanja and Rodney Chiwanga — Computational Mathematics, M.S. Data Analytics & Visualization, Yeshiva University (Fall 2025).
