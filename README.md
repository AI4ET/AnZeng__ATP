# AnZengATP: ATP Player Performance Analysis

An end-to-end data analysis pipeline for scraping, processing, and visualizing ATP tennis player performance using custom $\delta$ (delta) and $\alpha$ (alpha) metrics.

This project scrapes player data from the official ATP website, preprocesses it to calculate custom performance scores, and then runs a series of analyses to generate visualizations for:
* Grand Slam final performance (Winners vs. Losers)
* Player career trajectories (grouped by career length)
* Heatmaps of performance distribution
* Individual player performance-over-time scatter plots

---

## Installation

This project relies on several Python libraries. You can install them using `pip`:

```bash
pip install pandas numpy matplotlib seaborn openpyxl scipy tqdm DrissionPage
