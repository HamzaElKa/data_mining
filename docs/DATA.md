# Raw Dataset

`flickr_data2.csv` (~89 MB, 252k photo records) is **not tracked in this repository** — it's too large for git and is excluded via `.gitignore`.

## How to get it

The dataset was collected via a custom Flickr API scraper as part of the project's data collection phase (see [`subject/sujet.txt`](../subject/sujet.txt) for the assignment brief describing the collection method). It isn't publicly redistributed here.

To reproduce the pipeline:
1. Contact the author (see repo owner) for access to the raw CSV, **or**
2. Re-run your own Flickr API scraper following the methodology described in `subject/sujet.txt` (geo-located photos around Lyon, France).

Once obtained, place `flickr_data2.csv` at the repository root before running `src/main.py` or any of the pipeline scripts.
