# RAN-NEETI — Unified AI Prediction Lab

A cloud-deployable Streamlit research prototype for historical football and T20 cricket predictions, plus source-linked fictional-character lookup.

## What works
- Dark futuristic dashboard with Automatic and Advanced controls.
- Football: fetches historical English Premier League CSVs from Football-Data.co.uk, builds rolling pre-match form features, trains a Logistic Regression baseline, and calculates chronological holdout metrics.
- Cricket: downloads Cricsheet T20 JSON archive, parses match outcomes, builds rolling team-form features, trains a baseline and calculates chronological holdout metrics.
- Fiction: searches Wikipedia and displays linked character summaries. It deliberately does not invent combat statistics or pretend that fictional matchup probabilities are calibrated.
- Research Evidence tab documents model family, split, metrics, sources, and limitations.

## Deploy online without installing Python locally
1. Download and extract `RAN_NEETI_Website.zip`.
2. Create/sign in to a GitHub account at https://github.com.
3. On GitHub, choose **New repository**, name it `ran-neeti`, and create it.
4. Upload `app.py`, `requirements.txt`, and `README.md` from this folder to the repository root.
5. Open https://share.streamlit.io/ and sign in with GitHub.
6. Click **Create app**, select the `ran-neeti` repository, branch `main`, and main file path `app.py`.
7. Click **Deploy**. Wait for installation; Streamlit will provide a shareable URL.
8. To update the website later, edit/upload files in GitHub; Streamlit will redeploy.

No local Python setup is required. Public hosting and external data availability are subject to the providers' current limits. The first cricket run may be slower because the app downloads and parses a ZIP archive.

## Local preview (optional only)
If Python is already installed: `pip install -r requirements.txt` then `streamlit run app.py`.

## Academic honesty / methodology notes
- Metrics are calculated at runtime from downloaded data and are not hard-coded.
- The chronological holdout uses the latest 20% of rows for evaluation. The deployed model is then refit on all available rows; report the holdout metrics separately.
- For a final thesis, save immutable dated data snapshots, record retrieval dates, verify licenses/citations, run repeated experiments, and document limitations. Do not present the prototype alone as proof of cross-domain generalization.
- Fictional matchup estimates are subjective and require a separate evidence/annotation methodology to become a defensible research module.
