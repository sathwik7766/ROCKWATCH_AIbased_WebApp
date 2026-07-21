# Rockfall / Landslide Early-Warning System — Full Web App

Flask website with user authentication, image upload, computer-vision-based
movement detection, and a dashboard with charts + alerts. MySQL via XAMPP.

**Honest framing for interviews:** this does *anomaly detection based on
visual change between frames*, not true geological landslide prediction
(that needs sensor/rainfall/seismic data, which is out of scope here).

## Project structure

```
rockfall_project/
├── app.py                  # Main Flask app (routes, auth, upload logic)
├── config.py                # DB + app config (edit MySQL password here if needed)
├── schema.sql                # Run this in phpMyAdmin to create the database
├── movement_detector.py      # Step 3: frame diff + SSIM + optical flow (tested, works)
├── train_classifier.py       # Step 4: CNN training script (run on Colab)
├── predict.py                 # Step 4: load trained CNN, predict on new image
├── make_test_frames.py        # Regenerates 3 synthetic test images
├── requirements.txt
├── templates/                 # login, register, dashboard, history, alerts
├── static/css/style.css
├── static/uploads/            # uploaded frames get saved here
└── data/
    ├── frames/                # synthetic test images (generate with make_test_frames.py)
    └── dataset/stable/, risk/ # put your CNN training images here
```

## Setup — step by step

### 1. Install XAMPP and start MySQL
Download from apachefriends.org if you don't have it. Open the XAMPP Control
Panel and click **Start** next to both **Apache** and **MySQL**.

### 2. Create the database
Go to `http://localhost/phpmyadmin` in your browser → click the **SQL** tab
→ paste the entire contents of `schema.sql` → click **Go**.
This creates `rockfall_db` with 4 tables: `users`, `frames`, `movement_logs`, `alerts`.

### 3. Set up your Python environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install flask Flask-MySQLdb werkzeug opencv-python numpy pillow
```

> If `Flask-MySQLdb` fails to install on Windows (it needs a C compiler for
> `mysqlclient`), use this alternative instead — install `pymysql` and add
> these two lines at the very top of `app.py`:
> ```python
> import pymysql
> pymysql.install_as_MySQLdb()
> ```
> Then `pip install flask-mysqldb pymysql` should work fine.

### 4. Regenerate test images (if missing)
```bash
python make_test_frames.py
```

### 5. Test the movement detector alone first
```bash
python movement_detector.py data/frames/frame1.jpg data/frames/frame2.jpg
python movement_detector.py data/frames/frame1.jpg data/frames/frame3.jpg
```
Expect combined_score ≈ 0.002 for the stable pair, ≈ 0.29 for the movement pair.

### 6. Run the website
```bash
python app.py
```
Visit `http://127.0.0.1:5000` — you'll be redirected to `/login`.

### 7. Try it end-to-end (18-frame demo sequence — recommended for interviews)

Instead of just 3 static frames, use the full progressive demo:
```bash
python generate_demo_frames.py
```
This creates `data/demo_sequence/frame_01.jpg` through `frame_18.jpg`, simulating
a slope that's stable, then creeps gradually, then shifts sharply — a much
more convincing story than a single before/after pair.

1. Register an account, log in
2. Upload `frame_01.jpg` through `frame_18.jpg` **in order**, one at a time
3. Frames 1-6: dashboard shows "stable" (green), scores ~0.006-0.008
4. Frames 7-12: scores creep up slightly (~0.012-0.014), still stable
5. Frames 13-18: **structured alerts start firing** — scores climb to ~0.32,
   well past the 0.15 threshold
6. Watch the dashboard's Chart.js graph draw this exact rise as you go —
   flat, then a gentle slope, then a sharp spike
7. Check `/alerts` — each one shows the full structured message (see below)

(The original 3-frame `data/frames/` set from `make_test_frames.py` still
works too, for a quick sanity check — just use the 18-frame sequence for
your actual demo/interview walkthrough.)

## Alert format

When an anomaly fires, the alert (shown as a flash message and saved to the
`alerts` table) now looks like this:

```
🔴 HIGH RISK

CNN Prediction : Risk (96%)
Movement Score : 0.31
Threshold      : 0.15

Recommendation:
Inspect the slope immediately.
```

Risk level logic (in `build_alert_message()` in `app.py`):
- **HIGH RISK** — movement score ≥ `HIGH_RISK_THRESHOLD` (0.30, in `config.py`), OR the CNN classifies the frame as "risk"
- **MEDIUM RISK** — movement score ≥ `ANOMALY_THRESHOLD` (0.15) but below high
- The CNN's confidence score is now saved in the database (`confidence_score` column) and shown in the dashboard/history tables next to the classification badge

## How the detection pipeline works (rehearse this for interviews)

Every time you upload a new image, `app.py`:
1. Saves the file to `static/uploads/`
2. Looks up your **most recent previous frame** from the `frames` table
3. Passes both image paths to `combined_movement_score()` in `movement_detector.py`
4. That function computes 3 independent signals and combines them:
   - **Frame differencing** — fast pixel-level change, but noise-sensitive
   - **SSIM** — structural similarity, better at ignoring lighting noise
   - **Optical flow** — estimates actual displacement/motion magnitude
5. If `combined_score >= ANOMALY_THRESHOLD` (0.15, tunable in `config.py`), it's logged as an anomaly and an alert row is created
6. Everything is stored in MySQL and rendered on the dashboard, with a live Chart.js graph of scores over time

## Adding the CNN classifier (Step 4)

The classifier is now fully wired into the website — `app.py` auto-detects
whether `models/slope_classifier.h5` exists. If it doesn't yet, the app runs
fine using movement detection alone. Once you train it, classification
results appear automatically in the dashboard and history tables, and a
"risk" classification also triggers an alert even if the movement score alone
didn't cross the threshold.

### Steps

1. Download a dataset (see below) and place images into:
   ```
   data/dataset/stable/   <- non-landslide / normal slope images
   data/dataset/risk/     <- landslide / at-risk images
   ```
2. Check class balance:
   ```bash
   python check_dataset.py
   ```
   This reports how many images are in each folder. If one class has way
   more images than the other (common — the Bijie dataset has ~2003
   non-landslide vs ~770 landslide), it creates a balanced copy at
   `data/dataset_balanced/` automatically — your originals are untouched.
3. Install TensorFlow: `pip install tensorflow --break-system-packages`
4. Train:
   ```bash
   python train_classifier.py
   ```
   (If `check_dataset.py` created a balanced copy, first edit `DATA_DIR` at
   the top of `train_classifier.py` to `"data/dataset_balanced"`.)
   This saves `models/slope_classifier.h5` and prints final train/val accuracy
   — write those numbers down, you'll want them for interviews.
5. Restart the Flask app (`python app.py`). You should see a startup message
   confirming the classifier loaded. From now on, every upload runs both the
   movement detector AND the CNN, and both results are stored and displayed.

## Dataset sources for the CNN

- **Bijie Landslide Dataset** (recommended — simplest fit) — search
  "Bijie landslide dataset" on Kaggle. Contains 770 landslide + 2003
  non-landslide satellite images, already close to a stable/risk split.
  Ignore the DEM/shapefile/mask files that come with it — only keep the
  plain RGB images.
- **Landslide4Sense** — github.com/iarai/Landslide4Sense-2022 (multi-band
  satellite patches, more complex to work with)
- **CAS Landslide Dataset** — nature.com/articles/s41597-023-02847-z (large-scale)

## Interview talking points to rehearse

- Why 3 signals combined instead of one (each catches different failure modes)
- Why threshold-based alerting instead of pure ML (explainable, no labeled anomaly data needed)
- Why MySQL schema has 4 separate tables (users / frames / movement_logs / alerts) — normalized design, each `movement_log` traceable back to the two frames it compared and any alert it triggered
- Password security: `werkzeug.security.generate_password_hash` — never storing plaintext passwords
- Limitation to volunteer up front: this is CV-based change detection, not a geologically validated prediction model — real deployment would need seismic/rainfall/soil-moisture sensor fusion
