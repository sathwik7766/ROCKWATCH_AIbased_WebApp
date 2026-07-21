"""
Main Flask application.
Routes:
  /register       - create account
  /login          - log in
  /logout         - log out
  /dashboard      - upload a new frame, see recent movement scores + chart
  /upload         - POST endpoint: receives image, runs movement_detector,
                     stores result in MySQL, raises alert if anomaly
  /history        - table of all past frames + scores
  /alerts         - table of raised alerts

"""

import os
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from movement_detector import combined_movement_score


CLASSIFIER_AVAILABLE = False
try:
    from predict import predict_image
    import os as _os
    if _os.path.exists("models/slope_classifier.h5"):
        CLASSIFIER_AVAILABLE = True
    else:
        print("[INFO] models/slope_classifier.h5 not found yet -- "
              "run train_classifier.py first to enable CNN classification. "
              "App will run fine without it (movement detection still works).")
except ImportError:
    print("[INFO] TensorFlow not installed -- CNN classification disabled. "
          "Movement detection still works normally.")

app = Flask(__name__)
app.config.from_object(Config)
mysql = MySQL(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ---------- helpers ----------

def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def login_required(view_func):
    """Simple decorator: redirect to login if no user_id in session."""
    from functools import wraps

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


def get_owned_location(cur, location_id, user_id):
    """Fetch a location row, but only if it belongs to the current user.
    Returns None if it doesn't exist or belongs to someone else --
    callers should treat that as a 404/redirect, never trust the URL alone."""
    cur.execute(
        "SELECT * FROM locations WHERE id = %s AND user_id = %s",
        (location_id, user_id),
    )
    return cur.fetchone()


def build_alert_message(combined_score, classification, confidence):
    """
    Build a structured, human-readable alert message.

    Risk level is based on the movement score:
      - >= HIGH_RISK_THRESHOLD  -> HIGH RISK
      - >= ANOMALY_THRESHOLD    -> MEDIUM RISK
    A CNN "risk" classification bumps the level up by one notch, since it's
    an independent signal from a different technique (appearance-based,
    not motion-based).
    """
    threshold = app.config["ANOMALY_THRESHOLD"]
    high_threshold = app.config["HIGH_RISK_THRESHOLD"]

    if combined_score >= high_threshold or classification == "risk":
        level_label = "🔴 HIGH RISK"
        recommendation = "Inspect the slope immediately."
    else:
        level_label = "🟠 MEDIUM RISK"
        recommendation = "Monitor closely and schedule an inspection soon."

    lines = [level_label, ""]

    if classification is not None:
        pct = f"{confidence * 100:.0f}%" if confidence is not None else "n/a"
        lines.append(f"CNN Prediction : {classification.capitalize()} ({pct})")

    lines.append(f"Movement Score : {combined_score:.2f}")
    lines.append(f"Threshold      : {threshold:.2f}")
    lines.append("")
    lines.append(f"Recommendation:\n{recommendation}")

    return "\n".join(lines)



@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)

        cur = mysql.connection.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, password_hash),
            )
            mysql.connection.commit()
        except Exception as e:
            mysql.connection.rollback()
            flash(f"Registration failed: username or email already exists.", "danger")
            return redirect(url_for("register"))
        finally:
            cur.close()

        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("login"))


# ---------- core app routes ----------

# ---------- location management ----------

@app.route("/locations", methods=["GET", "POST"])
@login_required
def locations():
    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Location name is required.", "danger")
        else:
            cur.execute(
                "INSERT INTO locations (user_id, name) VALUES (%s, %s)",
                (user_id, name),
            )
            mysql.connection.commit()
            flash(f"Location '{name}' created.", "success")


    cur.execute(
        """SELECT l.*,
                  (SELECT COUNT(*) FROM frames f WHERE f.location_id = l.id) AS frame_count,
                  (SELECT COUNT(*) FROM alerts a
                     JOIN movement_logs ml ON a.movement_log_id = ml.id
                     JOIN frames f2 ON ml.curr_frame_id = f2.id
                     WHERE f2.location_id = l.id) AS alert_count
           FROM locations l
           WHERE l.user_id = %s
           ORDER BY l.created_at DESC""",
        (user_id,),
    )
    location_list = cur.fetchall()
    cur.close()
    return render_template("locations.html", locations=location_list)



@app.route("/dashboard")
@login_required
def dashboard():
    """No location picked yet -- route to the first one, or prompt to create one."""
    user_id = session["user_id"]
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT id FROM locations WHERE user_id = %s ORDER BY created_at ASC LIMIT 1",
        (user_id,),
    )
    first_location = cur.fetchone()
    cur.close()

    if first_location is None:
        flash("Create a camera location first.", "info")
        return redirect(url_for("locations"))
    return redirect(url_for("location_dashboard", location_id=first_location["id"]))


@app.route("/dashboard/<int:location_id>")
@login_required
def location_dashboard(location_id):
    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    location = get_owned_location(cur, location_id, user_id)
    if location is None:
        flash("Location not found.", "danger")
        cur.close()
        return redirect(url_for("locations"))


    cur.execute(
        """SELECT ml.*, f.filename, f.uploaded_at
           FROM movement_logs ml
           JOIN frames f ON ml.curr_frame_id = f.id
           WHERE f.location_id = %s
           ORDER BY ml.created_at DESC LIMIT 20""",
        (location_id,),
    )
    logs = list(reversed(cur.fetchall()))

    cur.execute(
        """SELECT COUNT(*) AS cnt FROM alerts a
           JOIN movement_logs ml ON a.movement_log_id = ml.id
           JOIN frames f ON ml.curr_frame_id = f.id
           WHERE f.location_id = %s""",
        (location_id,),
    )
    alert_count = cur.fetchone()["cnt"]
    cur.close()

    return render_template("dashboard.html", logs=logs, alert_count=alert_count,
                            threshold=app.config["ANOMALY_THRESHOLD"], location=location)


@app.route("/upload/<int:location_id>", methods=["POST"])
@login_required
def upload(location_id):
    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    location = get_owned_location(cur, location_id, user_id)
    if location is None:
        flash("Location not found.", "danger")
        cur.close()
        return redirect(url_for("locations"))

    if "frame" not in request.files:
        flash("No file selected.", "danger")
        cur.close()
        return redirect(url_for("location_dashboard", location_id=location_id))

    file = request.files["frame"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("Please upload a valid image (png/jpg/jpeg).", "danger")
        cur.close()
        return redirect(url_for("location_dashboard", location_id=location_id))

    filename = secure_filename(f"{user_id}_{location_id}_{int(datetime.now().timestamp())}_{file.filename}")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)


    cur.execute(
        """SELECT f.id, f.filename FROM frames f
           WHERE f.location_id = %s ORDER BY f.uploaded_at DESC LIMIT 1""",
        (location_id,),
    )
    prev_frame = cur.fetchone()

    cur.execute(
        "INSERT INTO frames (user_id, location_id, filename) VALUES (%s, %s, %s)",
        (user_id, location_id, filename),
    )
    mysql.connection.commit()
    curr_frame_id = cur.lastrowid

    if prev_frame is None:
        flash("First frame uploaded for this location. Upload another to start detecting movement.", "info")
        cur.close()
        return redirect(url_for("location_dashboard", location_id=location_id))

    prev_path = os.path.join(app.config["UPLOAD_FOLDER"], prev_frame["filename"])
    curr_path = filepath

    try:
        scores = combined_movement_score(prev_path, curr_path)
    except Exception as e:
        flash(f"Could not process image: {e}", "danger")
        cur.close()
        return redirect(url_for("location_dashboard", location_id=location_id))

    is_anomaly = scores["combined_score"] >= app.config["ANOMALY_THRESHOLD"]

    classification = None
    confidence = None
    if CLASSIFIER_AVAILABLE:
        try:
            cnn_result = predict_image(curr_path)
            classification = cnn_result["class"]
            confidence = cnn_result["confidence"]
            if classification == "risk":
                is_anomaly = True
        except Exception as e:
            print(f"[WARN] CNN classification failed: {e}")

    cur.execute(
        """INSERT INTO movement_logs
           (user_id, prev_frame_id, curr_frame_id, frame_diff_score,
            ssim_score, optical_flow_score, combined_score, classification,
            confidence_score, is_anomaly)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            user_id, prev_frame["id"], curr_frame_id,
            scores["frame_diff"], scores["ssim_dissimilarity"],
            scores["optical_flow"], scores["combined_score"], classification,
            confidence, int(is_anomaly),
        ),
    )
    mysql.connection.commit()
    movement_log_id = cur.lastrowid

    if is_anomaly:
        message = build_alert_message(scores["combined_score"], classification, confidence)
        cur.execute(
            "INSERT INTO alerts (user_id, movement_log_id, message) VALUES (%s, %s, %s)",
            (user_id, movement_log_id, message),
        )
        mysql.connection.commit()
        flash(message, "danger")
    else:
        flash(f"Frame processed. Movement score: {scores['combined_score']:.3f} (stable)", "success")

    cur.close()
    return redirect(url_for("location_dashboard", location_id=location_id))


@app.route("/history/<int:location_id>")
@login_required
def history(location_id):
    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    location = get_owned_location(cur, location_id, user_id)
    if location is None:
        flash("Location not found.", "danger")
        cur.close()
        return redirect(url_for("locations"))

    cur.execute(
        """SELECT ml.*, f.filename, f.uploaded_at
           FROM movement_logs ml
           JOIN frames f ON ml.curr_frame_id = f.id
           WHERE f.location_id = %s
           ORDER BY ml.created_at DESC""",
        (location_id,),
    )
    logs = cur.fetchall()
    cur.close()
    return render_template("history.html", logs=logs, location=location)


@app.route("/alerts/<int:location_id>")
@login_required
def alerts(location_id):
    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    location = get_owned_location(cur, location_id, user_id)
    if location is None:
        flash("Location not found.", "danger")
        cur.close()
        return redirect(url_for("locations"))

    cur.execute(
        """SELECT a.*, ml.combined_score FROM alerts a
           JOIN movement_logs ml ON a.movement_log_id = ml.id
           JOIN frames f ON ml.curr_frame_id = f.id
           WHERE f.location_id = %s ORDER BY a.created_at DESC""",
        (location_id,),
    )
    alert_list = cur.fetchall()
    cur.close()
    return render_template("alerts.html", alerts=alert_list, location=location)


if __name__ == "__main__":
    app.run(debug=True)
