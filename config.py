"""
config.py
---------
XAMPP's default MySQL setup: host=localhost, user=root, no password,
running on the default port 3306. Change MYSQL_PASSWORD if you set one.
"""

import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-this")

    MYSQL_HOST = "localhost"
    MYSQL_USER = "root"
    MYSQL_PASSWORD = ""          # default XAMPP MySQL has no root password
    MYSQL_DB = "rockfall_db"
    MYSQL_CURSORCLASS = "DictCursor"

    UPLOAD_FOLDER = "static/uploads"
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    # Movement score bands: below LOW = stable, LOW-MEDIUM = medium risk,
    # above MEDIUM = high risk. Tune these after testing on your own frames.
    ANOMALY_THRESHOLD = 0.15      # at/above this = at least "medium risk"
    HIGH_RISK_THRESHOLD = 0.30    # at/above this = "high risk"
