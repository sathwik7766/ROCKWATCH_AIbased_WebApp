-- schema.sql
-- Run this in phpMyAdmin (XAMPP) or via mysql CLI to set up the database.
--
-- In phpMyAdmin: open http://localhost/phpmyadmin, click "SQL" tab,
-- paste this whole file, click "Go".

CREATE DATABASE IF NOT EXISTS rockfall_db;
USE rockfall_db;

-- Registered users
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Camera locations -- each user can monitor multiple slopes/sites.
-- Frames are compared only against the previous frame from the SAME
-- location, so each location gets its own independent movement history.
CREATE TABLE IF NOT EXISTS locations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Every image a user uploads (simulating a camera frame arriving)
CREATE TABLE IF NOT EXISTS frames (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    location_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
);

-- Result of comparing a new frame against the previous one
CREATE TABLE IF NOT EXISTS movement_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    prev_frame_id INT,
    curr_frame_id INT NOT NULL,
    frame_diff_score FLOAT,
    ssim_score FLOAT,
    optical_flow_score FLOAT,
    combined_score FLOAT,
    classification VARCHAR(20),      -- 'stable' or 'risk' (from CNN, if used)
    confidence_score FLOAT,           -- CNN's confidence in that classification (0-1)
    is_anomaly BOOLEAN DEFAULT FALSE, -- combined_score crossed threshold
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (curr_frame_id) REFERENCES frames(id) ON DELETE CASCADE
);

-- Alerts raised when an anomaly is detected
CREATE TABLE IF NOT EXISTS alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    movement_log_id INT NOT NULL,
    message VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (movement_log_id) REFERENCES movement_logs(id) ON DELETE CASCADE
);

-- If you already ran this schema before and just pulled the update, run
-- ONLY this line in phpMyAdmin's SQL tab instead of the whole file again:
-- ALTER TABLE movement_logs ADD COLUMN confidence_score FLOAT AFTER classification;

-- MIGRATION: adding multi-location support to an existing database.
-- Run these lines (in order) in phpMyAdmin's SQL tab if you already have
-- data and just pulled this update -- do NOT re-run the whole file, it
-- would wipe your existing tables.
--
-- CREATE TABLE IF NOT EXISTS locations (
--     id INT AUTO_INCREMENT PRIMARY KEY,
--     user_id INT NOT NULL,
--     name VARCHAR(100) NOT NULL,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
-- );
--
-- ALTER TABLE frames ADD COLUMN location_id INT NULL AFTER user_id;
--
-- INSERT INTO locations (user_id, name)
-- SELECT id, 'Default Location' FROM users;
--
-- UPDATE frames f
-- JOIN locations l ON l.user_id = f.user_id AND l.name = 'Default Location'
-- SET f.location_id = l.id
-- WHERE f.location_id IS NULL;
--
-- ALTER TABLE frames MODIFY location_id INT NOT NULL;
-- ALTER TABLE frames ADD FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE;
