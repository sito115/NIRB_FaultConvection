import sqlite3

def create_db(cursor: sqlite3.Cursor):
    # Create the table with a UNIQUE constraint directly (index will be implicit)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nirb_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        norm TEXT,
        Version TEXT,
        Accuracy REAL,
        Q2_scaled FLOAT,
        Q2_unscaled FLOAT,
        R2_scaled FLOAT,
        R2_unscaled FLOAT,
        Entropy_MSE_test FLOAT,
        Entropy_R2_test FLOAT,
        Entropy_MSE_train FLOAT,
        Entropy_R2_train FLOAT,
        UNIQUE(norm, Version, Accuracy)  -- Required for ON CONFLICT
    )
    ''')

def upsert_db(cursor: sqlite3.Cursor, norm, q2_scaled, q2_unscaled, r2_scaled, r2_unscaled, version, entropy_mse, entropy_r2, accuracy):
    cursor.execute('''
    INSERT INTO nirb_results (norm, Q2_scaled, Q2_unscaled, R2_scaled, R2_unscaled, Version, Entropy_MSE, Entropy_R2, Accuracy)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(norm, Version, Accuracy) DO UPDATE SET
        Q2_scaled = excluded.Q2_scaled,
        Q2_unscaled = excluded.Q2_unscaled,
        R2_scaled = excluded.R2_scaled,
        R2_unscaled = excluded.R2_unscaled,
        Entropy_MSE = excluded.Entropy_MSE,
        Entropy_R2 = excluded.Entropy_R2,
        Accuracy = excluded.Accuracy
    ''', (norm, q2_scaled, q2_unscaled, r2_scaled, r2_unscaled, version, entropy_mse, entropy_r2, accuracy))
