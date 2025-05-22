import sqlite3


# Function to insert or update a row
def upsert_db(cursor: sqlite3.Cursor, norm, q2_scaled, q2_unscaled, r2_scaled, r2_unscaled, version, entropy_mse, entropy_r2):
    cursor.execute('''
    INSERT INTO nirb_results (norm, Q2_scaled, Q2_unscaled, R2_scaled, R2_unscaled, Version, Entropy_MSE, Entropy_R2)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(norm, Version) DO UPDATE SET
        Q2_scaled = excluded.Q2_scaled,
        Q2_unscaled = excluded.Q2_unscaled,
        R2_scaled = excluded.R2_scaled,
        R2_unscaled = excluded.R2_unscaled,
        Entropy_MSE = excluded.Entropy_MSE,
        Entropy_R2 = excluded.Entropy_R2
    ''', (norm, q2_scaled, q2_unscaled, r2_scaled, r2_unscaled, version, entropy_mse, entropy_r2))
    
    
def create_db(cursor: sqlite3.Cursor):
    # Create the table with needed columns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nirb_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        norm TEXT,
        Q2_scaled FLOAT,
        Q2_unscaled FLOAT,
        R2_scaled FLOAT,
        R2_unscaled FLOAT,
        Version TEXT,
        Entropy_MSE FLOAT,
        Entropy_R2 FLOAT,
        UNIQUE(norm, Version)
    )
    ''')

    # Create unique index on (norm, Version) to enable UPSERT on these columns
    cursor.execute('''
    CREATE UNIQUE INDEX IF NOT EXISTS idx_norm_version ON nirb_results(norm, Version)
    ''')