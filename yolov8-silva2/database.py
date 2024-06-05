import sqlite3

def init_db():
    """Initialize the SQLite database and create the tourist_counts table if it doesn't exist."""
    conn = sqlite3.connect('tourist_data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tourist_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def save_count_to_db(count):
    """Save the current tourist count to the database."""
    conn = sqlite3.connect('tourist_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO tourist_counts (count) VALUES (?)", (count,))
    conn.commit()
    conn.close()

def get_historical_counts():
    """Retrieve historical tourist counts from the database."""
    conn = sqlite3.connect('tourist_data.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, count FROM tourist_counts ORDER BY timestamp")
    data = c.fetchall()
    conn.close()
    return data

# Initialize the database
init_db()
