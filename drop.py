import sqlite3

# Define the database path
DB_PATH = 'app/data/database.db'

def drop_table(table_name):
    """
    Drops a specified table from the SQLite database.

    Args:
        table_name (str): The name of the table to drop.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        print(f"Table '{table_name}' dropped successfully.")
    except sqlite3.Error as e:
        print(f"Error dropping table '{table_name}': {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    table_to_drop = input("Enter the table name to drop: ").strip()
    drop_table(table_to_drop)
