import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import yaml
import logging


logging.basicConfig(level=logging.INFO)

file_path = r"D:\final_project\airflow\data\raw\meter_data.csv"

def load_db_config(config_path="config/db_config.yaml"):
    """Load PostgreSQL configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["postgres"]

class DataIngestion:

    def __init__(self, db_config):
        self.db_config = db_config
        logging.info(f"Using PostgreSQL Version: {db_config.get('version', 'Not specified')}")

    def get_db_connection(self):
        """Create and return PostgreSQL connection."""
        return psycopg2.connect(
            host=self.db_config["host"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            port=self.db_config["port"]
        )

    def extract_data(self, file_path):
        """Reads a CSV file and returns a dataframe."""
        logging.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)

        # Basic validation
        required_cols = ["meter_id", "timestamp", "load_value"]
        if not all(col in df.columns for col in required_cols):
            raise Exception(f"Missing required columns in input data: {required_cols}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df


    def create_table_if_not_exists(self):
        """Create raw_meter_data table if it doesn't exist."""
        create_table_query = """
            CREATE TABLE IF NOT EXISTS raw_meter_data (
                id SERIAL PRIMARY KEY,
                meter_id VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                load_value DOUBLE PRECISION NOT NULL,
                UNIQUE (meter_id, timestamp)
            );
            """
        conn = self.get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute(create_table_query)
            conn.commit()
            logging.info("Table 'raw_meter_data' ensured (created if not existed).")
        except Exception as e:
            conn.rollback()
            logging.error(f"Error creating table: {e}")
            raise

    def load_to_postgres(self, df):
        """Insert dataframe rows into PostgreSQL raw_meter_data table."""
        insert_query = """
            INSERT INTO raw_meter_data (meter_id, timestamp, load_value)
            VALUES (%s, %s, %s)
            ON CONFLICT (meter_id, timestamp) DO NOTHING;
        """

        data_tuples = [
            (row["meter_id"], row["timestamp"], row["load_value"])
            for _, row in df.iterrows()
        ]

        conn = self.get_db_connection()
        cur = conn.cursor()

        try:
            execute_batch(cur, insert_query, data_tuples)
            conn.commit()
            logging.info("Data successfully ingested into PostgreSQL 15.")
        except Exception as e:
            conn.rollback()
            logging.error(f"Error inserting data: {e}")
            raise
        finally:
            cur.close()
            conn.close()

    def run_pipeline(self, file_path):
        df = self.extract_data(file_path)
        self.load_to_postgres(df)


if __name__ == "__main__":
    db_config = load_db_config("config/db_config.yaml")
    
    ingestion = DataIngestion(db_config)
    ingestion.create_table_if_not_exists()
    ingestion.run_pipeline(file_path)
    print("Data inserted successfully!")