import psycopg2 as psy
import pickle
import numpy as np
from datetime import datetime

class DB:
    def __init__(self): #TODO: add exceptions for the cases DB not found or not connected
        db_connect_kwargs = {
            'dbname': 'BUFFER_DB',
            'user': 'postgres',
            'password': 'postgres',
            'host': "127.0.0.1", #localhost
            'port': "5432" #default port
        }

        self.connection = psy.connect(**db_connect_kwargs)
        self.connection.set_session(autocommit=True) #each individual SQL statement is treated as a transaction and is automatically committed right after it is executed.
        self.cursor = self.connection.cursor()

        print("DB connection is successfully opened!")

    def get_time_info(self):
        today = datetime.today() # month - date - year
        now = datetime.now() # hours - minutes - seconds

        current_date = str(today.strftime("%b_%d_%Y"))
        current_time = str(now.strftime("%H_%M_%S"))

        # month_date_year-hour_minute_second
        time_info = current_date + "-" + current_time

        return time_info

    def insert_data_to_training_table(self, model_name, total_step_num=1, global_episode_number=0, best_reward=0.0, latest_sample_id=0):
        insert_command = """
            INSERT INTO TRAINING_TABLE (model_name, total_step_num, global_episode_number, best_reward, latest_sample_id)
            VALUES (%s, %s, %s, %s, %s)
            """
        self.cursor.execute(insert_command, (model_name, total_step_num, global_episode_number, best_reward, latest_sample_id))

    def create_training_table(self):
        self.cursor.execute('''
            DROP TABLE IF EXISTS TRAINING_TABLE;
            CREATE TABLE TRAINING_TABLE (
                model_name VARCHAR(45) PRIMARY KEY,
                total_step_num INTEGER,
                global_episode_number INTEGER,
                best_reward FLOAT8,
                latest_sample_id INTEGER
                );
            ''')

    def create_buffer_table(self):
        self.cursor.execute('''
            DROP TABLE IF EXISTS BUFFER_TABLE;
            CREATE TABLE BUFFER_TABLE (
                id INTEGER PRIMARY KEY,
                image_features BYTEA,
                fused_inputs BYTEA,
                action BYTEA,
                reward FLOAT8,
                next_image_features BYTEA,
                next_fused_inputs BYTEA,
                done INTEGER
                );
            CREATE INDEX ON BUFFER_TABLE (id);
            ''')

    def initialize_tables(self):
        #create new model_name and insert it to db
        self.create_training_table()
        model_name = self.get_time_info()
        self.insert_data_to_training_table(model_name)

        self.create_buffer_table()
        print(f"Tables are initialized for training {model_name}!")

    def get_model_name(self):
        self.cursor.execute( #TODO: reading from DB performance can be enhanced by adding new table
            """
            SELECT
                model_name
            FROM TRAINING_TABLE
            """
        )
        model_name = self.cursor.fetchone()

        return model_name[0]

    def get_total_step_num(self):
        self.cursor.execute( #TODO: reading from DB performance can be enhanced by adding new table
            """
            SELECT
                total_step_num
            FROM TRAINING_TABLE
            """
        )
        total_step_num = self.cursor.fetchone()

        return total_step_num[0]

    def get_global_episode_number(self):
        self.cursor.execute( #TODO: reading from DB performance can be enhanced by adding new table
            """
            SELECT
                global_episode_number
            FROM TRAINING_TABLE
            """
        )
        global_episode_number = self.cursor.fetchone()

        return global_episode_number[0]

    def get_best_reward(self):
        self.cursor.execute( #TODO: reading from DB performance can be enhanced by adding new table
            """
            SELECT
                best_reward
            FROM TRAINING_TABLE
            """
        )
        best_reward = self.cursor.fetchone()

        return best_reward[0]

    def get_latest_sample_id(self):
        self.cursor.execute( #TODO: reading from DB performance can be enhanced by adding new table
            """
            SELECT
                latest_sample_id
            FROM TRAINING_TABLE
            """
        )
        latest_sample_id = self.cursor.fetchone()

        return latest_sample_id[0]

    def increment_and_update_global_episode_number(self):
        global_episode_number = self.get_global_episode_number()
        
        update_command = """
                UPDATE TRAINING_TABLE
                SET global_episode_number = %s
        """
        self.cursor.execute(update_command, (global_episode_number+1,))

    def update_total_step_num(self, total_step_num):
        update_command = """
                UPDATE TRAINING_TABLE
                SET total_step_num = %s
        """
        self.cursor.execute(update_command, (total_step_num,))

    def update_best_reward(self, best_reward):
        update_command = """
                UPDATE TRAINING_TABLE
                SET best_reward = %s
        """
        self.cursor.execute(update_command, (best_reward,))

    def update_latest_sample_id(self, latest_sample_id):
        update_command = """
                UPDATE TRAINING_TABLE
                SET latest_sample_id = %s
        """
        self.cursor.execute(update_command, (latest_sample_id,))

    def get_buffer_sample_count(self):
        count_command = """
                SELECT COUNT(*)
                FROM BUFFER_TABLE
        """

        data=[]
        self.cursor.execute(count_command, data)
        result = self.cursor.fetchone()
        count = result[0]
        return count

    def insert_data_to_buffer_table(self, id, image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done):
        insert_command = """
            INSERT INTO BUFFER_TABLE (id, image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET 
                (image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done) 
                = (EXCLUDED.image_features, EXCLUDED.fused_inputs, EXCLUDED.action, EXCLUDED.reward, EXCLUDED.next_image_features, EXCLUDED.next_fused_inputs, EXCLUDED.done);
            """

        #data with 'id' is inserted to DB
        self.cursor.execute(insert_command, 
            (id, 
            pickle.dumps(image_features), 
            pickle.dumps(fused_inputs),
            pickle.dumps(action),
            reward,
            pickle.dumps(next_image_features),
            pickle.dumps(next_fused_inputs),
            done
            ))

    def read_batch_data(self, sample_indexes_tuple, batch_size):
        self.cursor.execute( #TODO: reading from DB performance can be enhanced by adding new table
            """
            SELECT
                image_features, 
                fused_inputs,
                action,
                reward,
                next_image_features,
                next_fused_inputs,
                done
            FROM BUFFER_TABLE
            WHERE id in %s;
            """,
            (sample_indexes_tuple,)
        )
        raw_data_list = self.cursor.fetchall()  #sample_indexes_tuple is read from database

        #TODO: make these values hyperparams
        image_feature_batch = np.empty((batch_size, 1000))
        fused_input_batch = np.empty((batch_size, 3))
        action_batch = np.empty((batch_size, 2))
        reward_batch = np.empty((batch_size))
        next_image_feature_batch = np.empty((batch_size, 1000))
        next_fused_input_batch = np.empty((batch_size, 3))
        done_batch = np.empty((batch_size), dtype=np.int8)

        for i in range(batch_size):
            raw_data = raw_data_list[i]
            image_feature_batch[i] = pickle.loads(raw_data[0])
            fused_input_batch[i] = pickle.loads(raw_data[1])
            action_batch[i] = pickle.loads(raw_data[2])
            reward_batch[i] = raw_data[3]
            next_image_feature_batch[i] = pickle.loads(raw_data[4])
            next_fused_input_batch[i] = pickle.loads(raw_data[5])
            done_batch[i] = raw_data[6]
        
        return image_feature_batch, fused_input_batch, action_batch, reward_batch, next_image_feature_batch, next_fused_input_batch, done_batch

    def close(self):
        self.cursor.close()
        self.connection.close()
        print("DB connection is successfully closed!")