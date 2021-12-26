import psycopg2 as psy #pip install psycopg2-binary
import pickle
import numpy as np

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
        self.create_table()

    def create_table(self):
        #create DB table
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

    def insert_data(self, id, image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done):
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
        raw_data_list = self.cursor.fetchall() #sample_indexes_tuple is read from database

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