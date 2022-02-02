import psycopg2 as psy
import pickle
import numpy as np
np.random.seed(0)

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

        #print("\nDB connection is successfully opened!")

    def get_reward_table_entries(self, model_name):
        self.cursor.execute(
            """
            SELECT
                reward, episode_num, step_num
            FROM REWARD_TABLE
            WHERE model_name=%s
            ORDER BY episode_num ASC, step_num ASC
            """,
            (model_name,)
        )
        raw_data_list = self.cursor.fetchall()
        count = len(raw_data_list)

        reward_array = np.empty((count))
        episode_num_array = np.empty((count))
        step_num_array = np.empty((count))

        i = 0
        for raw_data in raw_data_list:
            reward_array[i] = raw_data[0]
            episode_num_array[i] = raw_data[1]
            step_num_array[i] = raw_data[2]
            i += 1

        return reward_array, episode_num_array, step_num_array

    def close(self):
        self.cursor.close()
        self.connection.close()
        #print("DB connection is successfully closed!\n")


model_name = "Jan_30_2022-18_52_34"

db = DB()
reward_array, episode_num_array, step_num_array = db.get_reward_table_entries(model_name)

count = reward_array.shape[0]
for i in range(count):
    print(f"{episode_num_array[i]} {step_num_array[i]} {reward_array[i]}")
db.close()
