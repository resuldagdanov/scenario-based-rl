import psycopg2 as psy
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

        print("\nDB connection is successfully opened!")

    def drop_tables(self):
        self.cursor.execute('''
            DROP TABLE IF EXISTS BUFFER_TABLE;
            DROP TABLE IF EXISTS TRAINING_TABLE;
            DROP TABLE IF EXISTS EVALUATION_TABLE;
            ''')
        print("Tables dropped!")

    def create_training_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS TRAINING_TABLE (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(45),
                total_step_num INTEGER,
                global_episode_number INTEGER,
                best_reward FLOAT8,
                latest_sample_id INTEGER,
                best_reward_episode_number INTEGER,
                is_cpu BOOLEAN,
                debug BOOLEAN,
                n_actions INTEGER,
                state_size INTEGER,
                random_seed INTEGER,
                buffer_size INTEGER,
                lrpolicy FLOAT8,
                lrvalue FLOAT8,
                tau FLOAT8,
                alpha FLOAT8,
                gamma FLOAT8,
                batch_size INTEGER,
                xml_file VARCHAR(100),
                json_file VARCHAR(100),
                epsilon_max FLOAT8,
                epsilon_decay FLOAT8,
                epsilon_min FLOAT8,
                epsilon FLOAT8
                );
            ''')

    def create_evaluation_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS EVALUATION_TABLE (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(45),
                model_episode_number INTEGER,
                total_step_num INTEGER,
                global_episode_number INTEGER,
                average_evaluation_score FLOAT8,
                is_cpu BOOLEAN,
                debug BOOLEAN,
                n_actions INTEGER,
                state_size INTEGER,
                xml_file VARCHAR(100),
                json_file VARCHAR(100)
                );
            ''')

    def create_buffer_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS BUFFER_TABLE (
                model_name VARCHAR(45) NOT NULL,
                id INTEGER NOT NULL,
                image_features BYTEA,
                fused_inputs BYTEA,
                action BYTEA,
                reward FLOAT8,
                next_image_features BYTEA,
                next_fused_inputs BYTEA,
                done INTEGER,
                PRIMARY KEY(model_name, id)
                );
            CREATE INDEX ON BUFFER_TABLE (model_name, id);
            ''')

    def insert_data_to_training_table(self, args, total_step_num=1, global_episode_number=0, best_reward=0.0, latest_sample_id=0, best_reward_episode_number=0):
        insert_command = """
            INSERT INTO TRAINING_TABLE (model_name, is_cpu, debug, n_actions, state_size, random_seed, buffer_size, lrpolicy, lrvalue, tau, alpha, gamma, batch_size, xml_file, json_file, epsilon_max, epsilon_decay, epsilon_min, epsilon, total_step_num, global_episode_number, best_reward, latest_sample_id, best_reward_episode_number)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        self.cursor.execute(insert_command, (args.model_name, args.is_cpu, args.debug, args.n_actions, args.state_size, args.random_seed, args.buffer_size, args.lrpolicy, args.lrvalue, args.tau, args.alpha, args.gamma, args.batch_size, args.xml_file, args.json_file, args.epsilon_max, args.epsilon_decay, args.epsilon_min, args.epsilon_max, total_step_num, global_episode_number, best_reward, latest_sample_id, best_reward_episode_number))

    def insert_data_to_evaluation_table(self, args, total_step_num=1, global_episode_number=0, average_evaluation_score=0.0):
        insert_command = """
            INSERT INTO EVALUATION_TABLE (model_name, is_cpu, debug, n_actions, state_size, xml_file, json_file, model_episode_number, total_step_num, global_episode_number, average_evaluation_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        self.cursor.execute(insert_command, (args.model_name, args.is_cpu, args.debug, args.n_actions, args.state_size, args.xml_file, args.json_file, args.load_episode_number, total_step_num, global_episode_number, average_evaluation_score))

    def get_model_name(self, id):
        self.cursor.execute(
            """
            SELECT
                model_name
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        model_name = self.cursor.fetchone()

        return model_name[0]

    def get_total_step_num(self, id):
        self.cursor.execute(
            """
            SELECT
                total_step_num
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        total_step_num = self.cursor.fetchone()

        return total_step_num[0]

    def get_global_episode_number(self, id):
        self.cursor.execute(
            """
            SELECT
                global_episode_number
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        global_episode_number = self.cursor.fetchone()

        return global_episode_number[0]

    def get_best_reward(self, id):
        self.cursor.execute(
            """
            SELECT
                best_reward
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        best_reward = self.cursor.fetchone()

        return best_reward[0]

    def get_latest_sample_id(self, id):
        self.cursor.execute(
            """
            SELECT
                latest_sample_id
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        latest_sample_id = self.cursor.fetchone()

        return latest_sample_id[0]

    def get_best_reward_episode_number(self, id):
        self.cursor.execute(
            """
            SELECT
                best_reward_episode_number
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        best_reward_episode_number = self.cursor.fetchone()

        return best_reward_episode_number[0]

    def get_is_cpu(self, id):
        self.cursor.execute(
            """
            SELECT
                is_cpu
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        is_cpu = self.cursor.fetchone()

        return is_cpu[0]

    def get_debug(self, id):
        self.cursor.execute(
            """
            SELECT
                debug
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        debug = self.cursor.fetchone()

        return debug[0]

    def get_n_actions(self, id):
        self.cursor.execute(
            """
            SELECT
                n_actions
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        n_actions = self.cursor.fetchone()

        return n_actions[0]
  
    def get_state_size(self, id):
        self.cursor.execute(
            """
            SELECT
                state_size
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        state_size = self.cursor.fetchone()

        return state_size[0]

    def get_random_seed(self, id):
        self.cursor.execute(
            """
            SELECT
                random_seed
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        random_seed = self.cursor.fetchone()

        return random_seed[0]

    def get_buffer_size(self, id):
        self.cursor.execute(
            """
            SELECT
                buffer_size
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        buffer_size = self.cursor.fetchone()

        return buffer_size[0]

    def get_lrpolicy(self, id):
        self.cursor.execute(
            """
            SELECT
                lrpolicy
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        lrpolicy = self.cursor.fetchone()

        return lrpolicy[0]

    def get_lrvalue(self, id):
        self.cursor.execute(
            """
            SELECT
                lrvalue
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        lrvalue = self.cursor.fetchone()

        return lrvalue[0]

    def get_tau(self, id):
        self.cursor.execute(
            """
            SELECT
                tau
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        tau = self.cursor.fetchone()

        return tau[0]

    def get_alpha(self, id):
        self.cursor.execute(
            """
            SELECT
                alpha
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        alpha = self.cursor.fetchone()

        return alpha[0]

    def get_gamma(self, id):
        self.cursor.execute(
            """
            SELECT
                gamma
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        gamma = self.cursor.fetchone()

        return gamma[0]

    def get_batch_size(self, id):
        self.cursor.execute(
            """
            SELECT
                batch_size
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        batch_size = self.cursor.fetchone()

        return batch_size[0]

    def get_epsilon_max(self, id):
        self.cursor.execute(
            """
            SELECT
                epsilon_max
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        epsilon_max = self.cursor.fetchone()

        return epsilon_max[0]

    def get_epsilon_decay(self, id):
        self.cursor.execute(
            """
            SELECT
                epsilon_decay
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        epsilon_decay = self.cursor.fetchone()

        return epsilon_decay[0]

    def get_epsilon_min(self, id):
        self.cursor.execute(
            """
            SELECT
                epsilon_min
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        epsilon_min = self.cursor.fetchone()

        return epsilon_min[0]

    def get_epsilon(self, id):
        self.cursor.execute(
            """
            SELECT
                epsilon
            FROM TRAINING_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        epsilon = self.cursor.fetchone()

        return epsilon[0]

    def get_training_id(self):
        self.cursor.execute(
            """
            SELECT
                id
            FROM TRAINING_TABLE
            ORDER BY id DESC
            """
        )
        id = self.cursor.fetchone() #get_latest id by sorting descending order and getting the first one
        return id[0]

    def get_evaluation_id(self):
        self.cursor.execute(
            """
            SELECT
                id
            FROM EVALUATION_TABLE
            ORDER BY id DESC
            """
        )
        id = self.cursor.fetchone() #get_latest id by sorting descending order and getting the first one
        return id[0]

    def get_evaluation_model_name(self, id):
        self.cursor.execute(
            """
            SELECT
                model_name
            FROM EVALUATION_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        model_name = self.cursor.fetchone()

        return model_name[0]

    def get_evaluation_model_episode_number(self, id):
        self.cursor.execute(
            """
            SELECT
                model_episode_number
            FROM EVALUATION_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        model_episode_number = self.cursor.fetchone()

        return model_episode_number[0]


    def get_evaluation_total_step_num(self, id):
        self.cursor.execute(
            """
            SELECT
                total_step_num
            FROM EVALUATION_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        total_step_num = self.cursor.fetchone()

        return total_step_num[0]

    def get_evaluation_global_episode_number(self, id):
        self.cursor.execute(
            """
            SELECT
                global_episode_number
            FROM EVALUATION_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        global_episode_number = self.cursor.fetchone()

        return global_episode_number[0]

    def get_evaluation_average_evaluation_score(self, id):
        self.cursor.execute(
            """
            SELECT
                average_evaluation_score
            FROM EVALUATION_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        average_evaluation_score = self.cursor.fetchone()

        return average_evaluation_score[0]

    def get_evaluation_is_cpu(self, id):
        self.cursor.execute(
            """
            SELECT
                is_cpu
            FROM EVALUATION_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        is_cpu = self.cursor.fetchone()

        return is_cpu[0]

    def get_evaluation_debug(self, id):
        self.cursor.execute(
            """
            SELECT
                debug
            FROM EVALUATION_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        debug = self.cursor.fetchone()

        return debug[0]

    def get_evaluation_n_actions(self, id):
        self.cursor.execute(
            """
            SELECT
                n_actions
            FROM EVALUATION_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        n_actions = self.cursor.fetchone()

        return n_actions[0]

    def get_evaluation_state_size(self, id):
        self.cursor.execute(
            """
            SELECT
                state_size
            FROM EVALUATION_TABLE
            WHERE id=%s;
            """,
            (id,)
        )
        state_size = self.cursor.fetchone()

        return state_size[0]

    def increment_and_update_global_episode_number(self, id):
        global_episode_number = self.get_global_episode_number(id)
        
        update_command = """
                UPDATE TRAINING_TABLE
                SET global_episode_number = %s
                WHERE id=%s
        """
        self.cursor.execute(update_command, (global_episode_number+1,id))

    def update_total_step_num(self, total_step_num, id):
        update_command = """
                UPDATE TRAINING_TABLE
                SET total_step_num = %s
                WHERE id=%s
        """
        self.cursor.execute(update_command, (total_step_num,id))

    def update_best_reward(self, best_reward, id):
        update_command = """
                UPDATE TRAINING_TABLE
                SET best_reward = %s
                WHERE id=%s
        """
        self.cursor.execute(update_command, (best_reward,id))

    def update_epsilon(self, epsilon, id):
        update_command = """
                UPDATE TRAINING_TABLE
                SET epsilon = %s
                WHERE id=%s
        """
        self.cursor.execute(update_command, (epsilon,id))

    def update_latest_sample_id(self, latest_sample_id, id):
        update_command = """
                UPDATE TRAINING_TABLE
                SET latest_sample_id = %s
                WHERE id=%s
        """
        self.cursor.execute(update_command, (latest_sample_id,id))

    def update_best_reward_episode_number(self, best_reward_episode_number,id):
        update_command = """
                UPDATE TRAINING_TABLE
                SET best_reward_episode_number = %s
                WHERE id=%s
        """
        self.cursor.execute(update_command, (best_reward_episode_number,id))

    def update_evaluation_total_step_num(self, total_step_num, id):
        update_command = """
                UPDATE EVALUATION_TABLE
                SET total_step_num = %s
                WHERE id=%s
        """
        self.cursor.execute(update_command, (total_step_num, id))

    def update_evaluation_average_evaluation_score(self, total_episode_reward, id):
        episode_number = self.get_evaluation_global_episode_number(id)
        if episode_number == 0:
            average_evaluation_score = total_episode_reward
        else:
            average_evaluation_score = self.get_evaluation_average_evaluation_score(id)
            total = (average_evaluation_score * episode_number) + total_episode_reward
            average_evaluation_score = total / (episode_number + 1)

        update_command = """
                UPDATE EVALUATION_TABLE
                SET average_evaluation_score = %s
                WHERE id=%s
        """
        self.cursor.execute(update_command, (average_evaluation_score, id))

    def increment_and_update_evaluation_global_episode_number(self, id):
        global_episode_number = self.get_evaluation_global_episode_number(id)
        
        update_command = """
                UPDATE EVALUATION_TABLE
                SET global_episode_number = %s
                WHERE id=%s;
                """
        self.cursor.execute(update_command, (global_episode_number+1, id))

    def insert_data_to_buffer_table(self, model_name, id, image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done):
        insert_command = """
            INSERT INTO BUFFER_TABLE (model_name, id, image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_name, id) DO UPDATE SET 
                (image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done) 
                = (EXCLUDED.image_features, EXCLUDED.fused_inputs, EXCLUDED.action, EXCLUDED.reward, EXCLUDED.next_image_features, EXCLUDED.next_fused_inputs, EXCLUDED.done);
            """

        #data with 'id' is inserted to DB
        self.cursor.execute(insert_command, 
            (model_name,
            id,
            pickle.dumps(image_features), 
            pickle.dumps(fused_inputs),
            pickle.dumps(action),
            reward,
            pickle.dumps(next_image_features),
            pickle.dumps(next_fused_inputs),
            done
            ))

    def read_batch_data(self, sample_indexes_tuple, batch_size, model_name): #TODO: reading from DB performance can be enhanced by adding new table
        self.cursor.execute(
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
            WHERE id in %s
            and model_name=%s;
            """,
            (sample_indexes_tuple, model_name)
        )
        raw_data_list = self.cursor.fetchall()  #sample_indexes_tuple is read from database

        #TODO: make these values hyperparams
        image_feature_batch = np.empty((batch_size, 1000))
        fused_input_batch = np.empty((batch_size, 3))
        action_batch = np.empty((batch_size, 1)) #this is 2 for SAC and DDPG and 1 for DQN
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
        print("DB connection is successfully closed!\n")