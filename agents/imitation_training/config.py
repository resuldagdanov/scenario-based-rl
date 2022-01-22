from datetime import datetime

today = datetime.today() # month - date - year
now = datetime.now() # hours - minutes - seconds

current_date = str(today.strftime("%b_%d_%Y"))
current_time = str(now.strftime("%H_%M_%S"))

# month_date_year-hour_minute_second
time_info = current_date + "-" + current_time

# path of main repository
base_path = "/home/resul/Research/Codes/Carla/scenario-based-rl" # Local
# base_path = "/cta/eatron/CarlaChallenge/ea202101001_platooning_demo/carla_ws" # WorkStation

best_model_date = "Nov_07_2021-15_32_09"
best_model_name = "epoch_290.pth"

# whether to train a pre-trained model
pretrained = False

# whether to use aggregated data (DAgger)
use_dagger_data = False

# network output is whether brake or throttle-steer. note that to use 'action_classifier', some comments has to be uncommented
type_of_training = "brake_classifier"

# location of all trained and saved models
model_save_path = base_path + "/checkpoint/results/" + time_info + "/"

# latest trained model
trained_model_path = base_path + "/checkpoint/" + best_model_date + "/" + type_of_training + "/" + best_model_name

# dataset for imitation learning (training)
training_data_path = base_path + "/checkpoint/dataset/test/" # Local
# training_data_path = "/cta/eatron/CarlaChallenge/TransFuser/data/14_weathers_data/" # WorkStation

# dataset for imitation learning (validation). will be used also for training dagger data
validation_data_path = base_path + "/checkpoint/dataset/test/" # Local
# validation_data_path = "/cta/eatron/CarlaChallenge/TransFuser/data/14_weathers_data/" # WorkStation

# aggregated data path. will be concatenated with training dataset
dagger_data_path = base_path + "/dataset/dagger/"

# training hyperparameters
batch_size = 32
learning_rate = 1e-2
momentum = 0.9
gamma = 0.1
min_milestone = 10
max_milestone = 20
max_number_of_epochs = 300
loss_print_interval = 32

# model saving frequency
save_every_n_epoch = 10

# test on validation set every 10 epoch
validate_per_n = 10

train_towns = ["town05_tiny"] # Local
# train_towns = [ # WorkStation
#         "Town01_long", "Town01_short", "Town01_tiny", \
#         "Town02_long", "Town02_short", "Town02_tiny", \
#         "Town03_long", "Town03_short", "Town03_tiny", \
#         "Town04_long", "Town04_short", "Town04_tiny", \
#                         "Town05_short", "Town05_tiny", \
#         "Town06_long", "Town06_short", "Town06_tiny", \
#                         "Town07_short", "Town07_tiny",\
#                         "Town10_short", "Town10_tiny"]

validation_towns = ["town01_tiny"] # Local
# validation_towns = ["Town05_long"] # WorkStation
