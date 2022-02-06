import os
import sys

import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from data import DatasetLoader

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from networks.imitation_network import ImitationNetwork


def calculate_loss(brake, gt_control, criterion_brake):
    brake_loss = criterion_brake(brake, gt_control[:, 2].view(-1, 1))
    return brake_loss


def trainer(writer_counter):
    i = 0
    running_loss = 0
    epoch_loss = 0

    # loop through each batch inside the dataset
    for data in tqdm(train_dataset_loader):

        rgb_input = data['image'].float().to(device)
        rgb_input = network.normalize_rgb(rgb_input)
        control_input = data['control'].to(device)
        target_point = torch.stack(data['target_point'], dim=1).to(device)
        speed_list = torch.stack(data['speed_sequence'], dim=1).to(device)

        # do not train dataset with amount of data being less than a batch size
        if speed_list.shape[0] != config.batch_size:
            continue

        speed_list = speed_list.view(-1, config.batch_size, config.speed_input_size).to(device)

        # forward propagation
        dnn_brake = network(front_images=rgb_input, waypoint_input=target_point, speed_sequence=speed_list)

        optimizer.zero_grad()
        total_loss = calculate_loss(brake=dnn_brake, gt_control=control_input, criterion_brake=criterion_brake)
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        if i % config.loss_print_interval == 0:
            print("[%d, %5d] train dataset loss: %.4f" %(epoch + 1, i + 1, running_loss / config.loss_print_interval))
            
            epoch_loss += running_loss
            writer.add_scalar("training-dataset-running-loss", running_loss / config.loss_print_interval, writer_counter)
            writer_counter += 1
            
            running_loss = 0

        i += 1

    return epoch_loss, writer_counter


def validator():
    i = 0
    epoch_loss = 0

    # loop through each batch inside the validation dataset
    for data in tqdm(validation_dataset_loader):

        rgb_input = data['image'].float().to(device)
        rgb_input = network.normalize_rgb(rgb_input)
        control_input = data['control'].to(device)
        target_point = torch.stack(data['target_point'], dim=1).to(device)
        speed_list = torch.stack(data['speed_sequence'], dim=1).to(device)

        if speed_list.shape[0] != config.batch_size:
            continue

        speed_list = speed_list.view(-1, config.batch_size, config.speed_input_size).to(device)

        dnn_brake = network(front_images=rgb_input, waypoint_input=target_point, speed_sequence=speed_list)

        optimizer.zero_grad()
        total_loss = calculate_loss(brake=dnn_brake, gt_control=control_input, criterion_brake=criterion_brake)
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

        if i % config.loss_print_interval == 0:
            print("[%d, %5d] validation dataset loss: %.4f" %(epoch + 1, i + 1, epoch_loss / config.loss_print_interval))
            
        i += 1

    return epoch_loss


if __name__ == "__main__":
    print("\n================== Initializing Training (chuff chuff) ! ==================\n")

    # create model save path and folder if not exsisted
    os.makedirs(config.model_save_path)
  
    # combine all imitation learning datasets
    training_dataset = []
    validation_dataset = []

    print("All Route Paths To Be Trained On:")

    # get all training set
    for train_town in config.train_towns:
        print("Train Town: ", train_town)
        town_directory = config.training_data_path + train_town + "/"

        # get all different route trajectories from this town
        training_subdirs = os.listdir(town_directory)
        
        # loop throuth all sub-directories of training dataset
        for i, subdir in enumerate(training_subdirs):
            training_path_temp = os.path.join(town_directory, subdir)
            print(str(i) + "-imitation dataset: ", training_path_temp)
            
            # get and append every dataset for training
            constructed_dataset = DatasetLoader(training_path_temp)
            training_dataset.append(constructed_dataset)

    # get validation set
    for validation_town in config.validation_towns:
        print("Validation Town: ", validation_town)
        town_directory = config.validation_data_path + validation_town + "/"

        # get all different route trajectories from this town
        validation_subdirs = os.listdir(town_directory)
        
        # loop throuth all sub-directories of validation dataset
        for i, subdir in enumerate(validation_subdirs):
            validation_path_temp = os.path.join(town_directory, subdir)
            print(str(i) + "-validation dataset: ", validation_path_temp)
            
            # get and append every dataset for validation
            constructed_dataset = DatasetLoader(validation_path_temp)
            validation_dataset.append(constructed_dataset)

    print("\n================== Finished Loading Training Dataset ! ==================\n")

    # concatenated all training and validation datasets
    total_training_data = torch.utils.data.ConcatDataset(training_dataset)
    total_validation_data = torch.utils.data.ConcatDataset(validation_dataset)

    print("Total Size of Training Dataset: ", len(total_training_data))
    print("Total Size of Validation Dataset: ", len(total_validation_data))

    # create dataloaders
    train_dataset_loader = DataLoader(total_training_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    validation_dataset_loader = DataLoader(total_validation_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training Device: ", device)

    # construct imitation learining agent object
    network = ImitationNetwork(device=device)
    network.to(device)
    
    # define loss criterion
    criterion_brake = nn.BCELoss()

    # define network optimizer and learning rate scheduler
    optimizer = optim.SGD(network.parameters(), config.learning_rate, momentum=config.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.min_milestone, config.max_milestone], gamma=config.gamma)

    if config.pretrained is True:
        network.load_state_dict(torch.load(config.trained_model_path))

    writer = SummaryWriter(log_dir=config.model_save_path + "runs/")
    writer_counter = 0

    # main training loop
    for epoch in range(config.max_number_of_epochs):

        epoch_loss_train, writer_counter = trainer(writer_counter)
        epoch_loss_validation = 0.0

        # save model periodically
        if epoch % config.save_every_n_epoch == 0:
            torch.save(network.state_dict(), os.path.join(config.model_save_path, "epoch_%d.pth"%(epoch)))
            
        # validate model periodically on unseen dataset
        if epoch % config.validate_per_n == 0:
            epoch_loss_validation = validator()

        writer.add_scalar("training-dataset-epoch-loss", epoch_loss_train / float(len(total_training_data)), epoch)
        writer.add_scalar("validation-dataset-epoch-loss", epoch_loss_validation / float(len(total_validation_data)), epoch)
    
        scheduler.step()

    print("\n================== Finished Training ! ==================\n")
