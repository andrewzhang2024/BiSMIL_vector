#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from model import *
from utils import *
import time

start_time = time.time()
seed= 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
seed_everything(seed)

debug = 0
def initialize_training(config):
    para = config['parameters']
    model_class = config["model_class"]  # 
    loss_func = nn.BCELoss()
    if ("ADMIL" in para['experiment_name']) or ("Pooling" in para['experiment_name']) :
        model = model_class()
    elif ("SiSMIL" in para['experiment_name']) or ("BiSMIL" in para['experiment_name']):
        model = BiSMIL(feature_dim=config['feature_dim'], num_heads=para['num_heads'], 
        num_layers=para['num_layers'],ff_dim=para['ff_dim'], output_dim=2, dropout = para['dropout'],  clip_ratio= para['clip_ratio'])
    elif "SA_DMIL" in para['experiment_name']:
        model = SA_DMIL()
        loss_func = SmoothMIL(alpha=para['alpha_SADMIL'], S_k=1)
    else:
        raise("Not implemented Error")
    optimizer = torch.optim.Adam(model.parameters(), lr=para["learning_rate"], weight_decay=para["weight_decay"])
    model = model.to(device)

    return model, optimizer, loss_func


def train_epoch(model, train_loader, optimizer, loss_func, epoch, parameters, num_epochs, incremental_training, alpha, beta, cumulative_loss, total_steps):
    model.train()
    correct_train = 0
    total_train = 0
    running_loss = 0
    misclassified_samples = []
    step_losses = []
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit="batch")


    for i, (bag, label, bag_id, bag_seq_digits) in enumerate(train_loader_tqdm):
        if debug != 0:
            print("Just need one label for a bag because they are all the same: ", label, label[0])
        bag, label = bag.to(device), torch.tensor(label[0]).to(device)
        optimizer.zero_grad()
        label = label.float()

        if incremental_training and 'BiSMIL' in parameters["experiment_name"]:
            outputs_f, outputs_r = model(bag)
            if debug != 0:
                print("outputs_f, outputs_r", outputs_f, outputs_r)
            final_output_avg = (outputs_f[-1] + outputs_r[-1]) / 2.0
            if debug != 0:
                print("final_output_avg 1: ", final_output_avg)
            if len(final_output_avg.shape) == 0:
                final_output_avg = final_output_avg.unsqueeze(0)
                print("final_output_avg 2: ", final_output_avg)
            output = final_output_avg
            loss_bce = loss_func(final_output_avg, label)
            softmax_sequence = construct_sequence(len(outputs_f))
            loss_wil_f = compute_weighted_incremental_loss(outputs_f, label, softmax_sequence)
            loss_wil_r = compute_weighted_incremental_loss(outputs_r, label, softmax_sequence)

            total_loss = alpha * loss_bce + beta * (loss_wil_f + loss_wil_r)
        elif 'SiSMIL' in parameters["experiment_name"]:
            outputs_f = model(bag)
            final_output_avg = outputs_f[-1]
            if len(final_output_avg.shape) == 0:
                final_output_avg = final_output_avg.unsqueeze(0)
            output = final_output_avg
            loss_bce = loss_func(final_output_avg, label)
            softmax_sequence = construct_sequence(len(outputs_f))
            loss_wil_f = compute_weighted_incremental_loss(outputs_f, label, softmax_sequence)
            total_loss = alpha * loss_bce + beta * (loss_wil_f)
        elif "SA_DMIL" in parameters["experiment_name"]:
            output, att_weights = model(bag)
            total_loss = loss_func(output, label, att_weights)
        else:
            output = model(bag, total_len=bag.shape[1])
            total_loss = loss_func(output, label)

        running_loss += total_loss.item()
        cumulative_loss += total_loss.item()
        total_steps += 1
        step_losses.append(cumulative_loss / total_steps)

        total_loss.backward()
        optimizer.step()

        predicted_train = (output > 0.5).float()
        #total_train += label.size(0)
        total_train += 1 #batch size is 1
        #print("Will compare predict and label: ", predicted_train, label)
        #correct_train += (predicted_train == label[0]).sum().item()
        #correct_train += (predicted_train == label).sum().item()
        
        missed = (predicted_train != label)
        #print("missed: ", missed)
        if missed.any():
            #
            #misclassified_samples.append((epoch, bag_id, predicted_train.item(), label.item()))
            misclassified_samples.append((epoch, bag_id, torch.sum(predicted_train), torch.sum(label)))
        else:
            correct_train +=  1
        train_loader_tqdm.set_postfix(avg_loss=running_loss/(i+1), accuracy=correct_train/total_train)
        

    return correct_train, total_train, running_loss, misclassified_samples, step_losses, cumulative_loss, total_steps



def test_epoch(model, data_loader, epoch, parameters, num_epochs, incremental_training, mode="test"):
    """
    Test or validate the model for a single epoch with batch size 1.
    """
    model.eval()
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    misclassified_samples = []

    # Updating the tqdm description based on the mode
    tqdm_desc = f"Epoch {epoch+1}/{num_epochs} {'Testing' if mode == 'test' else 'Validation'}"
    data_loader_tqdm = tqdm(data_loader, desc=tqdm_desc, unit="batch")

    with torch.no_grad():
        for i, (bag, label, bag_id, bag_seq_digits) in enumerate(data_loader_tqdm):
            if debug != 0:
                print("label as a list: ", label[0])
            bag, label = bag.to(device), torch.tensor(label[0]).to(device)
            if incremental_training and 'BiSMIL' in parameters["experiment_name"]:
                outputs_f, outputs_r = model(bag)
                final_output_avg = (outputs_f[-1] + outputs_r[-1]) / 2.0
                if len(final_output_avg.shape) == 0:
                    final_output_avg = final_output_avg.unsqueeze(0)
                output = final_output_avg
            elif 'SiSMIL' in parameters["experiment_name"]:
                outputs_f = model(bag)
                final_output_avg = (outputs_f[-1] ) 
                if len(final_output_avg.shape) == 0:
                    final_output_avg = final_output_avg.unsqueeze(0)
                output = final_output_avg
            elif "SA_DMIL" in parameters["experiment_name"]:
                output, att_weights = model(bag)
            else:
                output = model(bag, total_len=bag.shape[1])

            label = label.float()
            predicted = (output > 0.5).float()
            total += 1  # Since batch size is 1
            #correct += (predicted == label).sum().item()

            # Calculating TP, FP, FN for Precision, Recall, F1
            true_positives += ((predicted == 1) & (label == 1)).sum().item()
            false_positives += ((predicted == 1) & (label == 0)).sum().item()
            false_negatives += ((predicted == 0) & (label == 1)).sum().item()
            missed = (predicted != label)
            if debug != 0:
                print("missed: ", missed)
            if missed.any():
            #if predicted != label:
                misclassified_samples.append((epoch, bag_id, torch.sum(predicted), torch.sum(label)))
            else:
                correct +=  1    
            accuracy = correct / total
            data_loader_tqdm.set_postfix(accuracy=accuracy)

    # Calculating Precision, Recall, and F1-Score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if mode == 'test':
        print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
        return correct, total, misclassified_samples, precision, recall, f1_score
    else:
        return correct, total, misclassified_samples





#path_utd ='/data/MIL_dataset/data_final_exp/UTD'
#train_dataset = torch.load(path_utd + '/UTD_train_dataset.pt')
#validation_dataset = torch.load(path_utd + '/UTD_validation_dataset.pt')
#test_dataset = torch.load( path_utd+ '/UTD_test_dataset.pt')

path_utd ='data_utd'
train_dataset = torch.load(path_utd + '/UTD_train_dataset.pt', weights_only=False)
validation_dataset = torch.load(path_utd + '/UTD_test_dataset.pt', weights_only=False)
test_dataset = torch.load( path_utd+ '/UTD_test_dataset.pt', weights_only=False)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print("Successfully load the data")

#with open('config.pkl', 'rb') as config_file:
#    configurations = pickle.load(config_file)
configurations = {
    "BiSMIL": {
        "model_class": 'BiSMIL',
        "feature_dim" : 288,
        "parameters": {
            "num_epochs": 50, #100, #0, #60
            "seed": 1,
            "learning_rate": 2e-5,
            "weight_decay": 1e-4,
            "dropout": 0.25,
            "num_heads": 8,
            "num_layers": 4, #2,
            "ff_dim":128,
            "incremental_training": True,
            "alpha": 0.5,
            "beta": 0.5,
            "clip_ratio": 0.6,
            "experiment_name": "BiSMIL_UTD_Seed_1"
        }}
}
# Configuration for different model setups


# Selecting configuration and initializing parameters
selected_config = configurations["BiSMIL"]
parameters = selected_config['parameters']
print("selected_config : \n", selected_config, "\n")

# Seed setting for reproducibility
seed_everything(parameters["seed"])
print("Current Seed: ", parameters["seed"])

# Initializing variables for training
best_test_accuracy = 0.0
best_val_accuracy = 0.0
cumulative_loss = 0.0
total_steps = 0
train_accuracy, test_accuracy, validation_accuracy = [], [], []
precision_list, recall_list, f1_score_list = [], [], []
misclassified_samples_count = {}
step_losses_recorded = []

experiment_path =  parameters["experiment_name"] + "/"
os.makedirs(experiment_path, exist_ok=True)
excel_file_path = 'Results_UTD.xlsx'

# Saving the parameters to a file
parameters_file = experiment_path + "parameters.json"
with open(parameters_file, 'w') as file:
    json.dump(parameters, file, indent=4)

# Initializing model, optimizer, and loss function
model, optimizer, loss_function = initialize_training(selected_config)
print("Model Parameters: \n", count_parameters_and_proportion(model))
print("--- %s BiSMIL set up seconds ---" % (time.time() - start_time))


# In[2]:


# Training loop
for epoch in range(parameters["num_epochs"]):
    correct_train, total_train, running_loss, misclassified_train, step_losses, cumulative_loss, total_steps = train_epoch(
        model, train_loader, optimizer, loss_function, epoch,  parameters, 
        parameters["num_epochs"], parameters["incremental_training"], parameters["alpha"], parameters["beta"], cumulative_loss, total_steps)
    train_accuracy.append(correct_train / total_train)
    update_misclassified_samples(misclassified_samples_count, misclassified_train)
    step_losses_recorded.extend(step_losses)
    
    correct_val, total_val, misclassified_val = test_epoch(model, validation_loader, epoch, parameters,
        parameters["num_epochs"], incremental_training = parameters["incremental_training"], mode="val")
    validation_accuracy.append(correct_val / total_val)
    update_misclassified_samples(misclassified_samples_count, misclassified_val)
    
    correct_test, total_test, misclassified_test, precision, recall, f1_score = test_epoch(model, test_loader, epoch, parameters,
        parameters["num_epochs"], incremental_training = parameters["incremental_training"], mode="test")
    test_accuracy.append(correct_test / total_test)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1_score)
    update_misclassified_samples(misclassified_samples_count, misclassified_test)

    # Model checkpointing
    if correct_val / total_val > best_val_accuracy:
        best_val_accuracy = correct_val / total_val
        save_checkpoint(model, experiment_path + "model_best_val.pth", is_best=True)

    if correct_test / total_test > best_test_accuracy:
        best_test_accuracy = correct_test / total_test
        save_checkpoint(model, experiment_path + "model_best_test.pth", is_best=True)

    save_checkpoint(model, experiment_path + f"model_epoch_{epoch+1}.pth")

# Saving training artifacts and plotting results
training_artifacts = {
    'step_losses': step_losses_recorded,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'validation_accuracy': validation_accuracy,
    'best_test_accuracy': best_test_accuracy,
    'cumulative_loss': cumulative_loss,
    'total_steps': total_steps
}
save_training_artifacts(experiment_path, training_artifacts)
save_training_plots(step_losses_recorded, train_accuracy, test_accuracy, f1_score_list, 
    validation_accuracy, best_test_accuracy, parameters["experiment_name"], experiment_path)

highest_test_acc_epoch = test_accuracy.index(max(test_accuracy))
highest_val_acc_epoch = validation_accuracy.index(max(validation_accuracy))
print("Highest Test Accuracy:", max(test_accuracy), "at epoch", highest_test_acc_epoch)
print("Highest Validation Accuracy:", max(validation_accuracy), "at epoch", highest_val_acc_epoch)
print("Test Accuracy corresponding to Highest Validation Accuracy:", test_accuracy[highest_val_acc_epoch])
print("Test F1 Score corresponding to Highest Validation Accuracy:", f1_score_list[highest_val_acc_epoch])
print("Test Precision corresponding to Highest Validation Accuracy:", precision_list[highest_val_acc_epoch])
print("Test Recall corresponding to Highest Validation Accuracy:", recall_list[highest_val_acc_epoch])


experiment_results = {
    "Experiment Name": parameters["experiment_name"],
    "Seed Number": parameters["seed"],
    "Model Name": 'BiSMIL',#selected_config["model_class"].__name__,
    "Highest Test ACC": max(test_accuracy),
    "Highest Val ACC": max(validation_accuracy),
    "Test ACC for Highest Val ACC": test_accuracy[highest_val_acc_epoch],
    "Test F1 for Highest Val ACC": f1_score_list[highest_val_acc_epoch],
    "Test Pre for Highest Val ACC": precision_list[highest_val_acc_epoch],
    "Test Rec for Highest Val ACC": recall_list[highest_val_acc_epoch],
}
save_to_excel(experiment_results, excel_file_path)
print("--- %s BiSMIL total run seconds ---" % (time.time() - start_time))


# In[ ]:




