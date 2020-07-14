import torch
import numpy as np


def train(model, train_dataloader, val_dataloader, epochs, criterion_loss, optimizer, output_path):
    train_loss_list, val_loss_list =[], []

    for epoch in epochs:
        running_loss = 0.0
        #  TRAIN:  #
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'], data['gt']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / i
        train_loss_list.append(train_loss)

        #  VAL:  #
        val_running_loss = 0.0
        for val_i, val_data in enumerate(val_dataloader):
            inputs, labels = val_data['image'], val_data['gt']
            outputs = model(inputs)
            loss = criterion_loss(outputs, labels)
            running_loss += loss.item()
        val_loss = val_running_loss / val_i
        val_loss_list.append(val_loss)

        #  PRINT:  #
        print("Epoch {}:  train loss: {:.5f}, val loss: {:.5f}".format(epoch, train_loss, val_loss))