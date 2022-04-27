# Example Implementation
#
# Description:
#   train and test DCI network.
#
# Copyright (C) 2018 Santiago Cortes
#
# This software is distributed under the GNU General Public
# Licence (version 2 or later); please refer to the file
# Licence.txt, included with the software, for details.

import json
import sys
from model import vel_regressor
from dataset import ToTensor
from dataset import OdometryDataset
import csv
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import matplotlib
from torch.utils.tensorboard import SummaryWriter

import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()


matplotlib.use("AGG")

# Import python functions.

def get_datasets():
    # add path to used folders
    # Advio
    folders = []
    for i in [13, 15, 16, 17, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22]:
        path = '/advio-'+str(i).zfill(2)+'/'
        folders.append(path)
    # Extra data
    folders.append("/static/dataset-01/")
    folders.append("/static/dataset-02/")
    folders.append("/static/dataset-03/")
    folders.append("/swing/dataset-01/")
    T = OdometryDataset("../data", folders, transform=ToTensor())
    index = np.arange(len(T))
    np.random.shuffle(index)
    train = index[1:int(np.floor(len(T)/10*9))]
    test = index[int(np.floor(len(T)/10*9)):-1]
    # Split training and validation.
    training_loader = DataLoader(T, batch_size=10, shuffle=False, num_workers=CPU_COUNT,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train)))
    validation_loader = DataLoader(T, batch_size=10, shuffle=False, num_workers=CPU_COUNT,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test)))
    
    return training_loader, validation_loader

def main():
    # set options
    load_model = True
    save_model = False
    train_model = True
    CURRENT_CHECKPOINT = 1131
    learning_rate = 1e-6

    writer = SummaryWriter()


    # add path to used folders
    # Advio
    folders = []
    for i in [13, 15, 16, 17, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22]:
        path = '/advio-'+str(i).zfill(2)+'/'
        folders.append(path)
    # Extra data
    folders.append("/static/dataset-01/")
    folders.append("/static/dataset-02/")
    folders.append("/static/dataset-03/")
    folders.append("/swing/dataset-01/")

    # Load saved motion labels
    labs = []
    with open('labels.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            labs.append([int(row[0]), int(row[1]), int(row[2]), float(row[3]), ])


    # visualize labels in sample vector.
    ind = 0
    acc_lab = 0
    acc_dat = 0
    data_labels = []
    fig, axs = plt.subplots(23, 1, figsize=(8, 35))
    for idx, folder in enumerate(folders):
        # Load one folder at a time
        data = OdometryDataset("../data", [folder], transform=ToTensor())
        # Skip last label from previous dataset
        while labs[ind][3] == -2:
            ind = ind+1
        # Find corresponding labels
        stay = True
        dat = []
        dat.append([-1, 0])
        while stay:
            tim = labs[ind][3]
            tim = np.round(np.floor(tim)*60+(tim-np.floor(tim))*100)
            data_length = (2+(data[int(len(data))]['time'])-data[0]['time'])[0]
            if labs[ind][3] == -1:
                stay = False
                tim = 10000
            lab = labs[ind][2]
            dat.append([tim, lab])
            ind = ind+1
        # Make label vector for each sample
        label = []
        start = data[0]['time']
        for i in range(0, len(data)):
            t = data[i]['time']-start
            for j in range(0, len(dat)-1):
                if t < dat[j+1][0] and t > dat[j][0]:
                    label.append(dat[j+1][1])
        # plot results
        acc_dat = acc_dat+len(data)
        acc_lab = acc_lab+len(label)
        ax = axs[idx]
        ax.plot(label)
        ax.set_ylim(-1, 5)
        frame1 = fig.gca()
        frame1.axes.get_xaxis().set_visible(False)
        ax.set_yticks([0., 1., 2., 3., 4.], labels=['Standing', 'Walking',
                    'Stairs', 'Escalator', 'Elevator'])
        ax.grid(axis='y')
        ax.set_title(f"Data of folder {folder}")
        data_labels.append(label)

    writer.add_figure("Visualized samples", fig)



    # Create dataset reader.
    T = OdometryDataset("../data", folders, transform=ToTensor())


    # load pretrained model or create new one.
    if load_model:
        # To change model between re-trained and pre trained uncomment/recomment this torch.load and optimizer.load_state_dict()

        # model = torch.load('./full-model.pt', map_location=lambda storage, loc: storage)
        # model = torch.load('./full.pt', map_location=lambda storage, loc: storage)
        checkpoint = torch.load(f"./torch_models/model_checkpoint_{CURRENT_CHECKPOINT}.pt")
        model = vel_regressor(Nout=1, Nlinear=7440)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model = vel_regressor(Nout=1, Nlinear=7440)
        # define optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # if "cuda" in repr(device):
    #     model = model.cuda()



    l = []
    val = []

    # plot velocity and speed.
    velo = []
    sp = []
    t = []
    index = (np.round(np.linspace(0, len(T), 1000)))
    for i in index:
        data = T[int(i)]
        velo.append(data['gt'].numpy())
        sp.append((data['gt'].norm()))
        t.append(data['time'])

    # fig.savefig("fig1.png")
    fig, ax = plt.subplots()
    ax.plot(velo)
    ax.set_title('Velocity Vector')
    ax.xlabel = 'sample'
    ax.ylabel = 'Speed (m/s)'
    ax.legend(['x', 'z', 'y'])
    # fig.savefig("fig2.png")
    fig, ax = plt.subplots()
    ax.set_title('Speed')
    ax.xlabel = 'sample'
    ax.ylabel = 'Speed (m/s)'
    ax.plot(sp)
    writer.add_figure("Speed", fig)

    # Configure data loaders and optimizer
    loss_fn = torch.nn.MSELoss(reduction='sum')
    index = np.arange(len(T))
    np.random.shuffle(index)
    train = index[1:int(np.floor(len(T)/10*9))]
    test = index[int(np.floor(len(T)/10*9)):-1]
    # Split training and validation.
    training_loader = DataLoader(T, batch_size=10, shuffle=False, num_workers=CPU_COUNT,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train)))
    validation_loader = DataLoader(T, batch_size=10, shuffle=False, num_workers=CPU_COUNT,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test)))
    # Create secondary loaders
    single_train_Loader = DataLoader(T, batch_size=1, shuffle=False, num_workers=1,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train)))
    single_validation_Loader = DataLoader(
        T, batch_size=1, shuffle=False, num_workers=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test)))
    ordered_Loader = DataLoader(T, batch_size=1, shuffle=False, num_workers=1)

    # Being function removes these variables after namespace exit -> free memory
    def add_graph():
        dataiter = iter(training_loader)
        _data = dataiter.next()
        writer.add_graph(model.cpu(), _data["imu"].float())
    add_graph()


    # Train model
    if train_model:
        # Epochs
        model.train()
        start = 0 if not load_model else CURRENT_CHECKPOINT + 1
        for t in range(start, 2000):
            epoch = t
            ti = time.time()
            acc_loss = 0
            val_loss = 0
            # Train
            train_time = time.time()
            for i_batch, data in enumerate(training_loader):

                # Forward pass.
                y_pred = model(data['imu'].float()).reshape(-1)
                # Sample corresponding ground truth.
                y = torch.norm(data['gt'], 2, 1).type(torch.FloatTensor)
                # Compute and print loss.
                loss = loss_fn(y_pred, y)
                # Save loss.
                acc_loss += loss.data.item()
                # Zero the gradients before running the backward pass.
                model.zero_grad()
                # Backward pass.
                loss.backward()
                # Take optimizer step.
                optimizer.step()
            train_time = time.time() - train_time
            validation_time = time.time()

            # Validation
            for i_batch, sample_batched in enumerate(validation_loader):
                # Sample data.
                data = sample_batched
                # Forward pass.
                y_pred = model(data['imu'].float()).reshape(-1)
                vec = data['gt']
                y = torch.norm(data['gt'], 2, 1).type(torch.FloatTensor)
                loss = loss_fn(y_pred, y)
                val_loss += loss.data.item()
            
            validation_time = time.time() - validation_time
            # Save loss and print status.
            l.append(acc_loss/(len(T)*9/10))
            val.append(val_loss/(len(T)/10))
            print(t)
            print((l[-1]))
            print((val[-1]))
            elapsed = time.time() - ti
            print(elapsed)
            validation_loss = val_loss/(len(T)/10)
            training_loss = acc_loss/(len(T)*9/10)
            writer.add_scalar("Validation loss", validation_loss, t)
            writer.add_scalar("Training loss", training_loss, t)
            writer.add_scalar("Training time", train_time, t)
            writer.add_scalar("Validation time", validation_time, t)
            model_path = os.path.join("torch_models", f"model_checkpoint_{epoch}.pt")
            model_json = os.path.join("torch_models", f"model_checkpoint_{epoch}.json")
            torch.save({'model_state_dict': model.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
            print('Model saved to ', model_path)
            with open(model_json, "w+") as f:
                json.dump({"training_loss": training_loss, "validation_loss": validation_loss, "training_time": train_time, "validation_time": validation_time}, f)
            print('Model train info saved to', model_json)

        
        # Plot loss
        ax.plot(np.log(np.array(l)), label='Training loss')
        ax.plot(np.log(np.array(val)), label='Validation loss')

    writer.add_figure("Training", fig)
    # fig.savefig("fig3.png")


    # save model
    if save_model:
        torch.save(model, './full.pt')
    model.eval()
    # Load corresponding prediction and ground truth
    pred = []
    sp = []
    for i_batch, sample_batched in enumerate(ordered_Loader):
        data = sample_batched
        pred.append(model(data['imu'].float()).data[0].numpy())
        vec = data['gt']
        y = torch.norm(data['gt'], 2, 1).type(torch.FloatTensor)
        sp.append(y.type(torch.FloatTensor).numpy())

    # Plot prediction and ground truth.
    print((np.shape((np.asarray(sp)))))

    fig, ax = plt.subplots()
    ax.plot(np.asarray(pred)[:, :])
    ax.ylabel = 'Speed (m/s)'
    ax.set_title('Prediction')
    writer.add_figure("Predicted speed", fig)
    # fig.savefig("fig4.png")



    fig, ax = plt.subplots()
    ax.plot(np.asarray(sp)[:, 0])
    ax.ylabel = 'ground truth speed'
    pred = np.asarray(pred)
    sp = np.asarray(sp)
    print(f"MSE = { np.square(sp - pred).sum() / sp.shape[0] } m/s")
    dat_lab = []
    for label in data_labels:
        dat_lab = dat_lab+label

    writer.add_figure("Ground truth speed", fig)
    # fig.savefig("fig5.png")


    # Plot scatter of prediction and ground truth with labels.
    pred = []
    sp = []
    R = []
    for i_batch, sample_batched in enumerate(ordered_Loader):
        data = sample_batched
        pred.append(model(data['imu'].float()).data[0].numpy())
        vec = data['gt']
        y = torch.norm(data['gt'], 2, 1).type(torch.FloatTensor)
        sp.append(y.type(torch.FloatTensor).numpy())
        R.append(np.array([x[0].item() for x in data["range"]]))
    print((len(dat_lab)))
    print((len(sp)))
    pred = np.asarray(pred)
    sp = np.asarray(sp)
    stat = []
    stair = []
    walk = []
    esc = []
    ele = []

    Rstat = []
    Rstair = []
    Rwalk = []
    Resc = []
    Rele = []

    # Separte by label
    for i in range(0, len(dat_lab)):
        if dat_lab[i] == 0:
            stat.append([sp[i, 0], pred[i].item()])
            Rstat.append(R[i])
        elif dat_lab[i] == 1:
            walk.append([sp[i, 0], pred[i].item()])
            Rwalk.append(R[i])
        elif dat_lab[i] == 2:
            stair.append([sp[i, 0], pred[i].item()])
            Rstair.append(R[i])
        elif dat_lab[i] == 3:
            esc.append([sp[i, 0], pred[i].item()])
            Resc.append(R[i])
        else:
            ele.append([sp[i, 0], pred[i].item()])
            Rele.append(R[i])
    msize = 3


    fig, ax = plt.subplots(figsize=(8, 8))
    # Scatter plot.
    test = np.array(stat)
    ax.plot(test[:, 0], test[:, 1], 'r.', label='static', markersize=msize)
    test = np.array(stair)
    ax.plot(test[:, 0], test[:, 1], 'g.', label='stair', markersize=msize)
    test = np.array(walk)
    ax.plot(test[:, 0], test[:, 1], 'b.', label='walk', markersize=msize)
    test = np.array(esc)
    ax.plot(test[:, 0], test[:, 1], 'k.', label='escalator', markersize=msize)
    test = np.array(ele)
    ax.plot(test[:, 0], test[:, 1], 'y.', label='elevator', markersize=msize)

    ax.plot([0, 1.5], [0, 1.5], 'k')
    ax.xlabel = 'gt (m/s)'
    ax.ylabel = 'prediction (m/s)'

    # plot histograms by label
    axes = fig.gca()
    axes.set_xlim((0.0, 1.5))
    axes.set_ylim([0.0, 1.5])
    axes.legend()
    #axes.grid(b=True, which='major', color='k', linestyle='--')
    bins = np.linspace(0.0, 2.0, 20)
    f = 0

    writer.add_figure("Accuracy", fig)
    # fig.savefig("fig6.png")


    fig, axs = plt.subplots(2, 3)
    axs = axs.reshape(-1)
    # plt.subplots(511)
    print(axs)
    fig.suptitle('minimum')
    axs[0].ylabel = 'static'
    test = np.array(Rstat)
    axs[0].hist(test[:, f], bins=bins)
    # plt.subplots(512)
    axs[1].ylabel = 'stairs'
    test = np.array(Rstair)
    axs[1].hist(test[:, f], bins=bins)
    # plt.subplots(513)
    axs[2].ylabel = 'walk'
    test = np.array(Rwalk)
    axs[2].hist(test[:, f], bins=bins)
    # plt.subplots(514)
    axs[3].ylabel = 'escalator'
    test = np.array(Resc)
    axs[3].hist(test[:, f], bins=bins)
    # plt.subplots(515)
    axs[4].ylabel = 'elevator'
    test = np.array(Rele)
    axs[4].hist(test[:, f], bins=bins)

    f = 1
    # fig.savefig("fig7.png")
    writer.add_figure("Minimum", fig)


    fig, axs = plt.subplots(2, 3)
    # plt.subplots(511)
    fig.suptitle('maximum')
    axs[0][0].ylabel = 'static'
    test = np.array(Rstat)
    axs[0][0].hist(test[:, f], bins=bins)
    # plt.subplots(512)
    axs[0][1].ylabel = 'stairs'
    test = np.array(Rstair)
    axs[0][1].hist(test[:, f], bins=bins)
    # plt.subplots(513)
    axs[0][2].ylabel = 'walk'
    test = np.array(Rwalk)
    axs[0][2].hist(test[:, f], bins=bins)
    # plt.subplots(514)
    axs[1][0].ylabel = 'escalator'
    test = np.array(Resc)
    axs[1][0].hist(test[:, f], bins=bins)
    # plt.subplots(515)
    axs[1][1].ylabel = 'elevator'
    test = np.array(Rele)
    axs[1][1].hist(test[:, f], bins=bins)


    # fig.savefig("fig8.png")
    writer.add_figure("Maximum", fig)




    fig, axs = plt.subplots(2, 1)
    # Evaluate in unknown data to the network.
    nfolders = []
    nfolders.append("/static/dataset-04/")
    Test = OdometryDataset("./../data/", nfolders, transform=ToTensor())
    test_Loader = DataLoader(Test, batch_size=1, shuffle=False, num_workers=1)

    pred = []
    sp = []
    t = []
    for i_batch, sample_batched in enumerate(test_Loader):
        data = sample_batched
        pred.append(model(data['imu'].float()).data[0].cpu().numpy())
        vec = data['gt']
        # vertical=torch.norm(vec[:,[1]],2,1)
        # vertical=vec[:,1]
        # horizontal=torch.norm(vec[:,[0,2]],2,1)
        # y=torch.stack((vertical,horizontal),1)
        y = torch.norm(data['gt'], 2, 1).type(torch.FloatTensor)
        sp.append(y.type(torch.FloatTensor).numpy())
        t.append(data['time'])
    # plt.subplots(211)
    axs[0].plot(np.asarray(pred))
    axs[0].ylabel = 'Predicted speed'
    # plt.subplots(212)
    axs[1].plot(np.asarray(sp)[:, 0])
    axs[1].ylabel = 'ground truth speed'

    writer.add_figure("Predicted vs ground truth1", fig)
    # fig.savefig("fig9.png")


    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.asarray(sp)[:, 0], np.asarray(pred)[:], '.', label='test data')
    ax.plot([0, 2], [0, 2], 'k')
    ax.xlabel = 'gt (m/s)'
    ax.ylabel = 'prediction (m/s)'

    axes = plt.gca()

    axes.set_xlim((0.0, 2))
    axes.set_ylim([0.0, 2])
    axes.legend()
    # plt.show()
    writer.add_figure("Predicted vs ground truth", fig)
    # fig.savefig("fig10.png")



if __name__ == '__main__':
    main()
