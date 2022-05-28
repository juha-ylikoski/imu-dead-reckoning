# ekf-odometry

This folder contains scripts to run extended Kalman filter odometry on ADVIO dataset. 

It assumes you have ADVIO dataset root located in root of this repository inside folder `deep-speed-constrained-ins-dataset`, trained models for `deep-speed-constrained-ins` inside `saved_models` provided by this repo.

`ekf.py` contains ekf and loads pytorch model and dataloaders to run odometry estimations.

