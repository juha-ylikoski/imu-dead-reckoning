# ronin-odometry

This folder contains scripts to run imu-odometry with ronin ResNet velocity calculation.

Code inside folder `ronin` is from https://github.com/Sachini/ronin and mostly not modified.

`odometry.py` contains functionality to calculate odometry for ronin dataset. (`python3 odometry.py --help`)


These scripts cache files (ronin caches to `$(cwd)/.cache` and odometry.py to `odometry.py.cache`) to improve performance.


