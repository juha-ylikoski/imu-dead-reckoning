
import numpy as np
import pandas
from scipy.spatial.transform import Rotation

from model import vel_regressor

import matplotlib.pyplot as plt
import torch

from tqdm import tqdm

from timeit import default_timer as timer

from dataset import ToTensor
from dataset import OdometryDataset

DATASET_ROOT = "deep-speed-constrained-ins-dataset"
MODEL_PATH_FULL = "saved_models/ekf_vel_regressor/pre-trained.pt"
MODEL_PATH_RE_TRAINED = "saved_models/ekf_vel_regressor/re-trained.pt"


def get_datasets(datasets):
    # add path to used folders
    # Advio
    folders = []
    # for i in [13, 15, 16, 17, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22]:
    for i in datasets:
        path = '/advio-'+str(i).zfill(2)+'/'
        folders.append(path)
    # Extra data
    folders.append("/static/dataset-01/")
    folders.append("/static/dataset-02/")
    folders.append("/static/dataset-03/")
    folders.append("/swing/dataset-01/")
    T = OdometryDataset(DATASET_ROOT, folders, transform=ToTensor())
    return T

def quaternion_derivative(q: np.array):
    dR = np.array([
            [[2*q[0], -2*q[3],  2*q[2]],    [2*q[3],  2*q[0], -2*q[1]],    [2*q[2],  2*q[1],  2*q[0]]],
            [[2*q[1],  2*q[2],  2*q[3]],    [2*q[2], -2*q[1], -2*q[0]],    [2*q[3],  2*q[0], -2*q[1]]],
            [[-2*q[2],  2*q[1],  2*q[0]],   [2*q[1],  2*q[2],  2*q[3]],    [-2*q[0],  2*q[3], -2*q[2]]],
            [[-2*q[3], -2*q[0],  2*q[1]],   [2*q[0], -2*q[3],  2*q[2]],    [2*q[1],  2*q[2],  2*q[3]]]
            ])
    return dR

POS = np.array([0,1,2])
VEL = np.array([3,4,5])
ORI = np.array([6,7,8,9])

# BGA = BIAS GYRO ADDITIVE
BGA = np.array([10,11,12])
# BAA = BIAS ACCELEROMETER ADDITIVE
BAA = np.array([13,14,15])
# BAT = BIAS ACCELEROMETER TRANSLATIVE
BAT = np.array([16,17,18])

Q_ACC = np.array([0, 1, 2])
Q_GYRO = np.array([3, 4, 5])
Q_BGA_DRIFT = np.array([6, 7, 8])
Q_BAA_DRIFT = np.array([9, 10, 11])


class EKF:
    """Ektended Kalman Filter implementation.

    Ported from c++ (https://github.com/SpectacularAI/HybVIO)
    """
    def __init__(self, dim_x, dim_z) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z

        # state m = [
        # 0-2:   p0, p1, p2,
        # 3-5:   v0, v1, v2,
        # 6-9:   qw, qx, qy, qz,
        # 10-12: bga0, bga1, bga2,
        # 13-15: baa0, baa1, baa2,
        # 16-18: bat0, bat1, bat2,
        # ]

        self.state = np.zeros(19) # state
        self.state[ORI[0]] = 1 # Set quaternion[0] to 1
        self.Q = 4.8e-3 * np.eye(19)   # process uncertainty


        self.P = np.zeros(19)        # uncertainty covariance

        self.first_sample = True
        self.prev_sample_t: float = None

        self.acc_bias_additive = np.array([0.0407, -0.0623, 0.1017])
        self.gyro_bias_additive = np.array([-0.0067, 0.0070, -0.0065])
        self.acc_bias_transformative = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
            ])

        self.noise_scale = 1
        self.acc_bias_additive_MEAN_REV = 2.1e-4
        self.gyro_bias_additive_MEAN_REV = 5.1e-5

        self.gravity = np.array([0, 0, 0])

        self.pseudo_velocity_r = 1e-4
    

    def normalize_quaternions(self):
        self.state[ORI] /= np.linalg.norm(self.state[ORI])

    def predict(self, time: float, gyro_rot: np.array, imu_acc: np.array):
        dt = 0
        if not self.first_sample:
            dt = time - self.prev_sample_t
        else:
            self.first_sample = False
        self.prev_sample_t = time
        Q = self.Q
        P = self.P
        state = self.state

        Q[Q_BAA_DRIFT,Q_BAA_DRIFT] = np.ones(3) * self.noise_scale * (self.acc_bias_additive**2)
        Q[Q_BAA_DRIFT,Q_BAA_DRIFT] *= (1 - np.exp(-2 * dt * self.acc_bias_additive_MEAN_REV)) / ( 2 * self.acc_bias_additive_MEAN_REV )

        Q[Q_BGA_DRIFT,Q_BGA_DRIFT] = np.ones(3) * self.noise_scale * (self.gyro_bias_additive**2)
        Q[Q_BGA_DRIFT,Q_BGA_DRIFT] *= (1 - np.exp(-2 * dt * self.gyro_bias_additive_MEAN_REV)) / (2 * self.gyro_bias_additive_MEAN_REV)

        # Gyro rotation
        w = gyro_rot - state[BGA]

        # Angular velocity to angle
        S = np.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, -w[2], w[1]],
            [w[1], w[2], 0, -w[0]],
            [w[2], -w[1], w[0], 0]
        ])
        S *= -dt / 2
        A = np.exp(S)

        # Rotation from quaternion and the derivative
        current_rotation = A.dot(state[ORI])
        dR = quaternion_derivative(current_rotation)
        rotation = Rotation.from_quat(current_rotation)


        # Add velocity to position
        state[POS] += state[VEL] * dt

        # Add acceleration input to velocity
        a = self.acc_bias_transformative.dot(imu_acc) - self.acc_bias_additive
        state[VEL] += rotation.apply(a - self.gravity) * dt


        # Update orientation. Quaternion product
        prev_quat = state[ORI]
        state[ORI] = A.dot(state[ORI])

        # Additive biases mean reversion (BAA, BGA)
        state[BAA] *= np.exp(-dt * self.acc_bias_additive_MEAN_REV)
        state[BGA] *= np.exp(-dt * self.gyro_bias_additive_MEAN_REV)

        dydx = np.identity(19)
        dydq = np.zeros((19, 19))
        dydx[POS,VEL] *= dt

        # Derivatives of the velocity w.r.t to the quaternion
        for i in range(4):
            dydx[VEL, ORI[0]+i] = dR[i].T.dot(a * dt)

        # For some reason I cannot slice 3x4 array from larger numpy array with arrays but
        # with ranges it works...
        dydx[VEL][:,ORI] = dydx[VEL][:,ORI].dot(A)

        #  Derivatives of the quaternion w.r.t itself
        dydx[ORI][:,ORI] = A

        # Derivatives of the velocity w.r.t. acceleration noise
        dydq[VEL][:,Q_ACC] = rotation.as_matrix().T * dt

        # Derivatives of the quaternion w.r.t. gyroscope noise
        dS0 = np.array([0, dt / 2, 0, 0, -dt / 2, 0, 0, 0, 0, 0, 0, dt / 2, 0, 0, -dt / 2, 0]).reshape(4,4)
        dS1 = np.array([0, 0, dt / 2, 0, 0, 0, 0, -dt / 2, -dt / 2, 0, 0, 0, 0, dt / 2, 0, 0]).reshape(4,4)
        dS2 = np.array([0, 0, 0, dt / 2, 0, 0, dt / 2, 0, 0, -dt / 2, 0, 0, -dt / 2, 0, 0, 0]).reshape(4,4)
        dydq[ORI, Q_GYRO[0]+0] = A.dot(dS0).dot(prev_quat)
        dydq[ORI, Q_GYRO[0]+1] = A.dot(dS1).dot(prev_quat)
        dydq[ORI, Q_GYRO[0]+2] = A.dot(dS2).dot(prev_quat)
        dydq[BGA][:,Q_BGA_DRIFT][:] = 1
        dydq[BAA,Q_BAA_DRIFT] = 1

        # Derivatives of the velocity w.r.t gyroscope noise
        dydq[VEL][:,Q_GYRO] = dydq[VEL][:,ORI].dot(dydq[ORI][:,Q_GYRO])

        # Derivatives of the quaternion w.r.t to the gyro bias
        dydx[ORI][:,BGA] = -dydq[ORI][:,Q_GYRO]

        # Derivatives of the velocity w.r.t the acc. bias
        dydx[VEL][:,BAA] = -rotation.as_matrix() * dt

        # Derivatives of the velocity w.r.t the acc. transformation
        dydx[VEL][:,BAT] = rotation.as_matrix().T.dot(np.diag(imu_acc)) * dt

        new_P = dydx.dot(P).dot(dydx.T) + dydq.dot(Q).dot(dydq.T)
        self.P = new_P
        self.Q = Q
        self.state = state

        self.normalize_quaternions()


    def update(self, pseudo_velocity):
        state = self.state
        P = self.P

        R = self.pseudo_velocity_r
        h = np.linalg.norm(state[VEL[0:-1]])

        y = pseudo_velocity - h
        if h <= 1e-7:
            return

        H = np.zeros((1, VEL[0] + 3))
        H[0, VEL] = state[VEL] / h

        l = H.shape[1]
        HP = H.dot(P[0:l])
        S = HP[:,0:l].dot(H.T)
        s = S[0,0] + R
        K = HP.T / s
        state += K.dot(pseudo_velocity - h).reshape(-1)
        self.state = state
        self.P = P

def state_to_speed(x: np.array):
    return np.mean(x[3:6])




class Biases:
    ACC_DIAGONAL_SCALE_ERROR = np.eye(3)
    ACC_BIAS = np.array([0.0407, -0.0623, 0.1017])
    # ACC_BIAS = np.zeros((3))
    # GYRO_BIAS = np.zeros((3))
    GYRO_BIAS = np.array([-0.0067, 0.0070, -0.0063])




if __name__ == "__main__":
    stride = 5
    ax = plt.gca()
    for dataset_ind in range(1, 23):
        dataset = get_datasets([dataset_ind])
        ekf = EKF(dim_x=3 + 3 + 3, dim_z=1)


        cnn_time_used = 0
        cnn_time_run = 0

        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0')


        # To change between pre-trained and re-trained model uncomment/comment these lines
        model = torch.load(MODEL_PATH_FULL, map_location=lambda storage, loc: storage)
        # checkpoint = torch.load(MODEL_PATH_RE_TRAINED)
        # model = vel_regressor(Nout=1, Nlinear=7440)
        # model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device).eval()

        imu_data = np.vstack([imu.to_numpy() for imu in dataset.imu])
        imut = np.vstack([x.to_numpy() for x in dataset.imut]).reshape(-1)
        post = np.vstack([x.to_numpy() for x in dataset.post]).reshape(-1)

        imu_data = imu_data[200:16000]
        imut = imut[200:16000]
        post = imut[200:16000]
        ground_truth = np.vstack([imu.to_numpy() for imu in dataset.pos])[200:15000]

        state_matrix = []
        cov_matrix = []

        dataset[0]


        start_time = timer()

        for cur_time, data in enumerate(tqdm(imu_data[:-200])):
            cur_time += 200
            u = np.zeros((3+3+3, 1), dtype=np.float64)
            angular_velocity = data[3:6].reshape(-1) - Biases.GYRO_BIAS
            dt = 0
            if not ekf.first_sample:
                dt = imut[cur_time] - ekf.prev_sample_t

            acc = data[0:3] / 9.8065
            gyro = data[3:6]

            ekf.predict(imut[cur_time], gyro, acc)
            

            acc_nn = torch.from_numpy(imu_data[cur_time-200:cur_time].T.reshape(1, 6, -1)).float().to(device)
            start = timer()
            velocity_nn = float(model(acc_nn).cpu())
            end = timer()
            cnn_time_used += end - start
            cnn_time_run += 1
            velocity_calc = state_to_speed(ekf.state)
            ekf.update(velocity_nn)
            
            state_matrix.append(ekf.state.copy())
        
        end_time = timer()
        print(f"CNN Took {cnn_time_used} s over {cnn_time_run} averaging to {0 if cnn_time_run == 0 else cnn_time_used/cnn_time_run} s/run when whole ekf took {end_time - start_time}s ")
        
        
        state_matrix = np.stack(state_matrix, axis=0)
        ax = plt.axes(projection='3d')

        line, = ax.plot(ground_truth[:,0], ground_truth[:,2], ground_truth[:,1], label="ground_truth")
        line, = ax.plot(state_matrix[:,0], state_matrix[:,2], state_matrix[:,1], label="ekf")


        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        def update(num):
            line.set_data(state_matrix[:num,0], state_matrix[:num,1])
            line.set_3d_properties(state_matrix[:num,2])
            print(num)
            line.axes.axis()

        ax.legend()
        plt.show()



