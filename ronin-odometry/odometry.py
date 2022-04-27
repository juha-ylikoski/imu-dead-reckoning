

from argparse import ArgumentParser
import enum
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import FuncFormatter, MultipleLocator
import contextlib

from tqdm import tqdm
from torch.utils.data import DataLoader

from joblib import Memory

from scipy.interpolate import interp1d


from model import get_model
from dataset import get_dataset, DataSet, DatasetMode
from metrics import compute_ate_rte


data_list = [
    "a006_2",
    "a019_3",
    "a024_1",
    "a024_3",
    "a029_1",
    "a029_2",
    "a032_1",
    "a032_3",
    "a042_2",
    "a049_1",
    "a049_2",
    "a049_3",
    "a050_1",
    "a050_3",
    "a051_1",
    "a051_2",
    "a051_3",
    "a052_2",
    "a053_1",
    "a053_2",
    "a053_3",
    "a054_1",
    "a054_2",
    "a054_3",
    "a055_2",
    "a055_3",
    "a057_1",
    "a057_2",
    "a057_3",
    "a058_1",
    "a058_2",
    "a058_3"
]

MODEL_PATH_DEFAULT = "saved_models/ronin_resnet/checkpoint_gsn_latest.pt"
DATASET_ROOT_DEFAULT = "ronin-dataset"


# Location to cache @memory.cache functions
memory = Memory(__file__ + ".cache")


# Makes sure tqdm progressbar does not break from prints: https://stackoverflow.com/a/37243211/16544034
class DummyFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)
    
    def flush(self):
        self.file.flush()


@contextlib.contextmanager
def nostdout():
    """Makes sure tqdm progressbar does not break from prints

    https://stackoverflow.com/a/37243211/16544034
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


def get_dataset_unseen(dataset_root):
    return os.path.join(dataset_root, "unseen_subjects_test_set")


class DATASET_INDEXES(int, enum.Enum):
    FEATURE = 0
    TARGET = 1
    SEQ_ID = 2
    FRAME_ID = 3


def get_ts(timestamps, seq_ids, frame_ids):
    ts = []
    for i, seq_id in enumerate(seq_ids):
        frame_id = frame_ids[i]
        ts.append(timestamps[seq_id][frame_id])
    return np.array(ts)


def get_gt(gt, seq_ids, frame_ids):
    gts = []
    for i, seq_id in enumerate(seq_ids):
        frame_id = frame_ids[i]
        gts.append(gt[seq_id][frame_id][:2])
    return np.array(gts)


@memory.cache(verbose=False)
def calc_neural_network_output(data_list, dataset_root, model_path):
    """Calculate neural network output and cache it to file to reduce extra processing.
        dataset_root (_type_): "ronin/data/unseen_subjects_test_set"


    Args:
        data_list (_type_): which scenarios to calculate
        dataset_root (_type_): dataset root path
        model_path (_type_): Path to pytorch model (*.pt)

    Returns:
        tuple: velocities, target velocities (neural network training labels), features, time, ground truth
    """
    model = get_model()
    dataset = get_dataset(get_dataset_unseen(dataset_root),
                          data_list, DataSet.RONIN, DatasetMode.TEST)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        checkpoint = torch.load(
            model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    model.eval()
    print('Model {} loaded to device {}.'.format(model_path, device))

    time = []
    targets = []
    features = []
    preds = []
    gt = []
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    timestamps = dataset.ts
    for feature, target, seq_id, frame_id in tqdm(loader, leave=False, desc=f"Scenario {data_list} nn calculation"):
        resp = model(feature.to(device))
        out = resp.cpu().detach().numpy()
        targets.append(target)
        features.append(feature.numpy())
        preds.append(out)
        time.append(get_ts(timestamps, seq_id.numpy(), frame_id.numpy()))
        gt.append(get_gt(dataset.gt_pos, seq_id.numpy(), frame_id.numpy()))

    vel = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    features = np.concatenate(features, axis=0)
    time = np.concatenate(time, axis=0)
    gt = np.concatenate(gt)
    return vel, targets, features, time, gt


def interpolate_route(time: np.array, route: np.array, new_time: np.array):
    interp_route = interp1d(time, route, axis=0)
    return interp_route(new_time)


def calc_for_scenario(data, show_plot, dataset_root, model_path):

    vel, targets, features, time, gt = calc_neural_network_output(
        [data], dataset_root, model_path)
    pos = np.ones((vel.shape[0], 2)) * gt[0]
    dt = time - np.roll(time, shift=1)
    dt[0] = 0
    dt[-1] = 0
    # Remove last intergrated position to make gt and pos have same shape
    pos[1:] += np.cumsum(vel * np.array((dt, dt)).T, axis=0)[:-1]
    new_time = np.linspace(time[0], time[-1], 100000)
    interp_pos = interpolate_route(time, pos, new_time)
    interp_gt = interpolate_route(time, gt, new_time)

    # plt.figure()

    ate, rte = compute_ate_rte(pos, gt)
    print(f"ATE={ate}, RTE={rte}")

    time_used = new_time[-1] - new_time[0]
    minutes = int(time_used / 60)
    seconds = int(time_used - minutes * 60)
    if show_plot:
        gt_line = plt.plot(gt[:, 0], gt[:, 1], label="Ground truth")
        pos_line = plt.plot(pos[:, 0], pos[:, 1], label="Estimate")

        plt.plot([], [], ' ', label=f"ATE={ate:.2f}, RTE={rte:.2f}")
        plt.plot([], [], ' ', label=f"time {minutes} min {seconds} s")
        plt.legend()
        plt.xlabel("m")
        plt.ylabel("m")

        def update(num):
            pos_line[0].set_data(pos[:num, 0], pos[:num, 1])
            gt_line[0].set_data(gt[:num, 0], gt[:num, 1])

        ani = animation.FuncAnimation(
            plt.gcf(), update, frames=range(0, pos.shape[0]-1, 100))
        ani.save("animation.gif")

        plt.show()
        gt_line = plt.plot(gt[:, 0], gt[:, 1], label="Ground truth")
        pos_line = plt.plot(pos[:, 0], pos[:, 1], label="Estimate")

        plt.plot([], [], ' ', label=f"ATE={ate:.2f}, RTE={rte:.2f}")
        plt.plot([], [], ' ', label=f"time {minutes} min {seconds} s")
        plt.legend()
        plt.xlabel("m")
        plt.ylabel("m")

        gt_line = None
        pos_line = None
        fig, axs = plt.subplots(2, 3)
        for i in range(6):
            end_gt = int(len(gt) / 7 * (i + 1))
            end_pos = int(len(pos) / 7 * (i + 1))
            row = i // 3
            col = i % 3
            gt_line = axs[row, col].plot(
                gt[:end_gt, 0], gt[:end_gt, 1], label="Ground truth")[0]
            pos_line = axs[row, col].plot(
                pos[:end_pos, 0], pos[:end_pos, 1], label="Estimate")[0]

        info_label0 = axs[0, 0].plot(
            [], [], ' ', label=f"ATE={ate:.2f}, RTE={rte:.2f}")[0]
        info_label1 = axs[0, 0].plot(
            [], [], ' ', label=f"time {minutes} min {seconds} s")[0]
        lines = [gt_line, pos_line, info_label0, info_label1]
        fig.legend(lines, [line.get_label() for line in lines],
                   loc='upper left', mode="expand", ncol=4)
        fig.supxlabel("m")
        fig.supylabel("m")
        plt.show()

    return ate, rte, time_used


def find_nearest(array, value):
    array = np.asarray(array)
    error = np.abs(array.T - value).T
    idx = error.mean(axis=1).argmin()
    return idx


def format_func(x, pos):
    hours = int(x//3600)
    minutes = int((x % 3600)//60)
    seconds = int(x % 60)
    return "{:02d}:{:02d}".format(minutes, seconds)


def main(args):
    ates = []
    rtes = []
    time = []
    for data in tqdm(data_list, file=sys.stdout, desc="All scenario calculation"):
        with nostdout():
            ate, rte, used_time = calc_for_scenario(
                data, False, args.dataset, args.model)
            ates.append(ate)
            rtes.append(rte)
            time.append(used_time)

    worst_ate = np.argmax(ates)
    worst_rte = np.argmax(rtes)

    print(f"average ATE={np.mean(ates)}, average RTE={np.mean(rtes)}")
    print(f"max ATE={ates[worst_ate]}, RTE={ates[worst_ate]}")
    print(f"ATE={rtes[worst_rte]}, max RTE={rtes[worst_rte]}")

    print("Worst ATE")
    calc_for_scenario(data_list[worst_ate], True, args.dataset, args.model)

    print("Worst RTE")
    calc_for_scenario(data_list[worst_rte], True, args.dataset, args.model)

    average = find_nearest(np.vstack((ates, rtes)),
                           np.array([np.mean(ates), np.mean(rtes)]))
    print("Average")
    calc_for_scenario(data_list[average], True, args.dataset, args.model)

    print("Distribution of error")
    bar_x = np.arange(len(ates))

    plt.bar(bar_x - 0.2, ates, 0.4, label="ATE")
    plt.bar(bar_x + 0.2, rtes, 0.4, label="RTE")

    ate_labels = [""] * len(ates)
    ate_labels[average] = "average"
    ate_labels[worst_ate] = "worst ate"
    ate_labels[worst_rte] = "worst rte"
    plt.ylabel("Error (m)")
    plt.xlabel("Scenario")
    plt.legend()
    plt.grid(axis="y")
    plt.show()

    print(f"time mean={np.mean(time)}")
    print(f"time std={np.std(time)}")
    print(f"time min={np.min(time)}")
    print(f"time max={np.max(time)}")
    plt.bar(bar_x, time, label="Lenght of scenario")
    ax = plt.gca()
    formatter = FuncFormatter(format_func)
    ax.yaxis.set_major_formatter(formatter)
    # this locates y-ticks at the hours
    ax.yaxis.set_major_locator(MultipleLocator(base=90))
    # ax.xaxis.set_major_locator(MultipleLocator(base=1))
    plt.grid(axis="y")
    plt.ylabel("time (minutes:seconds)")
    plt.xlabel("Scenario")
    plt.ylim([5*60, 15*60 + 30])

    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", help="Path to trained .pt model. Download ResNet version from https://ronin.cs.sfu.ca/", default=MODEL_PATH_DEFAULT)
    parser.add_argument(
        "--dataset", help="Path to dataset root. Download from https://ronin.cs.sfu.ca/", default=DATASET_ROOT_DEFAULT)
    args = parser.parse_args()
    print(f"Using model {args.model}")
    print(f"Using dataset root {args.dataset}")
    main(args)
