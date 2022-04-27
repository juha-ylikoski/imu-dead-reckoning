import enum
import math
import os


from ronin.data_ridi import RIDIGlobSpeedSequence
from ronin.transformations import RandomHoriRotate
from ronin.data_glob_speed import GlobSpeedSequence, StridedSequenceDataset


class DataSet(enum.Enum):
    RONIN = "RONIN"
    RIDI = "RIDI"

class DatasetMode(enum.Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    VAL = "VAL"

CACHE_PATH = ".cache"
DEFAULT_STEP_SIZE = 10
DEFAULT_WINDOW_SIZE = 200
DEFAULT_MAX_ORI_ERROR = 20.0


def get_dataset(root_dir, data_list, dataset: DataSet, mode: DatasetMode, **kwargs):
    """Load dataset from dateset directory

    Args:
        root_dir (str): path to dataset root
        data_list (list[str]): list of scenarios which to load
        dataset (DataSet): dataset type
        mode (DatasetMode): dataset mode

    Returns:
        torch.util.dataset: PyTorch dataset object
    """
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == DatasetMode.TRAIN:
        random_shift = DEFAULT_STEP_SIZE // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == DatasetMode.VAL:
        shuffle = True
    elif mode == DatasetMode.TEST:
        shuffle = False
        grv_only = True

    if dataset == DataSet.RONIN:
        seq_type = GlobSpeedSequence
    elif dataset == DataSet.RIDI:
        seq_type = RIDIGlobSpeedSequence

    dataset = StridedSequenceDataset(
        seq_type, root_dir, data_list, CACHE_PATH, DEFAULT_STEP_SIZE, DEFAULT_WINDOW_SIZE,
        random_shift=random_shift, transform=transforms,
        shuffle=shuffle, grv_only=grv_only, max_ori_error=DEFAULT_MAX_ORI_ERROR)

    # TODO What are these
    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset

