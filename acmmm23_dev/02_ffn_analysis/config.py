from copy import deepcopy
from yacs.config import CfgNode as CN

# Initialize Configuration Class
__C = CN(new_allowed=True)

# Load this variable at other scripts
cfg = __C

# ========= Set Configurations ========= #
__C.MAIN_PARAMS = CN(new_allowed=True)

# === Data Parameters === #
__C.DATA = CN(new_allowed=False)
__C.DATA.root_path = "/home/kyle/PycharmProjects/TransT_KYLE/acmmm23_dev/ffn_data"
__C.DATA.benchmark = "OTB100"

__C.DATA.OPTS = CN(new_allowed=False)
__C.DATA.OPTS.BENCHMARK_DATA_OBJ = CN(new_allowed=False)
__C.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_criterion = "iou"
__C.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_thresholds = [0.5]
__C.DATA.OPTS.BENCHMARK_DATA_OBJ.labeling_type = "one_hot"
__C.DATA.OPTS.BENCHMARK_DATA_OBJ.random_seed = 1234

__C.DATA.OPTS.TEST_DATASET = CN(new_allowed=False)
__C.DATA.OPTS.TEST_DATASET.overlap_criterion = \
    deepcopy(__C.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_criterion)
__C.DATA.OPTS.TEST_DATASET.overlap_thresholds = \
    deepcopy(__C.DATA.OPTS.BENCHMARK_DATA_OBJ.overlap_thresholds)
__C.DATA.OPTS.TEST_DATASET.labeling_type = \
    deepcopy(__C.DATA.OPTS.BENCHMARK_DATA_OBJ.labeling_type)

__C.DATA.SPLIT_RATIO = CN(new_allowed=False)
__C.DATA.SPLIT_RATIO.train = 0.7
__C.DATA.SPLIT_RATIO.validation = 0.1
__C.DATA.SPLIT_RATIO.test = 0.2

# === Training Parameters === #
__C.TRAIN = CN(new_allowed=False)
__C.TRAIN.random_seed = 1111

__C.TRAIN.num_epochs = 100
__C.TRAIN.batch_size = 512
__C.TRAIN.base_lr = 0.25
__C.TRAIN.weight_decay = 0.0

__C.TRAIN.VALIDATION = CN(new_allowed=False)
__C.TRAIN.VALIDATION.switch = True
__C.TRAIN.VALIDATION.epoch_interval = 5

__C.TRAIN.LOSS = CN(new_allowed=False)
__C.TRAIN.LOSS.type = "WCE"

__C.TRAIN.OPTIM = CN(new_allowed=False)
__C.TRAIN.OPTIM.type = "SGD"

__C.TRAIN.MODEL = CN(new_allowed=False)
__C.TRAIN.MODEL.type = "NN_statistics"
# __C.TRAIN.MODEL.type = "NN_dense_encoding"

__C.TRAIN.SCHEDULER = CN(new_allowed=False)
__C.TRAIN.SCHEDULER.switch = True
__C.TRAIN.SCHEDULER.type = "CosineAnnealingLr"

# === Loss Configurations === #
pass

# === Optimizer Configurations === #
__C.OPTIM = CN(new_allowed=True)

__C.OPTIM.SGD = CN(new_allowed=False)
__C.OPTIM.SGD.momentum = 0.9
__C.OPTIM.SGD.weight_decay = deepcopy(__C.TRAIN.weight_decay)

__C.OPTIM.Adam = CN(new_allowed=False)
__C.OPTIM.Adam.beta1 = 0.9
__C.OPTIM.Adam.beta2 = 0.999
__C.OPTIM.Adam.weight_decay = deepcopy(__C.TRAIN.weight_decay)

__C.OPTIM.Adagrad = CN(new_allowed=False)
__C.OPTIM.Adagrad.weight_decay = deepcopy(__C.TRAIN.weight_decay)

# === Model Configurations === #
__C.MODEL = CN(new_allowed=True)

__C.MODEL.INITIALIZATION = CN(new_allowed=False)
__C.MODEL.INITIALIZATION.switch = True
__C.MODEL.INITIALIZATION.distribution = "uniform"

__C.MODEL.NN_statistics = CN(new_allowed=False)
__C.MODEL.NN_statistics.dimensions = [768, 100, 32, 2]
__C.MODEL.NN_statistics.layers = ["fc", "fc", "fc"]
__C.MODEL.NN_statistics.hidden_activations = ["ReLU", "ReLU"]
__C.MODEL.NN_statistics.batchnorm_layers = [True, True]
__C.MODEL.NN_statistics.dropout_probs = [0.2, 0.2]
__C.MODEL.NN_statistics.final_activation = "softmax"

__C.MODEL.NN_dense_encoding = CN(new_allowed=False)
__C.MODEL.NN_dense_encoding.dimensions = [262144, 2048, 256, 32, 2]
__C.MODEL.NN_dense_encoding.layers = ["fc", "fc", "fc", "fc"]
# __C.MODEL.NN_statistics.hidden_activations = ["ReLU", "ReLU", "ReLU"]
__C.MODEL.NN_dense_encoding.hidden_activations = ["LeakyReLU", "LeakyReLU", "LeakyReLU"]
__C.MODEL.NN_dense_encoding.batchnorm_layers = [True, True, True]
__C.MODEL.NN_dense_encoding.dropout_probs = [0.2, 0.2, 0.2]
__C.MODEL.NN_dense_encoding.final_activation = "softmax"

# === Scheduler Configurations === #
__C.SCHEDULER = CN(new_allowed=True)

__C.SCHEDULER.MultiStepLR = CN(new_allowed=False)
__C.SCHEDULER.MultiStepLR.step = 5
__C.SCHEDULER.MultiStepLR.gamma = 0.8
__C.SCHEDULER.MultiStepLR.min_lr = deepcopy(__C.TRAIN.base_lr * 0.1)

__C.SCHEDULER.CosineAnnealingLr = CN(new_allowed=False)
__C.SCHEDULER.CosineAnnealingLr.eta_min = deepcopy(__C.TRAIN.base_lr * 0.1)
__C.SCHEDULER.CosineAnnealingLr.T_max = deepcopy(__C.TRAIN.num_epochs)


if __name__ == "__main__":
    pass
