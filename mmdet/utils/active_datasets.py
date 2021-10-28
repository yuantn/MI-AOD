import mmcv
import numpy as np


def get_X_L_0(cfg):
    # load dataset anns
    anns = load_ann_list(cfg.data.train.dataset.ann_file)
    # get all indexes
    X_all = np.arange(len(anns[0]) + len(anns[1]))
    # randomly select labeled set
    np.random.shuffle(X_all)
    X_L = X_all[:cfg.X_L_0_size].copy()
    X_U = X_all[-cfg.X_L_0_size:].copy()
    X_L.sort()
    X_U.sort()
    return X_L, X_U, X_all, anns


def create_X_L_file(cfg, X_L, anns, cycle):
    # split labeled set into 2007 and 2012
    X_L = [X_L[X_L < len(anns[0])], X_L[X_L >= len(anns[0])] - len(anns[0])]
    # create labeled ann files
    X_L_path = []
    for ann, X_L_single, year in zip(anns, X_L, ['07', '12']):
        save_folder = cfg.work_directory + '/cycle' + str(cycle)
        mmcv.mkdir_or_exist(save_folder)
        save_path = save_folder + '/trainval_X_L_' + year + '.txt'
        np.savetxt(save_path, ann[X_L_single], fmt='%s')
        X_L_path.append(save_path)
    # update cfg
    cfg.data.train.dataset.ann_file = X_L_path
    cfg.data.train.times = cfg.X_L_repeat
    return cfg


def create_X_U_file(cfg, X_U, anns, cycle):
    # split unlabeled set into 2007 and 2012
    X_U = [X_U[X_U < len(anns[0])], X_U[X_U >= len(anns[0])] - len(anns[0])]
    # create labeled ann files
    X_U_path = []
    for ann, X_U_single, year in zip(anns, X_U, ['07', '12']):
        save_folder = cfg.work_directory + '/cycle' + str(cycle)
        mmcv.mkdir_or_exist(save_folder)
        save_path = save_folder + '/trainval_X_U_' + year + '.txt'
        np.savetxt(save_path, ann[X_U_single], fmt='%s')
        X_U_path.append(save_path)
    # update cfg
    cfg.data.train.dataset.ann_file = X_U_path
    cfg.data.train.times = cfg.X_U_repeat
    return cfg


def load_ann_list(paths):
    anns = []
    for path in paths:
        anns.append(np.loadtxt(path, dtype='str'))
    return anns


def update_X_L(uncertainty, X_all, X_L, X_S_size):
    uncertainty = uncertainty.cpu().numpy()
    all_X_U = np.array(list(set(X_all) - set(X_L)))
    uncertainty_X_U = uncertainty[all_X_U]
    arg = uncertainty_X_U.argsort()
    X_S = all_X_U[arg[-X_S_size:]]
    X_L_next = np.concatenate((X_L, X_S))
    all_X_U_next = np.array(list(set(X_all) - set(X_L_next)))
    np.random.shuffle(all_X_U_next)
    X_U_next = all_X_U_next[:X_L_next.shape[0]]
    if X_L_next.shape[0] > X_U_next.shape[0]:
        np.random.shuffle(X_L_next)
        X_U_next = np.concatenate((X_U_next, X_L_next[:X_L_next.shape[0] - X_U_next.shape[0]]))
    X_L_next.sort()
    X_U_next.sort()
    return X_L_next, X_U_next
