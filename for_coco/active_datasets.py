import mmcv
import numpy as np




def get_X_L_0(cfg):
    # load dataset anns
    anns = np.loadtxt(cfg.data.train.dataset.ann_file, dtype='str')
    # get all indexes
    X_all = np.arange(len(anns))
    # randomly select labeled set
    np.random.shuffle(X_all)
    X_L = X_all[:cfg.X_L_0_size].copy()
    X_U = X_all[cfg.X_L_0_size:cfg.X_L_0_size*2].copy()
    X_L.sort()
    X_U.sort()
    return X_L, X_U, X_all, anns



def create_X_L_file(cfg, X_L, anns, cycle):
    save_folder = cfg.work_directory + '/cycle' + str(cycle)
    mmcv.mkdir_or_exist(save_folder)
    X_L_path = save_folder + '/trainval_X_L' + '.txt'
    np.savetxt(X_L_path, anns[X_L], fmt='%s')
    # update cfg
    cfg.data.train.dataset.ann_file = X_L_path
    cfg.data.train.times = cfg.X_L_repeat
    return cfg


def create_X_U_file(cfg, X_U, anns, cycle):
    # create labeled ann files
    save_folder = cfg.work_directory + '/cycle' + str(cycle)
    mmcv.mkdir_or_exist(save_folder)
    X_U_path = save_folder + '/trainval_X_U' + '.txt'
    np.savetxt(X_U_path, anns[X_U], fmt='%s')
    # update cfg
    cfg.data.train.dataset.ann_file = X_U_path
    cfg.data.train.times = cfg.X_U_repeat
    return cfg


def create_selection_unlabeled_set(cfg, all_set, labeled_set, anns, cycle):
    all_unlabeled_set = np.array(list(set(all_set) - set(labeled_set)))
    np.random.shuffle(all_unlabeled_set)
    subset = all_unlabeled_set[:117266//cfg.subset_p]
    # create labeled ann files
    save_folder = cfg.work_directory + '/cycle' + str(cycle)
    mmcv.mkdir_or_exist(save_folder)
    save_path = save_folder + '/trainval_su' + '.txt'
    np.savetxt(save_path, anns[subset], fmt='%s')
    # update cfg
    cfg.data.test.ann_file = save_path
    return cfg, subset


def update_X_L(uncertainty, X_all, subset, X_L, X_S_size):
    uncertainty = uncertainty.cpu().numpy()
    arg = uncertainty.argsort()
    X_S = subset[arg[-X_S_size:]]

    X_L_next = np.concatenate((X_L, X_S))
    all_X_U_next = np.array(list(set(X_all) - set(X_L_next)))
    np.random.shuffle(all_X_U_next)
    X_U_next = all_X_U_next[:X_L_next.shape[0]]
    X_L_next.sort()
    X_U_next.sort()
    return X_L_next, X_U_next
