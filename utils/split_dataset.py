import os
import numpy as np


if __name__ == '__main__':
    """
    Split images ids to train/eval set, save to txt files.
    """

    np.random.seed(20)

    base_dir = '/some/path/contains/your/data'
    scene_name = 'LLFF/trex'

    scene_dir = os.path.join(base_dir, scene_name)
    img_dir = os.path.join(scene_dir, 'images')

    N_img = len(os.listdir(img_dir))
    ids = np.arange(N_img)

    val_ids = ids[::8]
    train_ids = np.array([i for i in ids if i not in val_ids])

    np.savetxt(os.path.join(scene_dir, 'val_ids.txt'), val_ids, fmt='%d')
    np.savetxt(os.path.join(scene_dir, 'train_ids.txt'), train_ids, fmt='%d')

    print('Num images: ', N_img)
    print('Num train: ', len(train_ids))
    print('Num val: ', len(val_ids))
