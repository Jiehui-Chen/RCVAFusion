import os

import projects.RCVAFusion.dataset_converter.TJ4D.TJ4DRadSet_converter as TJ4DRadSet
from projects.RCVAFusion.dataset_converter.TJ4D.update_infos_to_v2 import update_TJ4DRadSet_infos
from projects.RCVAFusion.dataset_converter.TJ4D.create_gt_database import create_groundtruth_database


def ourDatasetV2_dara_prep(
        root_path, prefix='TJ4DRadSet', remove_raw_infos=True):
    TJ4DRadSet.create_TJ4DRadSet_info_file(root_path)

    info_train_path = os.path.join(root_path, 'infos_train.pkl')
    info_valid_path = os.path.join(root_path, 'infos_valid.pkl')
    info_trainval_path = os.path.join(root_path, 'infos_trainval.pkl')
    info_test_path = os.path.join(root_path, 'infos_test.pkl')
    update_TJ4DRadSet_infos(pkl_path=info_train_path, out_dir=f'{prefix}_infos_train.pkl')
    update_TJ4DRadSet_infos(pkl_path=info_valid_path, out_dir=f'{prefix}_infos_valid.pkl')
    update_TJ4DRadSet_infos(pkl_path=info_trainval_path, out_dir=f'{prefix}_infos_trainval.pkl')
    update_TJ4DRadSet_infos(pkl_path=info_test_path, out_dir=f'{prefix}_infos_test.pkl')

    # create_groundtruth_database(
    #     dataset_class_name='TJ4DRadSetDataset',
    #     data_path=root_path,
    #     info_path=f'{prefix}_infos_train.pkl',
    #     info_prefix=prefix
    # )

    if remove_raw_infos:
        for info_path in [
                info_train_path, info_valid_path, info_trainval_path,
                info_test_path
        ]:
            if os.path.exists(info_path):
                os.remove(info_path)
                print(f'{info_path} removed.')

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    root_path = 'data/TJ4DRadSet'
    ourDatasetV2_dara_prep(root_path=root_path)
