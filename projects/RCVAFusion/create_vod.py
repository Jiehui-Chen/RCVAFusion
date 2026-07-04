import os
import projects.RCVAFusion.dataset_converter.VoD.VOD_converter as VOD
from projects.RCVAFusion.dataset_converter.VoD.update_infos_to_v2 import update_VOD_infos


def ourDatasetV2_dara_prep(root_path, prefix='VOD', remove_raw_infos=True):
    VOD.create_VOD_info_file(root_path)

    info_train_path = os.path.join(root_path, 'infos_train.pkl')
    info_valid_path = os.path.join(root_path, 'infos_valid.pkl')
    info_trainval_path = os.path.join(root_path, 'infos_trainval.pkl')
    info_test_path = os.path.join(root_path, 'infos_test.pkl')
    update_VOD_infos(pkl_path=info_train_path, out_dir=f'{prefix}_infos_train.pkl')
    update_VOD_infos(pkl_path=info_valid_path, out_dir=f'{prefix}_infos_valid.pkl')
    update_VOD_infos(pkl_path=info_trainval_path, out_dir=f'{prefix}_infos_trainval.pkl')
    update_VOD_infos(pkl_path=info_test_path, out_dir=f'{prefix}_infos_test.pkl')

    if remove_raw_infos:
        for info_path in [
                info_train_path, info_valid_path, info_trainval_path,
                info_test_path
        ]:
            if os.path.exists(info_path):
                os.remove(info_path)
                print(f'{info_path} removed.')
    # from projects.RCVAFusion.dataset_converter.VoD.create_gt_database import create_groundtruth_database
    # create_groundtruth_database(
    #     dataset_class_name='VODDataset',
    #     data_path=root_path,
    #     info_path=f'{prefix}_infos_train.pkl',
    #     info_prefix=prefix
    # )

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    root_path = 'data/VoD'
    ourDatasetV2_dara_prep(root_path=root_path)
