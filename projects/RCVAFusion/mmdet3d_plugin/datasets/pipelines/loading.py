from typing import List, Optional, Union, Tuple
import os
import mmcv
import numpy as np

import mmengine
import torch
from mmcv.transforms.base import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type



@TRANSFORMS.register_module()
class LoadPointsFromVOD(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 modality: str,
                 load_dim: int = 8,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 norm_dim: Union[int, List[int]] =[],
                 use_color: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.norm_dim=norm_dim
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR','DEPTH']
        self.modality=modality
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:

        points = np.fromfile(pts_filename, dtype=np.float32)

        return points


    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        if self.modality=='lidar':
            pts_file_path = results['lidar_points']['lidar_path']
        elif self.modality=='radar':
            pts_file_path = results['radar_points']['radar_path']

        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)

        points = points[:, self.use_dim]

        attribute_dims = None

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str



@TRANSFORMS.register_module()
class LoadVisualPointsFromVOD(BaseTransform):
    def __init__(self,
                 coord_type: str,
                 modality: str,
                 load_dim: int = 8,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 norm_dim: Union[int, List[int]] =[],
                 use_color: bool = False,
                 backend_args: Optional[dict] = None,
                 virtual_points_root_path='data/VoD/vod_virtual_points',
                 ) -> None:
        self.norm_dim=norm_dim
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))

        assert coord_type in ['CAMERA', 'LIDAR','DEPTH']
        self.modality=modality
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.backend_args = backend_args
        self.root_split_path = virtual_points_root_path

    def get_virtual_point(self, idx):
        lidar_file = os.path.join(self.root_split_path, ('%s.npy' % idx))

        points = np.load(str(lidar_file), allow_pickle=True).item()
        virtual_points = points['virtual_points']
        gt_real_points = points['real_points']
        other_points = points['other_points']

        return virtual_points, gt_real_points, other_points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """

        pts_file_path = results['radar_points']['radar_path']

        path_idx = os.path.splitext(os.path.basename(pts_file_path))[0]

        virtual_points, gt_real_points, real_points = self.get_virtual_point(path_idx)  # xyz_3 + feat_4 + label_8
        if virtual_points is None:
            points = np.zeros([real_points.shape[0], 17 + 2])
            points[:, :7] = real_points
        else:
            points = np.zeros(
                [virtual_points.shape[0] + real_points.shape[0] + gt_real_points.shape[0], virtual_points.shape[1] + 2])

            points[:real_points.shape[0], :7] = real_points
            points[real_points.shape[0]:, :-2] = np.concatenate([gt_real_points, virtual_points])
            points[real_points.shape[0]:, -2] = 1
            points[real_points.shape[0]:, -1] = 1
            points[-virtual_points.shape[0]:, -1] = 0

        points = points[:, self.use_dim]

        attribute_dims = None

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str

@TRANSFORMS.register_module()
class LoadVirtualPointsFromTJ4D(BaseTransform):
    def __init__(self,
                 coord_type: str,
                 modality: str,
                 load_dim: int = 8,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 norm_dim: Union[int, List[int]] =[],
                 use_color: bool = False,
                 virtual_points_root_path='data/TJ4DRadSet/tj4d_virtual_points',
                 backend_args: Optional[dict] = None) -> None:
        self.norm_dim=norm_dim
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))

        assert coord_type in ['CAMERA', 'LIDAR','DEPTH']
        self.modality=modality
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.backend_args = backend_args
        self.root_split_path = virtual_points_root_path

    def get_virtual_point(self, idx):
        lidar_file = os.path.join(self.root_split_path, ('%s.npy' % idx))

        points = np.load(str(lidar_file), allow_pickle=True).item()
        virtual_points = points['virtual_points']
        gt_real_points = points['real_points']
        other_points = points['other_points']

        return virtual_points, gt_real_points, other_points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """

        pts_file_path = results['lidar_points']['lidar_path']

        path_idx = os.path.splitext(os.path.basename(pts_file_path))[0]

        virtual_points, gt_real_points, real_points = self.get_virtual_point(path_idx)  # xyz_3 + feat_4 + label_8
        if virtual_points is None:
            points = np.zeros([real_points.shape[0], 15 + 2])
            points[:, :5] = real_points
        else:
            points = np.zeros(
                [virtual_points.shape[0] + real_points.shape[0] + gt_real_points.shape[0], virtual_points.shape[1] + 2])

            points[:real_points.shape[0], :5] = real_points
            points[real_points.shape[0]:, :-2] = np.concatenate([gt_real_points, virtual_points])
            points[real_points.shape[0]:, -2] = 1
            points[real_points.shape[0]:, -1] = 1
            points[-virtual_points.shape[0]:, -1] = 0

        points = points[:, self.use_dim]

        attribute_dims = None

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str

@TRANSFORMS.register_module()
class LoadPointsFromTJ4D(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 8,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 norm_dim: Union[int, List[int]] =[],
                 use_color: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.norm_dim=norm_dim
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:

        points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results['lidar_points']['lidar_path']
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        if len(self.norm_dim)>0:
            points[:, self.norm_dim]=np.tanh(points[:, self.norm_dim])

        points = points[:, self.use_dim]

        attribute_dims = None


        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str