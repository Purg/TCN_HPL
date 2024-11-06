import functools
import typing as tg

import numpy as np
from numpy import typing as npt

from tcn_hpl.data.vectorize._interface import Vectorize, FrameData

NUM_POSE_JOINTS = 22

class LocsAndConfs(Vectorize):
    """
    Previous manual approach to vectorization.

    Arguments:
        top_k: The number of top per-class examples to use in vector
            construction.
        num_classes: the number of classes in the object detector.
        use_joint_confs: use the confidence of each pose joint.
            (changes the length of the input vector, which needs to
            be manually updated if this flag changes.)
        use_pixel_norm: Normalize pixel coordinates by dividing by
            frame height and width, respectively. Normalized values 
            are between 0 and 1. Does not change input vector length.
        use_joint_obj_offsets: add abs(X and Y offsets) for between joints and
            each object.
            (changes the length of the input vector, which needs to
            be manually updated if this flag changes.)
        
    """

    def __init__(
        self,
        top_k: int = 1,
        num_classes: int = 7,
        use_joint_confs: bool = True,
        use_pixel_norm: bool = True,
        use_joint_obj_offsets: bool = False,
        background_idx: int = 0
    ):
        super().__init__()

        self._top_k = top_k
        self._num_classes = num_classes
        self._use_joint_confs = use_joint_confs
        self._use_pixel_norm = use_pixel_norm
        self._use_joint_obj_offsets = use_joint_obj_offsets
        self._background_idx = background_idx
    
    # Get the top "k" object indexes for each object
    @staticmethod
    def get_top_k_indexes_of_one_obj_type(f_dets, k, label_ind):
        """
        Find all instances of a label index in object detections.
        Then sort them and return the top K.
        Inputs:
        - object_dets:
        """
        labels = f_dets.labels
        scores = f_dets.scores
        # Get all labels of an obj type
        filtered_idxs = [i for i, e in enumerate(labels) if e == label_ind]
        if not filtered_idxs:
            return None
        filtered_scores = [scores[i] for i in filtered_idxs]
        # Sort labels by score values.
        sorted_inds = [i[1] for i in sorted(zip(filtered_scores, filtered_idxs))]
        return sorted_inds[:k]

    @staticmethod
    def append_vector(frame_feat, i, number):
        frame_feat[i] = number
        return frame_feat, i + 1
    
    def determine_vector_length(self, data: FrameData) -> int:
        #########################
        # Feature vector
        #########################
        # Length: pose confs * 22, pose X's * 22, pose Y's * 22,
        #         obj confs * num_objects(7 for M2), 
        #         obj X * num_objects(7 for M2), 
        #         obj Y * num_objects(7 for M2)
        #         obj W * num_objects(7 for M2)
        #         obj H * num_objects(7 for M2)
        #         casualty conf * 1
        vector_length = 0
        # Joint confidences
        if self._use_joint_confs:
            vector_length += NUM_POSE_JOINTS
        # X and Y for each joint
        vector_length += 2 * NUM_POSE_JOINTS
        # [Conf, X, Y, W, H] for k instances of each object class.
        vector_length = 5 * self._top_k * self._num_classes
        return vector_length


    def vectorize(self, data: FrameData) -> npt.NDArray[np.float32]:

        vector_len = self.determine_vector_length(data)
        frame_feat = np.zeros(vector_len, dtype=np.float32)
        vector_ind = 0
        if self._use_pixel_norm:
            W = data.size[0]
            H = data.size[1]
        else:
            W = 1
            H = 1
        f_dets = data.object_detections

        # Loop through all classes: populate obj conf, obj X, obj Y.
        # Assumption: class labels are [0, 1, 2,... num_classes-1].
        for obj_ind in range(0,self._num_classes):
            top_k_idxs = self.get_top_k_indexes_of_one_obj_type(f_dets, self._top_k, obj_ind)
            if top_k_idxs: # This is None if there were no detections to sort for this class
                for idx in top_k_idxs:
                    # Conf
                    frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, f_dets.scores[idx])
                    # X
                    frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, f_dets.boxes[idx][0] / W)
                    # Y
                    frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, f_dets.boxes[idx][1] / H)
                    # W
                    frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, f_dets.boxes[idx][2] / W)
                    # H
                    frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, f_dets.boxes[idx][3] / H)
            else:
                for _ in range(0, self._top_k * 5):
                    # 5 Zeros
                    frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, 0)
        
        f_poses = data.poses
        if f_poses:
            # Find most confident body detection
            confident_pose_idx = np.argmax(f_poses.scores)
            num_joints = f_poses.joint_positions.shape[1]
            frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, f_poses.scores[confident_pose_idx])

            for joint_ind in range(0, num_joints):
                # Conf
                if self._use_joint_confs:
                    frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, f_poses.joint_scores[confident_pose_idx][joint_ind])
                # X
                frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, f_poses.joint_positions[confident_pose_idx][joint_ind][0] / W)
                # Y
                frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, f_poses.joint_positions[confident_pose_idx][joint_ind][1] / H)
        else:
            num_joints = f_poses.joint_positions.shape[1]
            if self._use_joint_confs:
                rows_per_joint = 3
            else:
                rows_per_joint = 2
            for _ in range(num_joints * rows_per_joint + 1):
                frame_feat, vector_ind = self.append_vector(frame_feat, vector_ind, 0)
        
        assert vector_ind == vector_len

        return frame_feat
