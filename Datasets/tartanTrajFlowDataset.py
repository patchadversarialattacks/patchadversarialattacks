import random

import numpy as np
import cv2
from torch.utils.data import Dataset
from os import listdir
import os
from .transformation import pos_quats2SEs, pose2motion, SEs2ses, SE2pos_quat, ses2poses_quat
from .utils import make_intrinsics_layer
import torch
import kornia.geometry as kgm

import kornia.utils as ku
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, rotation_matrix_to_quaternion
from os import mkdir
from os.path import isdir
from shutil import rmtree


def extract_traj_data(traj_data):
    dataset_idx, dataset_name, traj_name, traj_len, \
    img1_I0, img2_I0, intrinsic_I0, \
    img1_I1, img2_I1, intrinsic_I1, \
    img1_delta, img2_delta, \
    motions_gt, scale_gt, pose_quat_gt, patch_pose, mask, perspective = traj_data

    dataset_idx = dataset_idx.item()
    dataset_name = dataset_name[0]
    traj_name = traj_name[0]
    traj_len = traj_len.item()
    img1_I0 = img1_I0.squeeze(0)
    img2_I0 = img2_I0.squeeze(0)
    intrinsic_I0 = intrinsic_I0.squeeze(0)
    img1_I1 = img1_I1.squeeze(0)
    img2_I1 = img2_I1.squeeze(0)
    intrinsic_I1 = intrinsic_I1.squeeze(0)
    img1_delta = img1_delta.squeeze(0)
    img2_delta = img2_delta.squeeze(0)

    motions_gt = motions_gt.squeeze(0)
    scale_gt = scale_gt.squeeze(0)
    pose_quat_gt = pose_quat_gt.squeeze(0)

    mask = mask.squeeze(0)
    perspective = perspective.squeeze(0)

    return dataset_idx, dataset_name, traj_name, traj_len,\
           img1_I0, img2_I0, intrinsic_I0, \
           img1_I1, img2_I1, intrinsic_I1, \
           img1_delta, img2_delta, \
           motions_gt, scale_gt, pose_quat_gt, patch_pose, mask, perspective


def get_perspective(data_size, dst_points, perspective_padding=(0, 0)):

    h, w = data_size  # src size
    h = h + 2 * perspective_padding[0]
    w = w + 2 * perspective_padding[1]
    points_src = torch.FloatTensor([[
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
    ]])
    perspective_list = []
    for img_points_dst in dst_points:
        # the destination points are the region to crop corners
        points_dst = torch.FloatTensor(img_points_dst).reshape((1, -1, 2))
        perspective_list.append(kgm.get_perspective_transform(points_src, points_dst).squeeze(0))
    return torch.stack(perspective_list)


def pose_R_t(pose):
    R = pose[:, 0:3, 0:3]
    t = pose[:, 0:3, 3]
    return R, t


def poses_inverse(poses):
    pose_inv = torch.zeros_like(poses)
    R = poses[:, 0:3, 0:3]
    R_inv = R.transpose(dim0=1, dim1=2)
    t = poses[:, 0:3, 3]
    t_inv = - R_inv.bmm(t.unsqueeze(2)).squeeze(2)
    pose_inv[:, 0:3, 0:3] = R_inv
    pose_inv[:, 0:3, 3] = t_inv
    pose_inv[:, 3, 3] = 1

    return pose_inv


def cumulative_poses(poses):
    cumulative_poses = torch.zeros(poses.shape[0] + 1, poses.shape[1], poses.shape[2], device=poses.device, dtype=poses.dtype)
    curr_cumulative_pose = torch.eye(4, device=poses.device, dtype=poses.dtype)
    cumulative_poses[0] = curr_cumulative_pose
    for pose_idx, pose in enumerate(poses):
        curr_cumulative_pose = curr_cumulative_pose.mm(pose)
        cumulative_poses[pose_idx + 1] = curr_cumulative_pose
    return cumulative_poses


def rtvec_to_pose(rtvec):
    pose = torch.zeros(rtvec.shape[0], 4, 4, device=rtvec.device, dtype=rtvec.dtype)

    pose[:, 0:3, 0:3] = angle_axis_to_rotation_matrix(rtvec[:, 3:6])
    pose[:, 0:3, 3] = rtvec[:, 0:3]
    pose[:, 3, 3] = 1
    return pose


def pose_to_quat(pose):
    pose_np = pose.numpy()
    quat_list = []
    for p in pose_np:
        quat_list.append(SE2pos_quat(p))
    return np.array(quat_list)


def kitti2SE(data):
    data = data.reshape((3, 4))
    SE = np.eye(4)
    SE[0:3, 0:3] = data[:, 0:3]
    SE[0:3, 3] = data[:, -1]
    return SE


def kitti_traj2SE_matrices(kitti_traj):
    SEs = []
    for data in kitti_traj:
        data = data.reshape((3, 4))
        SE = np.eye(4)
        SE[0:3, 0:3] = data[:, 0:3]
        SE[0:3, 3]   = data[:, -1]
        SEs.append(SE)
    return SEs


class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder , posefile = None, transform = None,
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.imgfolder = imgfolder

        print('Find {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        if posefile is not None and posefile!="":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist)
            self.matrix = pose2motion(poses)
            self.motions     = SEs2ses(self.matrix).astype(np.float32)
            # self.motions = self.motions / self.pose_std
            assert(len(self.motions) == len(self.rgbfiles)) - 1
        else:
            self.motions = None

        self.N = len(self.rgbfiles) - 1

        # self.N = len(self.lines)
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        imgfile1 = self.rgbfiles[idx].strip()
        imgfile2 = self.rgbfiles[idx+1].strip()
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)

        res = {'img1': img1, 'img2': img2 }

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)

        if self.motions is None:
            return res
        else:
            res['motion'] = self.motions[idx]
            return res


class TrajFolderDatasetCustom(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, rootfolder, transform=None, data_size=(448, 640),
                 focalx=320.0/np.tan(np.pi/4.5), focaly=320.0/np.tan(np.pi/4.5), centerx=320.0, centery=240.0,
                 max_traj_len=100, max_dataset_traj_num=1000):
        print("TrajFolderDatasetCustom")

        print("using custom dataset, dataset intrinsics:")
        print("focalx:" + str(focalx) + " focaly:" + str(focaly) + " centerx:" + str(centerx) + " centery:" + str(centery))
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery
        self.transform = transform
        self.max_traj_len = max_traj_len
        self.max_dataset_traj_num = max_dataset_traj_num
        self.data_size = data_size

        self.img1_I0_list = []
        self.img2_I0_list = []
        self.intrinsic_I0_list = []
        self.img1_I1_list = []
        self.img2_I1_list = []
        self.intrinsic_I1_list = []

        self.motions = []
        self.scales = []
        self.poses_quat = []

        self.mask_list = []
        self.perspective_list = []

        tot_files_num = 0
        self.rootfolder = rootfolder
        rootfolder_files = listdir(rootfolder)
        dataset_idx = 0
        dataset_name = rootfolder.split('/')[-1]
        self.datasets_num = 1
        self.datasets_indices = [0]
        self.datasets_names = [dataset_name]
        self.traj_names = [ff for ff in rootfolder_files if os.path.isdir(rootfolder + '/' + ff)]
        self.traj_names.sort()
        self.N = len(self.traj_names)
        if self.N > max_dataset_traj_num:
            self.traj_names = self.traj_names[:max_dataset_traj_num]
            self.N = len(self.traj_names)
        self.traj_dataset_names = [dataset_name] * self.N
        self.traj_dataset_indices = [dataset_idx] * self.N
        self.traj_len = None

        traj_folders = [(rootfolder + '/' + ff) for ff in self.traj_names]

        for traj_folder in traj_folders:
            traj_len, img1_I0_tensor, img2_I0_tensor, intrinsic_I0_tensor, \
            img1_I1_tensor, img2_I1_tensor, intrinsic_I1_tensor, \
            motions, scales, poses_quat, perspective, mask_tensor = self.process_trajectory_folder(traj_folder)
            if self.traj_len is None:
                self.traj_len = traj_len
            else:
                assert self.traj_len == traj_len

            tot_files_num += traj_len

            self.img1_I0_list.append(img1_I0_tensor)
            self.img2_I0_list.append(img2_I0_tensor)
            self.intrinsic_I0_list.append(intrinsic_I0_tensor)
            self.img1_I1_list.append(img1_I1_tensor)
            self.img2_I1_list.append(img2_I1_tensor)
            self.intrinsic_I1_list.append(intrinsic_I1_tensor)

            self.motions.append(motions)
            self.scales.append(scales)
            self.poses_quat.append(poses_quat)

            self.perspective_list.append(perspective)
            self.mask_list.append(mask_tensor)

        print('Find {} image files from {} trajectories and {} datasets in root folder:{}'.format(tot_files_num, self.N,
                                                                                      self.datasets_num, rootfolder))
        print("trajectories found:")
        print(self.traj_names)

    def process_imgfiles(self, img_files):
        img1_list = []
        img2_list = []
        intrinsic_list = []
        for img1_idx, imgfile1 in enumerate(img_files[:-1]):
            imgfile1 = imgfile1.strip()
            imgfile2 = img_files[img1_idx + 1].strip()
            img1 = cv2.cvtColor(cv2.imread(imgfile1), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(imgfile2), cv2.COLOR_BGR2RGB)

            sample = {'img1': img1, 'img2': img2}

            h, w, _ = img1.shape
            intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
            sample['intrinsic'] = intrinsicLayer

            if self.transform:
                sample = self.transform(sample)

            img1_list.append(sample['img1'])
            img2_list.append(sample['img2'])
            intrinsic_list.append(sample['intrinsic'])

        return img1_list, img2_list, intrinsic_list

    def process_trajectory_folder(self, traj_folder):
        files = listdir(traj_folder)
        rgbfiles = [ff for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        I0files = [(traj_folder + '/' + ff) for ff in rgbfiles if (ff.startswith('I0'))]
        I0files.sort()
        I1files = [(traj_folder + '/' + ff) for ff in rgbfiles if (ff.startswith('I1'))]
        I1files.sort()
        traj_len = len(I0files)
        if traj_len > self.max_traj_len:
            I0files = I0files[:self.max_traj_len]
            I1files = I1files[:self.max_traj_len]
            traj_len = len(I0files)

        img1_I0_traj_list, img2_I0_traj_list, intrinsic_I0_traj_list = self.process_imgfiles(I0files)
        img1_I1_traj_list, img2_I1_traj_list, intrinsic_I1_traj_list = self.process_imgfiles(I1files)

        img1_I0_tensor = torch.stack(img1_I0_traj_list)
        img2_I0_tensor = torch.stack(img2_I0_traj_list)
        intrinsic_I0_tensor = torch.stack(intrinsic_I0_traj_list)
        img1_I1_tensor = torch.stack(img1_I1_traj_list)
        img2_I1_tensor = torch.stack(img2_I1_traj_list)
        intrinsic_I1_tensor = torch.stack(intrinsic_I1_traj_list)


        posefile = traj_folder + '/' + 'pose_file.csv'
        assert os.path.isfile(posefile)
        gt_poses = np.loadtxt(posefile, delimiter=',').astype(np.float32)
        if gt_poses.shape[0] > self.max_traj_len:
            gt_poses = gt_poses[:self.max_traj_len]

        gt_poses_shifted = np.array(kitti_traj2SE_matrices(gt_poses))
        matrix = pose2motion(gt_poses_shifted[:, 0:3])
        motions = SEs2ses(matrix).astype(np.float32)
        scales = torch.tensor(np.linalg.norm(motions[:, :3], axis=1))
        poses_quat = ses2poses_quat(motions)
        assert (len(motions) == len(I0files) - 1)


        mask_coords_path = traj_folder + '/' + 'mask_coords.csv'
        assert os.path.isfile(mask_coords_path)
        mask_coords = np.genfromtxt(mask_coords_path, delimiter=',')
        # we want point order topLeft -> clockwise, but have bl->br->tl->tr, so:
        points = [np.array([[pts[4], pts[5], pts[6], pts[7]], [pts[2], pts[3], pts[0], pts[1]]], dtype=np.int32)
                  for pts in mask_coords]
        if len(points) > self.max_traj_len:
            points = points[:self.max_traj_len]

        perspective = get_perspective(self.data_size, points)

        mask_files = [ff for ff in files if (ff.startswith('patch_mask') and ff.endswith('.npy'))]
        mask_files.sort()
        if len(mask_files) > self.max_traj_len:
            mask_files = mask_files[:self.max_traj_len]
        assert (len(mask_files) == traj_len)
        mask_list = [np.load((traj_folder + '/' + ff)) for ff in mask_files]
        mask_list = [np.flip(mask.reshape((self.data_size[0], self.data_size[1], -1)), 0).copy() for mask in mask_list]
        mask_list = [ku.image_to_tensor(mask).ge(0.5).repeat(3, 1, 1) for mask in mask_list]

        mask_tensor = torch.stack(mask_list)

        return traj_len, img1_I0_tensor, img2_I0_tensor, intrinsic_I0_tensor, \
               img1_I1_tensor, img2_I1_tensor, intrinsic_I1_tensor, \
               motions, scales, poses_quat, perspective, mask_tensor

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.traj_dataset_indices[idx], self.traj_dataset_names[idx], self.traj_names[idx], \
               self.img1_I0_list[idx], self.img2_I0_list[idx], self.intrinsic_I0_list[idx], \
               self.img1_I1_list[idx], self.img2_I1_list[idx], self.intrinsic_I1_list[idx], \
               self.motions[idx], self.scales[idx], self.poses_quat[idx], \
               self.mask_list[idx], self.perspective_list[idx]


class MultiTrajFolderDatasetCustom(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, rootfolder, processed_data_folder="", preprocessed_data=False,
                 transform=None, data_size=(448, 640),
                 focalx=320.0 / np.tan(np.pi / 4.5), focaly=320.0 / np.tan(np.pi / 4.5), centerx=320.0, centery=240.0,
                 max_traj_len=100, max_dataset_traj_num=100, max_traj_datasets=10, folder_indices_list=None,
                 perspective_padding=(0, 0)):

        print("using custom dataset, dataset intrinsics:")
        print("focalx:" + str(focalx) + " focaly:" + str(focaly) + " centerx:" + str(centerx) + " centery:" + str(
            centery))
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery
        self.transform = transform
        self.max_traj_len = max_traj_len
        self.max_dataset_traj_num = max_dataset_traj_num
        self.data_size = data_size
        self.perspective_padding = perspective_padding

        self.rootfolder = rootfolder
        rootfolder_files = listdir(rootfolder)

        if processed_data_folder is None:
            processed_data_folder = rootfolder + '_processed'
        self.processed_data_folder = processed_data_folder

        if not isdir(processed_data_folder):
            preprocessed_data = False
            mkdir(processed_data_folder)
            print("processing data folder: " + str(processed_data_folder))
        elif not preprocessed_data:
            rmtree(processed_data_folder)
            mkdir(processed_data_folder)
            print("processing data folder, previously processed data will be deleted: " + str(processed_data_folder))
        else:
            print("using preprocessed data folder: " + str(processed_data_folder))

        self.datasets_names = [ff for ff in rootfolder_files if os.path.isdir(rootfolder + '/' + ff)]
        self.datasets_names.sort()
        self.datasets_num = len(self.datasets_names)
        if self.datasets_num > max_traj_datasets:
            self.datasets_names = self.datasets_names[:max_traj_datasets]
            self.datasets_num = len(self.datasets_names)
        self.datasets_indices = list(range(self.datasets_num))
        if folder_indices_list is not None:
            self.datasets_names = [self.datasets_names[idx] for idx in folder_indices_list]
            self.datasets_num = len(self.datasets_names)
            self.datasets_indices = list(range(self.datasets_num))
        self.N = 0
        self.traj_names = []
        self.traj_dataset_names = []
        self.traj_dataset_indices = []
        self.processed_traj_folders = []
        self.traj_len = None
        tot_files_num = 0
        for dataset_idx, dataset_name in enumerate(self.datasets_names):
            dataset_folder = rootfolder + '/' + dataset_name
            processed_dataset_folder = processed_data_folder + '/' + dataset_name
            if not preprocessed_data:
                mkdir(processed_dataset_folder)
            dataset_files = listdir(dataset_folder)
            dataset_traj_names = [ff for ff in dataset_files if os.path.isdir(dataset_folder + '/' + ff)]
            dataset_traj_names.sort()
            dataset_size = len(dataset_traj_names)
            if dataset_size > max_dataset_traj_num:
                dataset_traj_names = dataset_traj_names[:max_dataset_traj_num]
                dataset_size = len(dataset_traj_names)
            dataset_traj_folders = [(dataset_folder + '/' + ff) for ff in dataset_traj_names]
            processed_dataset_traj_folders = [(processed_dataset_folder + '/' + ff) for ff in dataset_traj_names]
            if not preprocessed_data:
                [mkdir(processed_traj_folder) for processed_traj_folder in processed_dataset_traj_folders]
            self.N += dataset_size
            self.traj_names += dataset_traj_names
            self.traj_dataset_names += [dataset_name] * dataset_size
            self.traj_dataset_indices += [dataset_idx] * dataset_size
            for traj_idx, traj_folder in enumerate(dataset_traj_folders):
                processed_traj_folder = processed_dataset_traj_folders[traj_idx]

                traj_len = self.process_and_save_trajectory_folder(traj_folder, processed_traj_folder, preprocessed_data)
                if self.traj_len is None:
                    self.traj_len = traj_len
                else:
                    assert self.traj_len == traj_len
                tot_files_num += traj_len

            self.processed_traj_folders += processed_dataset_traj_folders


        print('Find {} trajectories within {} datasets in root folder:{}'.format(self.N, self.datasets_num, rootfolder))
        print('Each trajectory contains {} images, and {} images in total within {} trajectories'.format(self.traj_len, tot_files_num, self.N))

        print("trajectories found:")
        print(self.traj_names)

    def process_imgfiles(self, img_files):
        img1_list = []
        img2_list = []
        intrinsic_list = []
        for img1_idx, imgfile1 in enumerate(img_files[:-1]):
            imgfile1 = imgfile1.strip()
            imgfile2 = img_files[img1_idx + 1].strip()
            img1 = cv2.cvtColor(cv2.imread(imgfile1), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(imgfile2), cv2.COLOR_BGR2RGB)

            sample = {'img1': img1, 'img2': img2}

            h, w, _ = img1.shape
            intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
            sample['intrinsic'] = intrinsicLayer

            if self.transform:
                sample = self.transform(sample)

            img1_list.append(sample['img1'])
            img2_list.append(sample['img2'])
            intrinsic_list.append(sample['intrinsic'])

        return img1_list, img2_list, intrinsic_list

    def process_and_save_trajectory_folder(self, traj_folder, processed_traj_folder, preprocessed_data):
        traj_len, img1_I0, img2_I0, intrinsic_I0, img1_I1, img2_I1, intrinsic_I1, \
        motions, scales, poses_quat, patch_rel_pose, perspective, mask = self.process_trajectory_folder(traj_folder, preprocessed_data)
        if not preprocessed_data:
            torch.save(img1_I0, processed_traj_folder + '/img1_I0.pt')
            torch.save(img2_I0, processed_traj_folder + '/img2_I0.pt')
            torch.save(intrinsic_I0, processed_traj_folder + '/intrinsic_I0.pt')
            torch.save(img1_I1, processed_traj_folder + '/img1_I1.pt')
            torch.save(img2_I1, processed_traj_folder + '/img2_I1.pt')
            torch.save(intrinsic_I1, processed_traj_folder + '/intrinsic_I1.pt')
            torch.save(motions, processed_traj_folder + '/motions.pt')
            torch.save(scales, processed_traj_folder + '/scales.pt')
            torch.save(poses_quat, processed_traj_folder + '/poses_quat.pt')
            torch.save(patch_rel_pose, processed_traj_folder + '/patch_rel_pose.pt')
            torch.save(perspective, processed_traj_folder + '/perspective.pt')
            torch.save(mask, processed_traj_folder + '/mask.pt')

        del img1_I0
        del img2_I0
        del intrinsic_I0
        del img1_I1
        del img2_I1
        del intrinsic_I1
        del motions
        del scales
        del poses_quat
        del patch_rel_pose
        del mask
        del perspective

        return traj_len

    def load_processed_trajectory_folder(self, processed_traj_folder):
        img1_I0 = torch.load(processed_traj_folder + '/img1_I0.pt')
        img2_I0 = torch.load(processed_traj_folder + '/img2_I0.pt')
        intrinsic_I0 = torch.load(processed_traj_folder + '/intrinsic_I0.pt')
        img1_I1 = torch.load(processed_traj_folder + '/img1_I1.pt')
        img2_I1 = torch.load(processed_traj_folder + '/img2_I1.pt')
        intrinsic_I1 = torch.load(processed_traj_folder + '/intrinsic_I1.pt')
        motions = torch.load(processed_traj_folder + '/motions.pt')
        scales = torch.load(processed_traj_folder + '/scales.pt')
        poses_quat = torch.load(processed_traj_folder + '/poses_quat.pt')
        patch_rel_pose = torch.load(processed_traj_folder + '/patch_rel_pose.pt')
        perspective = torch.load(processed_traj_folder + '/perspective.pt')
        mask = torch.load(processed_traj_folder + '/mask.pt')

        img1_delta = (img1_I1.clone().detach() - img1_I0).detach()
        img2_delta = (img2_I1.clone().detach() - img2_I0).detach()

        return img1_I0, img2_I0, intrinsic_I0, img1_I1, img2_I1, intrinsic_I1, img1_delta, img2_delta, \
        motions, scales, poses_quat, patch_rel_pose, perspective, mask

    def process_trajectory_folder(self, traj_folder, preprocessed_data):
        files = listdir(traj_folder)
        rgbfiles = [ff for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        I0files = [(traj_folder + '/' + ff) for ff in rgbfiles if (ff.startswith('I0'))]
        I0files.sort()
        I1files = [(traj_folder + '/' + ff) for ff in rgbfiles if (ff.startswith('I1'))]
        I1files.sort()
        traj_len = len(I0files)
        if traj_len > self.max_traj_len:
            I0files = I0files[:self.max_traj_len]
            I1files = I1files[:self.max_traj_len]
            traj_len = len(I0files)
        if preprocessed_data:
            return traj_len, None, None, None, \
                   None, None, None, \
                   None, None, None, None, None, None

        img1_I0_traj_list, img2_I0_traj_list, intrinsic_I0_traj_list = self.process_imgfiles(I0files)
        img1_I1_traj_list, img2_I1_traj_list, intrinsic_I1_traj_list = self.process_imgfiles(I1files)

        img1_I0_tensor = torch.stack(img1_I0_traj_list)
        img2_I0_tensor = torch.stack(img2_I0_traj_list)
        intrinsic_I0_tensor = torch.stack(intrinsic_I0_traj_list)
        img1_I1_tensor = torch.stack(img1_I1_traj_list)
        img2_I1_tensor = torch.stack(img2_I1_traj_list)
        intrinsic_I1_tensor = torch.stack(intrinsic_I1_traj_list)

        posefile = traj_folder + '/' + 'pose_file.csv'
        assert os.path.isfile(posefile)
        gt_poses = np.loadtxt(posefile, delimiter=',').astype(np.float32)
        if gt_poses.shape[0] > self.max_traj_len:
            gt_poses = gt_poses[:self.max_traj_len]

        gt_poses_shifted = np.array(kitti_traj2SE_matrices(gt_poses))
        matrix = pose2motion(gt_poses_shifted[:, 0:3])
        motions = SEs2ses(matrix).astype(np.float32)
        scales = torch.tensor(np.linalg.norm(motions[:, :3], axis=1))
        poses_quat = ses2poses_quat(motions)
        assert (len(motions) == len(I0files) - 1)

        patch_rel_posefile = traj_folder + '/' + 'patch_pose_VO.csv'
        assert os.path.isfile(patch_rel_posefile)
        patch_rel_pose = np.loadtxt(patch_rel_posefile, delimiter=',').astype(np.float32)
        patch_rel_pose = torch.tensor([patch_rel_pose[3], patch_rel_pose[7], patch_rel_pose[11]])

        mask_coords_path = traj_folder + '/' + 'mask_coords.csv'
        assert os.path.isfile(mask_coords_path)
        mask_coords = np.genfromtxt(mask_coords_path, delimiter=',')
        # we want point order topLeft -> clockwise, but have bl->br->tl->tr, so:
        points = [np.array([[pts[4], pts[5], pts[6], pts[7]], [pts[2], pts[3], pts[0], pts[1]]], dtype=np.int32)
                  for pts in mask_coords]
        if len(points) > self.max_traj_len:
            points = points[:self.max_traj_len]

        perspective = get_perspective(self.data_size, points, self.perspective_padding)

        mask_files = [ff for ff in files if (ff.startswith('patch_mask') and ff.endswith('.npy'))]
        mask_files.sort()
        if len(mask_files) > self.max_traj_len:
            mask_files = mask_files[:self.max_traj_len]
        assert (len(mask_files) == traj_len)
        mask_list = [np.load((traj_folder + '/' + ff)) for ff in mask_files]
        mask_list = [np.flip(mask.reshape((self.data_size[0], self.data_size[1], -1)), 0).copy() for mask in mask_list]
        mask_list = [ku.image_to_tensor(mask).ge(0.5).repeat(3, 1, 1) for mask in mask_list]

        mask_tensor = torch.stack(mask_list)

        return traj_len, img1_I0_tensor, img2_I0_tensor, intrinsic_I0_tensor, \
               img1_I1_tensor, img2_I1_tensor, intrinsic_I1_tensor, \
               motions, scales, poses_quat, patch_rel_pose, perspective, mask_tensor

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        processed_traj_folder = self.processed_traj_folders[idx]
        img1_I0, img2_I0, intrinsic_I0, img1_I1, img2_I1, intrinsic_I1, img1_delta, img2_delta, \
        motions, scales, poses_quat, patch_rel_pose, perspective, mask = self.load_processed_trajectory_folder(processed_traj_folder)

        return self.traj_dataset_indices[idx], self.traj_dataset_names[idx], self.traj_names[idx], self.traj_len, \
               img1_I0, img2_I0, intrinsic_I0, img1_I1, img2_I1, intrinsic_I1, img1_delta, img2_delta, \
               motions, scales, poses_quat, patch_rel_pose, mask, perspective


class MultiTrajFolderDatasetRealData(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, rootfolder, processed_data_folder="", preprocessed_data=False,
                 transform=None, data_size=(448, 640),
                 focalx=320.0 / np.tan(np.pi / 4.5), focaly=320.0 / np.tan(np.pi / 4.5), centerx=320.0, centery=240.0,
                 max_traj_len=100, max_dataset_traj_num=100, max_traj_datasets=10, folder_indices_list=None,
                 perspective_padding=(0, 0)):

        print("using custom dataset, dataset intrinsics:")
        print("focalx:" + str(focalx) + " focaly:" + str(focaly) + " centerx:" + str(centerx) + " centery:" + str(
            centery))
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery
        self.transform = transform
        self.max_traj_len = max_traj_len
        self.max_dataset_traj_num = max_dataset_traj_num
        self.data_size = data_size
        self.perspective_padding = perspective_padding

        self.rootfolder = rootfolder
        rootfolder_files = listdir(rootfolder)

        if processed_data_folder is None:
            processed_data_folder = rootfolder + '_processed'
        self.processed_data_folder = processed_data_folder

        if not isdir(processed_data_folder):
            preprocessed_data = False
            mkdir(processed_data_folder)
            print("processing data folder: " + str(processed_data_folder))
        elif not preprocessed_data:
            rmtree(processed_data_folder)
            mkdir(processed_data_folder)
            print("processing data folder, previously processed data will be deleted: " + str(processed_data_folder))
        else:
            print("using preprocessed data folder: " + str(processed_data_folder))

        self.datasets_num = max_traj_datasets
        self.datasets_indices = list(range(self.datasets_num))
        self.datasets_names = [str(idx) for idx in self.datasets_indices]
        if folder_indices_list is not None:
            self.datasets_names = [self.datasets_names[idx] for idx in folder_indices_list]
            self.datasets_num = len(self.datasets_names)
            self.datasets_indices = list(range(self.datasets_num))
        self.N = 0
        self.traj_names = []
        self.traj_dataset_names = []
        self.traj_dataset_indices = []
        self.processed_traj_folders = []
        self.traj_len = max_traj_len
        traj_len_list = []
        traj_folders = [ff for ff in rootfolder_files if os.path.isdir(rootfolder + '/' + ff)]
        traj_folders.sort()
        for traj_idx, traj_folder in enumerate(traj_folders):
            files = listdir(rootfolder + '/' + traj_folder)
            rgbfiles = [ff for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
            I0files = [(traj_folder + '/' + ff) for ff in rgbfiles if (ff.startswith('I0'))]
            I0files.sort()
            traj_len = len(I0files)
            traj_len_list.append(traj_len)
        self.traj_len = np.min(traj_len_list)
        if self.max_traj_len < self.traj_len:
            self.traj_len = max_traj_len
        else:
            self.max_traj_len = self.traj_len

        if preprocessed_data:
            # self.processed_traj_folders = None
            tot_files_num = 0
            for dataset_idx, dataset_name in enumerate(self.datasets_names):
                processed_dataset_folder = processed_data_folder + '/' + dataset_name
                processed_dataset_files = listdir(processed_dataset_folder)
                dataset_traj_names = [ff for ff in processed_dataset_files if os.path.isdir(processed_dataset_folder + '/' + ff)]
                dataset_traj_names.sort()
                dataset_size = len(dataset_traj_names)
                if dataset_size > max_dataset_traj_num:
                    dataset_traj_names = dataset_traj_names[:max_dataset_traj_num]
                    dataset_size = len(dataset_traj_names)
                processed_dataset_traj_folders = [(processed_dataset_folder + '/' + ff) for ff in dataset_traj_names]
                self.N += dataset_size
                self.traj_names += dataset_traj_names
                self.traj_dataset_names += [dataset_name] * dataset_size
                self.traj_dataset_indices += [dataset_idx] * dataset_size
                self.processed_traj_folders += processed_dataset_traj_folders
                tot_files_num += self.traj_len * len(processed_dataset_traj_folders)

        else:
            tot_files_num = 0
            random.shuffle(traj_folders)
            datasets_trajs = [traj_folders[i::self.datasets_num] for i in range(self.datasets_num)]
            for dataset_idx, dataset_name in enumerate(self.datasets_names):
                processed_dataset_folder = processed_data_folder + '/' + dataset_name
                mkdir(processed_dataset_folder)
                dataset_traj_names = datasets_trajs[dataset_idx]
                dataset_traj_names.sort()
                dataset_size = len(dataset_traj_names)
                if dataset_size > max_dataset_traj_num:
                    dataset_traj_names = dataset_traj_names[:max_dataset_traj_num]
                    dataset_size = len(dataset_traj_names)
                dataset_traj_folders = [(rootfolder + '/' + ff) for ff in dataset_traj_names]
                processed_dataset_traj_folders = [(processed_dataset_folder + '/' + ff) for ff in dataset_traj_names]
                [mkdir(processed_traj_folder) for processed_traj_folder in processed_dataset_traj_folders]
                self.N += dataset_size
                self.traj_names += dataset_traj_names
                self.traj_dataset_names += [dataset_name] * dataset_size
                self.traj_dataset_indices += [dataset_idx] * dataset_size
                for traj_idx, traj_folder in enumerate(dataset_traj_folders):
                    processed_traj_folder = processed_dataset_traj_folders[traj_idx]
                    traj_len = self.process_and_save_trajectory_folder(traj_folder, processed_traj_folder, preprocessed_data)
                    if self.traj_len is None:
                        self.traj_len = traj_len
                    else:
                        assert self.traj_len == traj_len
                self.processed_traj_folders += processed_dataset_traj_folders
                tot_files_num += self.traj_len * len(processed_dataset_traj_folders)

        print('processed data folder:{}, contains {} trajectories within {} datasets'.format(processed_data_folder, self.N, self.datasets_num))
        print('Each trajectory contains {} images, and {} images in total within {} trajectories'.format(self.traj_len, tot_files_num, self.N))

        print("trajectories found:")
        print(self.traj_names)

    def process_imgfiles(self, img_files):
        img1_list = []
        img2_list = []
        intrinsic_list = []
        for img1_idx, imgfile1 in enumerate(img_files[:-1]):
            imgfile1 = imgfile1.strip()
            imgfile2 = img_files[img1_idx + 1].strip()
            img1 = cv2.cvtColor(cv2.imread(imgfile1), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(imgfile2), cv2.COLOR_BGR2RGB)

            sample = {'img1': img1, 'img2': img2}

            h, w, _ = img1.shape
            intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
            sample['intrinsic'] = intrinsicLayer

            if self.transform:
                sample = self.transform(sample)

            img1_list.append(sample['img1'])
            img2_list.append(sample['img2'])
            intrinsic_list.append(sample['intrinsic'])

        return img1_list, img2_list, intrinsic_list

    def process_and_save_trajectory_folder(self, traj_folder, processed_traj_folder, preprocessed_data):
        traj_len, img1_I0, img2_I0, intrinsic_I0, img1_I1, img2_I1, intrinsic_I1, \
        motions, scales, poses_quat, patch_rel_pose, perspective, mask = self.process_trajectory_folder(traj_folder, preprocessed_data)
        if not preprocessed_data:
            torch.save(img1_I0, processed_traj_folder + '/img1_I0.pt')
            torch.save(img2_I0, processed_traj_folder + '/img2_I0.pt')
            torch.save(intrinsic_I0, processed_traj_folder + '/intrinsic_I0.pt')
            torch.save(img1_I1, processed_traj_folder + '/img1_I1.pt')
            torch.save(img2_I1, processed_traj_folder + '/img2_I1.pt')
            torch.save(intrinsic_I1, processed_traj_folder + '/intrinsic_I1.pt')
            torch.save(motions, processed_traj_folder + '/motions.pt')
            torch.save(scales, processed_traj_folder + '/scales.pt')
            torch.save(poses_quat, processed_traj_folder + '/poses_quat.pt')
            torch.save(patch_rel_pose, processed_traj_folder + '/patch_rel_pose.pt')
            torch.save(perspective, processed_traj_folder + '/perspective.pt')
            torch.save(mask, processed_traj_folder + '/mask.pt')

        del img1_I0
        del img2_I0
        del intrinsic_I0
        del img1_I1
        del img2_I1
        del intrinsic_I1
        del motions
        del scales
        del poses_quat
        del patch_rel_pose
        del mask
        del perspective

        return traj_len

    def load_processed_trajectory_folder(self, processed_traj_folder):
        img1_I0 = torch.load(processed_traj_folder + '/img1_I0.pt')
        img2_I0 = torch.load(processed_traj_folder + '/img2_I0.pt')
        intrinsic_I0 = torch.load(processed_traj_folder + '/intrinsic_I0.pt')
        img1_I1 = torch.load(processed_traj_folder + '/img1_I1.pt')
        img2_I1 = torch.load(processed_traj_folder + '/img2_I1.pt')
        intrinsic_I1 = torch.load(processed_traj_folder + '/intrinsic_I1.pt')
        motions = torch.load(processed_traj_folder + '/motions.pt')
        scales = torch.load(processed_traj_folder + '/scales.pt')
        poses_quat = torch.load(processed_traj_folder + '/poses_quat.pt')
        patch_rel_pose = torch.load(processed_traj_folder + '/patch_rel_pose.pt')
        perspective = torch.load(processed_traj_folder + '/perspective.pt')
        mask = torch.load(processed_traj_folder + '/mask.pt')

        img1_delta = (img1_I1.clone().detach() - img1_I0).detach()
        img2_delta = (img2_I1.clone().detach() - img2_I0).detach()

        if img1_I0.shape[0] > self.traj_len - 1:
            img1_I0 = img1_I0[:self.traj_len - 1]
            img2_I0 = img2_I0[:self.traj_len - 1]
            intrinsic_I0 = intrinsic_I0[:self.traj_len - 1]
            img1_I1 = img1_I1[:self.traj_len - 1]
            img2_I1 = img2_I1[:self.traj_len - 1]
            intrinsic_I1 = intrinsic_I1[:self.traj_len]
            motions = motions[:self.traj_len - 1]
            scales = scales[:self.traj_len - 1]
            poses_quat = poses_quat[:self.traj_len]
            perspective = perspective[:self.traj_len]
            mask = mask[:self.traj_len]
            img1_delta = img1_delta[:self.traj_len - 1]
            img2_delta = img2_delta[:self.traj_len - 1]

        return img1_I0, img2_I0, intrinsic_I0, img1_I1, img2_I1, intrinsic_I1, img1_delta, img2_delta, \
        motions, scales, poses_quat, patch_rel_pose, perspective, mask

    def process_trajectory_folder(self, traj_folder, preprocessed_data):
        files = listdir(traj_folder)
        rgbfiles = [ff for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        I0files = [(traj_folder + '/' + ff) for ff in rgbfiles if (ff.startswith('I0'))]
        I0files.sort()
        I1files = [(traj_folder + '/' + ff) for ff in rgbfiles if (ff.startswith('I1'))]
        I1files.sort()
        traj_len = len(I0files)
        if traj_len > self.max_traj_len:
            I0files = I0files[:self.max_traj_len]
            I1files = I1files[:self.max_traj_len]
            traj_len = len(I0files)
        if preprocessed_data:
            return traj_len, None, None, None, \
                   None, None, None, \
                   None, None, None, None, None, None

        img1_I0_traj_list, img2_I0_traj_list, intrinsic_I0_traj_list = self.process_imgfiles(I0files)
        img1_I1_traj_list, img2_I1_traj_list, intrinsic_I1_traj_list = self.process_imgfiles(I1files)

        img1_I0_tensor = torch.stack(img1_I0_traj_list)
        img2_I0_tensor = torch.stack(img2_I0_traj_list)
        intrinsic_I0_tensor = torch.stack(intrinsic_I0_traj_list)
        img1_I1_tensor = torch.stack(img1_I1_traj_list)
        img2_I1_tensor = torch.stack(img2_I1_traj_list)
        intrinsic_I1_tensor = torch.stack(intrinsic_I1_traj_list)

        posefile = traj_folder + '/' + 'pose_file.csv'
        assert os.path.isfile(posefile)
        gt_poses = np.loadtxt(posefile, delimiter=',').astype(np.float32)
        if gt_poses.shape[0] > self.max_traj_len:
            gt_poses = gt_poses[:self.max_traj_len]

        gt_poses_shifted = np.array(kitti_traj2SE_matrices(gt_poses))
        matrix = pose2motion(gt_poses_shifted[:, 0:3])
        motions = SEs2ses(matrix).astype(np.float32)
        scales = torch.tensor(np.linalg.norm(motions[:, :3], axis=1))
        poses_quat = ses2poses_quat(motions)
        assert (len(motions) == len(I0files) - 1)

        patch_rel_posefile = traj_folder + '/' + 'patch_pose_VO.csv'
        assert os.path.isfile(patch_rel_posefile)
        patch_rel_pose = np.loadtxt(patch_rel_posefile, delimiter=',').astype(np.float32)
        patch_rel_pose = torch.tensor([patch_rel_pose[3], patch_rel_pose[7], patch_rel_pose[11]])

        mask_coords_path = traj_folder + '/' + 'mask_coords.csv'
        assert os.path.isfile(mask_coords_path)
        mask_coords = np.genfromtxt(mask_coords_path, delimiter=',')
        # we want point order topLeft -> clockwise, but have bl->br->tl->tr, so:
        points = [np.array([[pts[4], pts[5], pts[6], pts[7]], [pts[2], pts[3], pts[0], pts[1]]], dtype=np.int32)
                  for pts in mask_coords]
        if len(points) > self.max_traj_len:
            points = points[:self.max_traj_len]

        perspective = get_perspective(self.data_size, points, self.perspective_padding)

        mask_files = [ff for ff in files if (ff.startswith('patch_mask') and ff.endswith('.npy'))]
        mask_files.sort()
        if len(mask_files) > self.max_traj_len:
            mask_files = mask_files[:self.max_traj_len]
        assert (len(mask_files) == traj_len)
        mask_list = [np.load((traj_folder + '/' + ff)) for ff in mask_files]
        # mask_list = [np.flip(mask.reshape((self.data_size[0], self.data_size[1], -1)), 0).copy() for mask in mask_list]
        mask_list = [ku.image_to_tensor(mask).ge(0.5).repeat(3, 1, 1) for mask in mask_list]

        mask_tensor = torch.stack(mask_list)

        return traj_len, img1_I0_tensor, img2_I0_tensor, intrinsic_I0_tensor, \
               img1_I1_tensor, img2_I1_tensor, intrinsic_I1_tensor, \
               motions, scales, poses_quat, patch_rel_pose, perspective, mask_tensor

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        processed_traj_folder = self.processed_traj_folders[idx]
        img1_I0, img2_I0, intrinsic_I0, img1_I1, img2_I1, intrinsic_I1, img1_delta, img2_delta, \
        motions, scales, poses_quat, patch_rel_pose, perspective, mask = self.load_processed_trajectory_folder(processed_traj_folder)

        return self.traj_dataset_indices[idx], self.traj_dataset_names[idx], self.traj_names[idx], self.traj_len, \
               img1_I0, img2_I0, intrinsic_I0, img1_I1, img2_I1, intrinsic_I1, img1_delta, img2_delta, \
               motions, scales, poses_quat, patch_rel_pose, mask, perspective
