import itertools
import os
import random
from pathlib import Path
import cv2
import torch
import matplotlib.pyplot as plt

from utils import get_args
import numpy as np
from Datasets.utils import plot_traj, visflow
from Datasets.transformation import ses2poses_quat, pos_quats2SE_matrices
from evaluator.tartanair_evaluator import TartanAirEvaluator
from os import mkdir
from os.path import isdir, join
from torchvision.utils import save_image
from Datasets.tartanTrajFlowDataset import extract_traj_data
from torch.utils.data import DataLoader
from random import sample
import gc
import csv


def load_img(path, transform):
    img_dict = {'img': cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)}
    img = transform(img_dict)['img']
    return img


def masked_list(lst, indices):
    return [lst[idx] for idx in indices]


def collect_columns(src_list, target_list):
    for idx, val in enumerate(src_list):
        target_list[idx].append(val)


def avg_criterions_samples(criterions_list_samples):
    criterions_num = len(criterions_list_samples[0])
    traj_num = len(criterions_list_samples[0][0])
    frames_num = len(criterions_list_samples[0][0][0])
    criterions_samples_list = [[[[] for _ in range(frames_num)]
                                for _ in range(traj_num)]
                               for _ in range(criterions_num)]

    for criterions_list in criterions_list_samples:
        for crit_idx, crit_list in enumerate(criterions_list):
            for traj_idx, traj_list in enumerate(crit_list):
                for frame_idx, frame_crit_val in enumerate(traj_list):
                    criterions_samples_list[crit_idx][traj_idx][frame_idx].append(frame_crit_val)
    averaged_criterions_list = [[[np.mean(criterions_samples_list[crit_idx][traj_idx][frame_idx])
                                  for frame_idx in range(frames_num)]
                                 for traj_idx in range(traj_num)]
                                for crit_idx in range(criterions_num)]
    return averaged_criterions_list


def save_flow_imgs(flow, flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_clean'):
    flow_save_dir = flowdir + '/' + dataset_name
    if not isdir(flow_save_dir):
        mkdir(flow_save_dir)
    flow_save_dir = flow_save_dir + '/' + traj_name + save_dir_suffix
    if not isdir(flow_save_dir):
        mkdir(flow_save_dir)

    print("flow.shape")
    print(flow.shape)
    flow = flow.permute(0, 2, 3, 1).cpu().numpy()
    print('==> saving flow to {}'.format(flow_save_dir))
    print()

    flowcount = 0
    for f in flow:
        np.save(flow_save_dir + '/' + str(flowcount).zfill(6) + '.npy', f)
        flow_vis = visflow(f)
        cv2.imwrite(flow_save_dir + '/' + str(flowcount).zfill(6) + '.png', flow_vis)
        flowcount += 1


def save_pose_list(motions, pose_quat_gt, datastr, pose_dir):

    poselist = ses2poses_quat(motions.cpu().numpy())
    if not isdir(pose_dir):
        mkdir(pose_dir)
    print('==> saving pose list to {}'.format(pose_dir))

    if pose_quat_gt is None:
        np.savetxt(pose_dir + '/output.txt', poselist)
    else:
        pose_quat_list_gt = pose_quat_gt.cpu().numpy()
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(pose_quat_list_gt, poselist, scale=True, kittitype=(datastr=='kitti'))
        if datastr=='euroc':
            print("==> ATE: %.4f" %(results['ate_score']))
        else:
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname=pose_dir+'/output.png', title='ATE %.4f' %(results['ate_score']))
        np.savetxt(pose_dir+'/output.txt',results['est_aligned'])
        np.savetxt(pose_dir+'/gt.txt',results['gt_aligned'])


def save_poses_se(motions, pose_quat_gt, pose_dir, plot_est=True, plot_gt=True, sc=0.05,
               dataset_name="dataset_dir", traj_name="traj_dir", save_dir_suffix='_clean'):

    pose_save_dir = pose_dir + '/' + dataset_name
    if not isdir(pose_save_dir):
        mkdir(pose_save_dir)
    pose_save_dir = pose_save_dir + '/' + traj_name + save_dir_suffix
    if not isdir(pose_save_dir):
        mkdir(pose_save_dir)
    print('==> saving poses to {}'.format(pose_save_dir))
    print()

    gt_poses = np.array(pos_quats2SE_matrices(pose_quat_gt))
    poses_quat = ses2poses_quat(motions.cpu().numpy())
    poses = np.array(pos_quats2SE_matrices(poses_quat))

    np.savetxt(pose_save_dir + '/est_poses_quat.txt', poses_quat)
    np.savetxt(pose_save_dir + '/gt_poses_quat.txt', pose_quat_gt)

    factor = 1
    alpha = 0.8
    fig_traj = plt.figure()
    ax = fig_traj.add_subplot(projection='3d')
    if plot_est:
        ax.scatter(poses[:, 0, -1], poses[:, 1, -1], poses[:, 2, -1], s=15, color=(1,0,1))
    if plot_gt:
        ax.scatter(gt_poses[:, 0, -1], gt_poses[:, 1, -1], gt_poses[:, 2, -1], s=15, color=(1,0,0,alpha))

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    zlims = ax.get_zlim()
    xlims = np.abs(xlims[1] - xlims[0])
    ylims = np.abs(ylims[1] - ylims[0])
    zlims = np.abs(zlims[1] - zlims[0])


    t_norm = []
    for p_idx, (p_gt, p) in enumerate(zip(gt_poses, poses)):
        tgt = p_gt[0:3, -1].reshape((3, 1))
        t = p[0:3, -1].reshape((3, 1))
        t_norm.append(np.linalg.norm(t-tgt))
        if p_idx % factor == 0:
            if plot_est:
                ax.plot([p[0, -1], p[0, -1] + sc * xlims * p[0, 0]], [p[1, -1], p[1, -1] + sc * ylims * p[1, 0]],
                        zs=[p[2, -1], p[2, -1] + sc * zlims * p[2, 0]], color=(1,0,1))
                ax.plot([p[0, -1], p[0, -1] + sc * xlims * p[0, 1]], [p[1, -1], p[1, -1] + sc * ylims * p[1, 1]],
                        zs=[p[2, -1], p[2, -1] + sc * zlims * p[2, 1]], color=(0,1,1))
                ax.plot([p[0, -1], p[0, -1] + sc * xlims * p[0, 2]], [p[1, -1], p[1, -1] + sc * ylims * p[1, 2]],
                        zs=[p[2, -1], p[2, -1] + sc * zlims * p[2, 2]], color=(0.5,0.5,0.5))
            if plot_gt:
                ax.plot([p_gt[0, -1], p_gt[0, -1] + sc * xlims * p_gt[0, 0]], [p_gt[1, -1], p_gt[1, -1] + sc * ylims * p_gt[1, 0]],
                        zs=[p_gt[2, -1], p_gt[2, -1] + sc * zlims * p_gt[2, 0]], color=(1,0,0,alpha))
                ax.plot([p_gt[0, -1], p_gt[0, -1] + sc * xlims * p_gt[0, 1]], [p_gt[1, -1], p_gt[1, -1] + sc * ylims * p_gt[1, 1]],
                        zs=[p_gt[2, -1], p_gt[2, -1] + sc * zlims * p_gt[2, 1]], color=(0,1,0,alpha))
                ax.plot([p_gt[0, -1], p_gt[0, -1] + sc * xlims * p_gt[0, 2]], [p_gt[1, -1], p_gt[1, -1] + sc * ylims * p_gt[1, 2]],
                        zs=[p_gt[2, -1], p_gt[2, -1] + sc * zlims * p_gt[2, 2]], color=(0,0,1,alpha))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_zaxis()  # Since we use NED system
    plt.title(f'L2 final translation norm = {t_norm[-1]:.3f}[m]')
    plt.legend(labels=['estimated_x', 'estimated_y', 'estimated_z', 'GT_x', 'GT_y', 'GT_z'])
    ax.view_init(elev=89.99, azim=-90.)
    plt.savefig(join(pose_save_dir, 'xy_view.png'))
    ax.view_init(elev=180., azim=-90.)
    plt.savefig(join(pose_save_dir, 'xz_view.png'))
    ax.view_init(elev=-180., azim=0.)
    plt.savefig(join(pose_save_dir, 'yz_view.png'))
    ax.view_init(elev=20., azim=135)
    plt.savefig(join(pose_save_dir, 'view3d.png'))
    plt.close()


def save_img_tensors(img1, img2, path, names=None):
    if not isdir(path):
        mkdir(path)
    if names is None:
        imgs_num = len(img1)
        if img2 is not None:
            imgs_num += 1
        names = [str(idx) for idx in range(imgs_num)]
    for img_idx, img in enumerate(img1):
        save_image(img, path + '/' + names[img_idx] +'.png')
    if img2 is not None:
        save_image(img2[-1], path + '/' + names[-1] + '.png')


def save_unprocessed_imgs(img_dir, dataset_name, traj_name, img1_I0, img2_I0, img1_I1, img2_I1):
    img_save_dir = img_dir + '/' + dataset_name
    if not isdir(img_save_dir):
        mkdir(img_save_dir)

    save_img_tensors(img1_I0, img2_I0, img_save_dir + '/' + traj_name + '_I0')
    save_img_tensors(img1_I1, img2_I1, img_save_dir + '/' + traj_name + '_I1')


def save_adv_imgs(adv_img_dir, adv_pert_dir, dataset_name, traj_name, img1_adv, img2_adv, best_pert):
    adv_img_save_dir = adv_img_dir + '/' + dataset_name
    if not isdir(adv_img_save_dir):
        mkdir(adv_img_save_dir)
    adv_pert_save_dir = adv_pert_dir + '/' + dataset_name
    if not isdir(adv_pert_save_dir):
        mkdir(adv_pert_save_dir)
    save_img_tensors(img1_adv, img2_adv, adv_img_save_dir + '/' + traj_name)
    save_img_tensors(best_pert, best_pert, adv_pert_save_dir + '/' + traj_name)


def save_unprocessed_flow(flowdir, dataset_name, traj_name, flow_I0, flow_I1):
    save_flow_imgs(flow_I0, flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_I0')
    save_flow_imgs(flow_I1, flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_I1')


def save_unprocessed_pose(pose_dir, dataset_name, traj_name, motions_I0, motions_I1, pose_quat_gt):
    save_poses_se(motions_I0, pose_quat_gt, pose_dir,
               dataset_name=dataset_name, traj_name=traj_name, save_dir_suffix='_I0')
    save_poses_se(motions_I1, pose_quat_gt, pose_dir,
               dataset_name=dataset_name, traj_name=traj_name, save_dir_suffix='_I1')


def make_output_folders(output_dir, save_flow, save_pose, save_imgs):
    flowdir = None
    pose_dir = None
    img_dir = None
    adv_img_dir = None
    adv_pert_dir = None

    if save_flow:
        flowdir = output_dir + '/flow'
        if not isdir(flowdir):
            mkdir(flowdir)
    if save_pose:
        pose_dir = output_dir + '/pose'
        if not isdir(pose_dir):
            mkdir(pose_dir)
    if save_imgs:
        img_dir = output_dir + '/clean_images'
        if not isdir(img_dir):
            mkdir(img_dir)
        adv_img_dir = output_dir + '/adv_images'
        if not isdir(adv_img_dir):
            mkdir(adv_img_dir)
        adv_pert_dir = output_dir + '/adv_pert'
        if not isdir(adv_pert_dir):
            mkdir(adv_pert_dir)

    return flowdir, pose_dir, img_dir, adv_img_dir, adv_pert_dir


def make_exp_folders(output_dir, test_fold_idx,
                               save_results, save_best_pert, save_flow, save_pose, save_imgs,
                               fold_idx_str='out_of_sample_test_fold_idx', make_test_folder=True):


    adv_best_pert_dir = None
    train_output_dir = None
    train_flowdir = None
    train_pose_dir = None
    train_img_dir = None
    train_adv_img_dir = None
    train_adv_pert_dir = None
    eval_output_dir = None
    eval_flowdir = None
    eval_pose_dir = None
    eval_img_dir = None
    eval_adv_img_dir = None
    eval_adv_pert_dir = None
    test_output_dir = None
    test_flowdir = None
    test_pose_dir = None
    test_img_dir = None
    test_adv_img_dir = None
    test_adv_pert_dir = None

    print("save_results")
    print(save_results)
    if save_results:
        output_dir = output_dir + '/' + fold_idx_str + '_' + str(test_fold_idx)
        if not isdir(output_dir):
            mkdir(output_dir)
        print("save_best_pert")
        print(save_best_pert)
        if save_best_pert:
            adv_best_pert_dir = output_dir + '/adv_best_pert'
            if not isdir(adv_best_pert_dir):
                mkdir(adv_best_pert_dir)
            print("adv_best_pert_dir")
            print(adv_best_pert_dir)

        train_output_dir = output_dir + '/train'
        if not isdir(train_output_dir):
            mkdir(train_output_dir)
        train_flowdir, train_pose_dir, train_img_dir, train_adv_img_dir, train_adv_pert_dir = \
            make_output_folders(train_output_dir, save_flow, save_pose, save_imgs)
        eval_output_dir = output_dir + '/eval'
        if not isdir(eval_output_dir):
            mkdir(eval_output_dir)
        eval_flowdir, eval_pose_dir, eval_img_dir, eval_adv_img_dir, eval_adv_pert_dir = \
            make_output_folders(eval_output_dir, save_flow, save_pose, save_imgs)
        if make_test_folder:
            test_output_dir = output_dir + '/test'
            if not isdir(test_output_dir):
                mkdir(test_output_dir)
            test_flowdir, test_pose_dir, test_img_dir, test_adv_img_dir, test_adv_pert_dir = \
                make_output_folders(test_output_dir, save_flow, save_pose, save_imgs)

    print("adv_best_pert_dir")
    print(adv_best_pert_dir)

    return output_dir, adv_best_pert_dir, \
           train_output_dir, train_flowdir, train_pose_dir, train_img_dir, train_adv_img_dir, train_adv_pert_dir, \
           eval_output_dir, eval_flowdir, eval_pose_dir, eval_img_dir, eval_adv_img_dir, eval_adv_pert_dir, \
           test_output_dir, test_flowdir, test_pose_dir, test_img_dir, test_adv_img_dir, test_adv_pert_dir


def out_of_sample_make_folders(output_dir, test_fold_idx,
                               save_results, save_best_pert, save_flow, save_pose, save_imgs):

    return make_exp_folders(output_dir, test_fold_idx,
                               save_results, save_best_pert, save_flow, save_pose, save_imgs)


def open_loop_make_folders(output_dir, eval_fold_idx,
                               save_results, save_best_pert, save_flow, save_pose, save_imgs):
    output_dir, adv_best_pert_dir, \
    train_output_dir, train_flowdir, train_pose_dir, train_img_dir, train_adv_img_dir, train_adv_pert_dir, \
    eval_output_dir, eval_flowdir, eval_pose_dir, eval_img_dir, eval_adv_img_dir, eval_adv_pert_dir, \
    test_output_dir, test_flowdir, test_pose_dir, test_img_dir, test_adv_img_dir, test_adv_pert_dir=\
        make_exp_folders(output_dir, eval_fold_idx,
                               save_results, save_best_pert, save_flow, save_pose, save_imgs,
                               fold_idx_str='open_loop_eval_fold_idx', make_test_folder=False)
    return output_dir, adv_best_pert_dir, \
    train_output_dir, train_flowdir, train_pose_dir, train_img_dir, train_adv_img_dir, train_adv_pert_dir, \
    eval_output_dir, eval_flowdir, eval_pose_dir, eval_img_dir, eval_adv_img_dir, eval_adv_pert_dir


def test_model(model, criterions, img1, img2, intrinsic, scale_gt, motions_target, target_pose=None,
               window_size=None, device=None):
    if window_size is None:
        if device is None:
            motions, flow = model.test_batch(img1, img2, intrinsic, scale_gt)
            crit_results = [crit((motions, flow), scale_gt, motions_target, target_pose).detach().cpu()
                            for crit in criterions]
            return (motions, flow), crit_results

        img1_device = img1.clone().detach().to(device)
        img2_device = img2.clone().detach().to(device)
        intrinsic_device = intrinsic.clone().detach().to(device)
        scale_gt_device = scale_gt.clone().detach().to(device)
        motions_target_device = motions_target.clone().detach().to(device)
        target_pose_device = target_pose.clone().detach().to(device)

        motions_device, flow_device = model.test_batch(img1_device, img2_device, intrinsic_device, scale_gt_device)
        crit_results_device = [crit((motions_device, flow_device), scale_gt_device,
                                    motions_target_device, target_pose_device)
                               for crit in criterions]
        motions = motions_device.clone().detach().cpu()
        flow = flow_device.clone().detach().cpu()
        crit_results = [crit_result_device.clone().detach().cpu() for crit_result_device in crit_results_device]

        del img1_device
        del img2_device
        del intrinsic_device
        del scale_gt_device
        del motions_target_device
        del target_pose_device
        del motions_device
        del flow_device
        del crit_results_device
        torch.cuda.empty_cache()

        return (motions, flow), crit_results

    data_ind = list(range(img1.shape[0] + 1))
    window_start_list = data_ind[0::window_size]
    window_end_list = data_ind[window_size::window_size]
    if window_end_list[-1] != data_ind[-1]:
        window_end_list.append(data_ind[-1])
    motions_window_list = []
    flow_window_list = []
    if device is None:
        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]

            img1_window = img1[window_start:window_end].clone().detach()
            img2_window = img2[window_start:window_end].clone().detach()
            intrinsic_window = intrinsic[window_start:window_end].clone().detach()
            scale_gt_window = scale_gt[window_start:window_end].clone().detach()

            motions_window, flow_window = model.test_batch(img1_window, img2_window, intrinsic_window, scale_gt_window)
            motions_window_list.append(motions_window)
            flow_window_list.append(flow_window)

            del img1_window
            del img2_window
            del intrinsic_window
            del scale_gt_window
            torch.cuda.empty_cache()

        motions = torch.cat(motions_window_list, dim=0)
        flow = torch.cat(flow_window_list, dim=0)
        crit_results = [crit((motions, flow), scale_gt, motions_target, target_pose).detach().cpu()
                        for crit in criterions]
        del motions_window_list
        del flow_window_list
        torch.cuda.empty_cache()
        return (motions, flow), crit_results

    for window_idx, window_end in enumerate(window_end_list):
        window_start = window_start_list[window_idx]

        img1_window = img1[window_start:window_end].clone().detach().to(device)
        img2_window = img2[window_start:window_end].clone().detach().to(device)
        intrinsic_window = intrinsic[window_start:window_end].clone().detach().to(device)
        scale_gt_window = scale_gt[window_start:window_end].clone().detach().to(device)

        motions_window, flow_window = model.test_batch(img1_window, img2_window, intrinsic_window, scale_gt_window)
        print("")
        print()
        motions_window_list.append(motions_window)
        flow_window_list.append(flow_window)

        del img1_window
        del img2_window
        del intrinsic_window
        del scale_gt_window
        torch.cuda.empty_cache()

    motions_device = torch.cat(motions_window_list, dim=0)
    flow_device = torch.cat(flow_window_list, dim=0)
    scale_gt_device = scale_gt.clone().detach().to(device)
    motions_target_device = motions_target.clone().detach().to(device)
    target_pose_device = target_pose.clone().detach().to(device)

    crit_results_device = [crit((motions_device, flow_device), scale_gt_device,
                                motions_target_device, target_pose_device)
                           for crit in criterions]
    motions = motions_device.clone().detach().cpu()
    flow = flow_device.clone().detach().cpu()
    crit_results = [crit_result_device.clone().detach().cpu() for crit_result_device in crit_results_device]

    del scale_gt_device
    del motions_target_device
    del target_pose_device
    del motions_device
    del flow_device
    del crit_results_device
    del motions_window_list
    del flow_window_list
    torch.cuda.empty_cache()

    return (motions, flow), crit_results


def test_clean_multi_inputs(args):
    dataset_idx_list = []
    dataset_name_list = []
    traj_name_list = []
    traj_indices = []
    motions_gt_list = []
    traj_clean_motions = []
    traj_motions_scales = []
    traj_mask_l0_ratio_list = []

    traj_clean_criterions_list = [[] for crit in args.criterions]
    frames_clean_criterions_list = [[[] for i in range(args.traj_len)] for crit in args.criterions]

    print("len(args.testDataloader)")
    print(len(args.testDataloader))
    for traj_idx, traj_data in enumerate(args.testDataloader):
        dataset_idx, dataset_name, traj_name, traj_len, \
        img1_I0, img2_I0, intrinsic_I0, \
        img1_I1, img2_I1, intrinsic_I1, \
        img1_delta, img2_delta, \
        motions_gt, scale_gt, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(traj_data)

        print("dataset_idx")
        print(dataset_idx)
        print("dataset_name")
        print(dataset_name)
        print("traj_idx")
        print(traj_idx)
        print("traj_name")
        print(traj_name)
        print("traj_len")
        print(traj_len)
        print("img1_I0.shape")
        print(img1_I0.shape)
        print("img2_I0.shape")
        print(img2_I0.shape)
        print("intrinsic_I0.shape")
        print(intrinsic_I0.shape)
        print("img1_I1.shape")
        print(img1_I1.shape)
        print("img2_I1.shape")
        print(img2_I1.shape)
        print("intrinsic_I1.shape")
        print(intrinsic_I1.shape)
        print("img1_delta.shape")
        print(img1_delta.shape)
        print("img2_delta.shape")
        print(img2_delta.shape)
        print("motions_gt.shape")
        print(motions_gt.shape)
        print("scale_gt.shape")
        print(scale_gt.shape)
        print("pose_quat_gt.shape")
        print(pose_quat_gt.shape)
        print("patch_pose.shape")
        print(patch_pose.shape)

        print("mask.shape")
        print(mask.shape)
        print("perspective.shape")
        print(perspective.shape)
        traj_mask_l0_ratio = (mask.count_nonzero() / mask.numel()).item()

        print("traj_mask_l0_ratio")
        print(traj_mask_l0_ratio)


        dataset_idx_list.append(dataset_idx)
        dataset_name_list.append(dataset_name)
        traj_name_list.append(traj_name)
        traj_indices.append(traj_idx)
        motions_gt_list.append(motions_gt)
        traj_mask_l0_ratio_list.append(traj_mask_l0_ratio)

        if args.save_imgs:
            save_unprocessed_imgs(args.img_dir, dataset_name, traj_name, img1_I0, img2_I0, img1_I1, img2_I1)

        with torch.no_grad():
            (motions, flow), crit_results = \
                test_model(args.model, args.criterions,
                           img1_I0, img2_I0, intrinsic_I0, scale_gt, motions_gt, patch_pose,
                           window_size=args.window_size, device=args.device)
            for crit_idx, crit_result in enumerate(crit_results):
                crit_result_list = crit_result.tolist()
                traj_clean_criterions_list[crit_idx].append(crit_result_list)
                print(args.criterions_names[crit_idx] + " for trajectory: " + traj_name)
                print(crit_result_list)
                for frame_idx, frame_clean_crit in enumerate(crit_result_list):
                    frames_clean_criterions_list[crit_idx][frame_idx].append(frame_clean_crit)
            del crit_results

        if args.save_flow:
            save_flow_imgs(flow, args.flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_clean')

        if args.save_pose:
            save_poses_se(motions, pose_quat_gt, args.pose_dir,
                       dataset_name=dataset_name, traj_name=traj_name, save_dir_suffix='_clean')

        traj_clean_motions.append(motions)
        traj_motions_scales.append(scale_gt.numpy())
        del flow
        del img1_I0
        del img2_I0
        del intrinsic_I0
        del img1_I1
        del img2_I1
        del intrinsic_I1
        del img1_delta
        del img2_delta
        del motions_gt
        del scale_gt
        del pose_quat_gt
        del mask
        del perspective
        torch.cuda.empty_cache()

    for crit_idx, frames_clean_crit_list in enumerate(frames_clean_criterions_list):
        frames_clean_crit_mean = [np.mean(crit_list) for crit_list in frames_clean_crit_list]
        frames_clean_crit_std = [np.std(crit_list) for crit_list in frames_clean_crit_list]
        print("frames_clean_" + args.criterions_names[crit_idx] + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
        print(frames_clean_crit_mean)
        print("frames_clean_" + args.criterions_names[crit_idx] + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
        print(frames_clean_crit_std)

    traj_mask_l0_ratio_mean = np.mean(traj_mask_l0_ratio_list)
    traj_mask_l0_ratio_std = np.std(traj_mask_l0_ratio_list)
    print("traj_mask_l0_ratio_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(traj_mask_l0_ratio_mean)
    print("traj_mask_l0_ratio_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(traj_mask_l0_ratio_std)

    frames_motions_scales = [[motions_scales[i] for motions_scales in traj_motions_scales]
                             for i in range(traj_len - 1)]
    frames_motions_scales_means = [np.mean(scale_list) for scale_list in frames_motions_scales]
    frames_motions_scales_stds = [np.std(scale_list) for scale_list in frames_motions_scales]
    frames_mean_dist = [0]
    curr_dist = 0
    for motion_scale in frames_motions_scales_means:
        curr_dist += motion_scale
        frames_mean_dist.append(curr_dist)

    print("frames_motions_scales_means over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_motions_scales_means)
    print("frames_motions_scales_stds over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_motions_scales_stds)
    print("frames_mean_dist over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_mean_dist)

    del frames_clean_criterions_list
    del frames_motions_scales_means
    del frames_motions_scales_stds
    del frames_mean_dist
    del traj_motions_scales
    torch.cuda.empty_cache()

    return dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, motions_gt_list, \
           traj_clean_criterions_list, traj_clean_motions


def test_adv_trajectories(dataloader, model, motions_target_list, attack, pert,
                          criterions, window_size,
                          save_imgs, save_flow, save_pose,
                          adv_img_dir, adv_pert_dir, flowdir, pose_dir,
                          device=None, multi_perturb=False):
    traj_adv_criterions_list = [[] for crit in criterions]
    for traj_idx, traj_data in enumerate(dataloader):
        dataset_idx, dataset_name, traj_name, traj_len, \
        img1_I0, img2_I0, intrinsic_I0, \
        img1_I1, img2_I1, intrinsic_I1, \
        img1_delta, img2_delta, \
        motions_gt, scale_gt, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(traj_data)
        motions_target = motions_target_list[traj_idx]
        traj_pert = pert
        if multi_perturb:
            traj_pert = pert[traj_idx]

        with torch.no_grad():
            img1_adv, img2_adv = attack.apply_pert(traj_pert, img1_I0, img2_I0, img1_delta, img2_delta,
                                                   mask, perspective, device)
            (motions_adv, flow_adv), crit_results = \
                test_model(model, criterions,
                           img1_adv, img2_adv, intrinsic_I0, scale_gt, motions_target, patch_pose,
                           window_size=window_size, device=device)
            for crit_idx, crit_result in enumerate(crit_results):
                traj_adv_criterions_list[crit_idx].append(crit_result.tolist())
            del crit_results


        if save_imgs:
            save_adv_imgs(adv_img_dir, adv_pert_dir, dataset_name, traj_name,
                          img1_adv,  img2_adv, traj_pert)

        if save_flow:
            save_flow_imgs(flow_adv, flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_adv')

        if save_pose:
            save_poses_se(motions_adv, pose_quat_gt, pose_dir,
                       dataset_name=dataset_name, traj_name=traj_name, save_dir_suffix='_adv')

        del motions_adv
        del flow_adv
        del img1_I0
        del img2_I0
        del intrinsic_I0
        del img1_I1
        del img2_I1
        del intrinsic_I1
        del img1_delta
        del img2_delta
        del motions_gt
        del scale_gt
        del pose_quat_gt
        del patch_pose
        del mask
        del perspective
        del img1_adv
        del img2_adv
        torch.cuda.empty_cache()

    return traj_adv_criterions_list


def report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_crit_list, traj_adv_crit_list, save_csv, output_dir, crit_str,
                         experiment_name=""):
    traj_len = len(traj_clean_crit_list[0])
    csv_path = os.path.join(output_dir, 'results_' + crit_str + '.csv')
    frames_clean_crit_list = [[] for i in range(traj_len)]
    frames_adv_crit_list = [[] for i in range(traj_len)]
    frames_delta_crit_list = [[] for i in range(traj_len)]
    frames_ratio_crit_list = [[] for i in range(traj_len)]
    frames_delta_ratio_crit_list = [[] for i in range(traj_len)]
    traj_delta_crit_list = []
    traj_ratio_crit_list = []
    traj_delta_ratio_crit_list = []

    for traj_idx, traj_name in enumerate(traj_name_list):
        traj_clean_crit = traj_clean_crit_list[traj_idx]
        traj_adv_crit = traj_adv_crit_list[traj_idx]
        traj_delta_crit = []
        traj_ratio_crit = []
        traj_delta_ratio_crit = []

        for frame_idx, frame_clean_crit in enumerate(traj_clean_crit):
            frame_adv_crit = traj_adv_crit[frame_idx]
            frame_delta_crit = frame_adv_crit - frame_clean_crit
            if frame_clean_crit == 0:
                frame_ratio_crit = 0
                frame_delta_ratio_crit = 0
            else:
                frame_ratio_crit = frame_adv_crit / frame_clean_crit
                frame_delta_ratio_crit = frame_delta_crit / frame_clean_crit

            traj_delta_crit.append(frame_delta_crit)
            traj_ratio_crit.append(frame_ratio_crit)
            traj_delta_ratio_crit.append(frame_delta_ratio_crit)

            frames_clean_crit_list[frame_idx].append(frame_clean_crit)
            frames_adv_crit_list[frame_idx].append(frame_adv_crit)
            frames_delta_crit_list[frame_idx].append(frame_delta_crit)
            frames_ratio_crit_list[frame_idx].append(frame_ratio_crit)
            frames_delta_ratio_crit_list[frame_idx].append(frame_delta_ratio_crit)

        traj_delta_crit_list.append(traj_delta_crit)
        traj_ratio_crit_list.append(traj_ratio_crit)
        traj_delta_ratio_crit_list.append(traj_delta_ratio_crit)

        print(crit_str + "_crit_clean for trajectory: " + traj_name)
        print(traj_clean_crit)
        print(crit_str + "_crit_adv for trajectory: " + traj_name)
        print(traj_adv_crit)
        print(crit_str + "_crit_adv_delta for trajectory: " + traj_name)
        print(traj_delta_crit)
        print(crit_str + "_crit_adv_ratio for trajectory: " + traj_name)
        print(traj_ratio_crit)
        print(crit_str + "_crit_adv_delta_ratio for trajectory: " + traj_name)
        print(traj_delta_ratio_crit)

    if save_csv:
        fieldsnames = ['dataset_idx', 'dataset_name', 'traj_idx', 'traj_name', 'frame_idx',
                       'clean_' + crit_str, 'adv_' + crit_str, 'adv_delta_' + crit_str,
                       'adv_ratio_' + crit_str, 'adv_delta_ratio_' + crit_str, ]

        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldsnames)
            writer.writeheader()
            for traj_idx, traj_name in enumerate(traj_name_list):
                for frame_idx, frame_clean_crit in enumerate(traj_clean_crit_list[traj_idx]):
                    data = {'dataset_idx': dataset_idx_list[traj_idx],
                            'dataset_name': dataset_name_list[traj_idx],
                            'traj_idx': traj_indices[traj_idx],
                            'traj_name': traj_name,
                            'frame_idx': frame_idx,
                            'clean_' + crit_str: frame_clean_crit,
                            'adv_' + crit_str: traj_adv_crit_list[traj_idx][frame_idx],
                            'adv_delta_' + crit_str: traj_delta_crit_list[traj_idx][frame_idx],
                            'adv_ratio_' + crit_str: traj_ratio_crit_list[traj_idx][frame_idx],
                            'adv_delta_ratio_' + crit_str: traj_adv_crit_list[traj_idx][frame_idx]}
                    writer.writerow(data)
    del traj_delta_crit_list
    del traj_ratio_crit_list
    del traj_adv_crit_list

    frames_clean_crit_mean = [np.mean(crit_list) for crit_list in frames_clean_crit_list]
    frames_clean_crit_std = [np.std(crit_list) for crit_list in frames_clean_crit_list]
    frames_adv_crit_mean = [np.mean(crit_list) for crit_list in frames_adv_crit_list]
    frames_adv_crit_std = [np.std(crit_list) for crit_list in frames_adv_crit_list]
    frames_delta_crit_mean = [np.mean(crit_list) for crit_list in frames_delta_crit_list]
    frames_delta_crit_std = [np.std(crit_list) for crit_list in frames_delta_crit_list]
    frames_ratio_crit_mean = [np.mean(crit_list) for crit_list in frames_ratio_crit_list]
    frames_ratio_crit_std = [np.std(crit_list) for crit_list in frames_ratio_crit_list]
    frames_delta_ratio_crit_mean = [np.mean(crit_list) for crit_list in frames_delta_ratio_crit_list]
    frames_delta_ratio_crit_std = [np.std(crit_list) for crit_list in frames_delta_ratio_crit_list]

    print("frames_clean_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_clean_crit_mean)
    print("frames_clean_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_clean_crit_std)
    print("frames_adv_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_adv_crit_mean)
    print("frames_adv_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_adv_crit_std)
    print("frames_delta_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_delta_crit_mean)
    print("frames_delta_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_delta_crit_std)
    print("frames_ratio_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_ratio_crit_mean)
    print("frames_ratio_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_ratio_crit_std)
    print("frames_delta_ratio_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_delta_ratio_crit_mean)
    print("frames_delta_ratio_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_delta_ratio_crit_std)

    del frames_clean_crit_list
    del frames_adv_crit_list
    del frames_delta_crit_list
    del frames_ratio_crit_list
    del frames_delta_ratio_crit_list

    return frames_clean_crit_mean, frames_clean_crit_std, \
           frames_adv_crit_mean, frames_adv_crit_std, \
           frames_delta_crit_mean, frames_delta_crit_std, \
           frames_ratio_crit_mean, frames_ratio_crit_std, \
           frames_delta_ratio_crit_mean, frames_delta_ratio_crit_std


def collect_and_report_folds_mean_result(folds_frames_clean_crit_means, folds_frames_clean_crit_stds,
                                    folds_frames_adv_crit_means, folds_frames_adv_crit_stds,
                                    folds_frames_delta_crit_means, folds_frames_delta_crit_stds,
                                    folds_frames_ratio_crit_means, folds_frames_ratio_crit_stds,
                                    folds_frames_delta_ratio_crit_means, folds_frames_delta_ratio_crit_stds,
                                    frames_clean_crit_mean, frames_clean_crit_std,
                                    frames_adv_crit_mean, frames_adv_crit_std,
                                    frames_delta_crit_mean, frames_delta_crit_std,
                                    frames_ratio_crit_mean, frames_ratio_crit_std,
                                    frames_delta_ratio_crit_mean, frames_delta_ratio_crit_std,
                                    crit_str):

    collect_columns(frames_clean_crit_mean, folds_frames_clean_crit_means)
    collect_columns(frames_clean_crit_std, folds_frames_clean_crit_stds)
    collect_columns(frames_adv_crit_mean, folds_frames_adv_crit_means)
    collect_columns(frames_adv_crit_std, folds_frames_adv_crit_stds)
    collect_columns(frames_delta_crit_mean, folds_frames_delta_crit_means)
    collect_columns(frames_delta_crit_std, folds_frames_delta_crit_stds)
    collect_columns(frames_ratio_crit_mean, folds_frames_ratio_crit_means)
    collect_columns(frames_ratio_crit_std, folds_frames_ratio_crit_stds)
    collect_columns(frames_delta_ratio_crit_mean, folds_frames_delta_ratio_crit_means)
    collect_columns(frames_delta_ratio_crit_std, folds_frames_delta_ratio_crit_stds)

    report_mean_results(folds_frames_clean_crit_means, folds_frames_clean_crit_stds,
                                    folds_frames_adv_crit_means, folds_frames_adv_crit_stds,
                                    folds_frames_delta_crit_means, folds_frames_delta_crit_stds,
                                    folds_frames_ratio_crit_means, folds_frames_ratio_crit_stds,
                                    folds_frames_delta_ratio_crit_means, folds_frames_delta_ratio_crit_stds,
                                    crit_str, final_report=False)


def report_mean_results(folds_frames_clean_crit_means, folds_frames_clean_crit_stds,
                                    folds_frames_adv_crit_means, folds_frames_adv_crit_stds,
                                    folds_frames_delta_crit_means, folds_frames_delta_crit_stds,
                                    folds_frames_ratio_crit_means, folds_frames_ratio_crit_stds,
                                    folds_frames_delta_ratio_crit_means, folds_frames_delta_ratio_crit_stds,
                                    crit_str, final_report=True):

    folds_frames_clean_crit_mean = [np.mean(rms_list) for rms_list in folds_frames_clean_crit_means]
    folds_frames_clean_crit_std = [np.mean(rms_list) for rms_list in folds_frames_clean_crit_stds]
    folds_frames_adv_crit_mean = [np.mean(rms_list) for rms_list in folds_frames_adv_crit_means]
    folds_frames_adv_crit_std = [np.mean(rms_list) for rms_list in folds_frames_adv_crit_stds]
    folds_frames_delta_crit_mean = [np.mean(rms_list) for rms_list in folds_frames_delta_crit_means]
    folds_frames_delta_crit_std = [np.mean(rms_list) for rms_list in folds_frames_delta_crit_stds]
    folds_frames_ratio_crit_mean = [np.mean(rms_list) for rms_list in folds_frames_ratio_crit_means]
    folds_frames_ratio_crit_std = [np.mean(rms_list) for rms_list in folds_frames_ratio_crit_stds]
    folds_frames_delta_ratio_crit_mean = [np.mean(rms_list) for rms_list in
                                               folds_frames_delta_ratio_crit_means]
    folds_frames_delta_ratio_crit_std = [np.mean(rms_list) for rms_list in folds_frames_delta_ratio_crit_stds]

    folds_tested_str = str(len(folds_frames_clean_crit_means[0]))
    if final_report:
        folds_tested_str = "all (" + folds_tested_str + ")"
    print("folds_frames_clean_" + crit_str + "_mean over " + folds_tested_str + " train-test splits:")
    print(folds_frames_clean_crit_mean)
    print("folds_frames_clean_" + crit_str + "_std over " + folds_tested_str + " train-test splits:")
    print(folds_frames_clean_crit_std)
    print("folds_frames_adv_" + crit_str + "_mean over " + folds_tested_str + " train-test splits:")
    print(folds_frames_adv_crit_mean)
    print("folds_frames_adv_" + crit_str + "_std over " + folds_tested_str + " train-test splits:")
    print(folds_frames_adv_crit_std)
    print("folds_frames_delta_" + crit_str + "_mean over " + folds_tested_str + " train-test splits:")
    print(folds_frames_delta_crit_mean)
    print("folds_frames_delta_" + crit_str + "_std over " + folds_tested_str + " train-test splits:")
    print(folds_frames_delta_crit_std)
    print("folds_frames_ratio_" + crit_str + "_mean over " + folds_tested_str + " train-test splits:")
    print(folds_frames_ratio_crit_mean)
    print("folds_frames_ratio_" + crit_str + "_std over " + folds_tested_str + " train-test splits:")
    print(folds_frames_ratio_crit_std)
    print("folds_frames_delta_ratio_" + crit_str + "_mean over " + folds_tested_str + " train-test splits:")
    print(folds_frames_delta_ratio_crit_mean)
    print("folds_frames_delta_ratio_" + crit_str + "_std over " + folds_tested_str + " train-test splits:")
    print(folds_frames_delta_ratio_crit_std)


def collect_and_report_folds_best_result(folds_frames_clean_crit_means, folds_frames_clean_crit_stds,
                                    folds_frames_adv_crit_means, folds_frames_adv_crit_stds,
                                    folds_frames_delta_crit_means, folds_frames_delta_crit_stds,
                                    folds_frames_ratio_crit_means, folds_frames_ratio_crit_stds,
                                    folds_frames_delta_ratio_crit_means, folds_frames_delta_ratio_crit_stds,
                                    frames_clean_crit_mean, frames_clean_crit_std,
                                    frames_adv_crit_mean, frames_adv_crit_std,
                                    frames_delta_crit_mean, frames_delta_crit_std,
                                    frames_ratio_crit_mean, frames_ratio_crit_std,
                                    frames_delta_ratio_crit_mean, frames_delta_ratio_crit_std,
                                    crit_str):

    collect_columns(frames_clean_crit_mean, folds_frames_clean_crit_means)
    collect_columns(frames_clean_crit_std, folds_frames_clean_crit_stds)
    collect_columns(frames_adv_crit_mean, folds_frames_adv_crit_means)
    collect_columns(frames_adv_crit_std, folds_frames_adv_crit_stds)
    collect_columns(frames_delta_crit_mean, folds_frames_delta_crit_means)
    collect_columns(frames_delta_crit_std, folds_frames_delta_crit_stds)
    collect_columns(frames_ratio_crit_mean, folds_frames_ratio_crit_means)
    collect_columns(frames_ratio_crit_std, folds_frames_ratio_crit_stds)
    collect_columns(frames_delta_ratio_crit_mean, folds_frames_delta_ratio_crit_means)
    collect_columns(frames_delta_ratio_crit_std, folds_frames_delta_ratio_crit_stds)

    report_best_results(folds_frames_clean_crit_means, folds_frames_clean_crit_stds,
                        folds_frames_adv_crit_means, folds_frames_adv_crit_stds,
                        folds_frames_delta_crit_means, folds_frames_delta_crit_stds,
                        folds_frames_ratio_crit_means, folds_frames_ratio_crit_stds,
                        folds_frames_delta_ratio_crit_means, folds_frames_delta_ratio_crit_stds,
                        crit_str, final_report=False)


def report_best_results(folds_frames_clean_crit_means, folds_frames_clean_crit_stds,
                                    folds_frames_adv_crit_means, folds_frames_adv_crit_stds,
                                    folds_frames_delta_crit_means, folds_frames_delta_crit_stds,
                                    folds_frames_ratio_crit_means, folds_frames_ratio_crit_stds,
                                    folds_frames_delta_ratio_crit_means, folds_frames_delta_ratio_crit_stds,
                                    crit_str, final_report=True):
    best_result_idx = np.argmax(folds_frames_adv_crit_means[-1])

    folds_frames_clean_crit_mean = [rms_list[best_result_idx] for rms_list in folds_frames_clean_crit_means]
    folds_frames_clean_crit_std = [rms_list[best_result_idx] for rms_list in folds_frames_clean_crit_stds]
    folds_frames_adv_crit_mean = [rms_list[best_result_idx] for rms_list in folds_frames_adv_crit_means]
    folds_frames_adv_crit_std = [rms_list[best_result_idx] for rms_list in folds_frames_adv_crit_stds]
    folds_frames_delta_crit_mean = [rms_list[best_result_idx] for rms_list in folds_frames_delta_crit_means]
    folds_frames_delta_crit_std = [rms_list[best_result_idx] for rms_list in folds_frames_delta_crit_stds]
    folds_frames_ratio_crit_mean = [rms_list[best_result_idx] for rms_list in folds_frames_ratio_crit_means]
    folds_frames_ratio_crit_std = [rms_list[best_result_idx] for rms_list in folds_frames_ratio_crit_stds]
    folds_frames_delta_ratio_crit_mean = [rms_list[best_result_idx] for rms_list in
                                               folds_frames_delta_ratio_crit_means]
    folds_frames_delta_ratio_crit_std = [rms_list[best_result_idx] for rms_list in folds_frames_delta_ratio_crit_stds]

    folds_tested_str = str(len(folds_frames_clean_crit_means[0]))
    if final_report:
        folds_tested_str = "all (" + folds_tested_str + ")"

    print("reporting the best results for criterion: " + crit_str)
    print("the best results over " + folds_tested_str + " eval-sets were achieved for evaluation folder idx: " + str(best_result_idx))
    print("best folds_frames_clean_" + crit_str + "_mean over " + folds_tested_str + " eval-sets:")
    print(folds_frames_clean_crit_mean)
    print("best folds_frames_clean_" + crit_str + "_std over " + folds_tested_str + " eval-sets:")
    print(folds_frames_clean_crit_std)
    print("best folds_frames_adv_" + crit_str + "_mean over " + folds_tested_str + " eval-sets:")
    print(folds_frames_adv_crit_mean)
    print("best folds_frames_adv_" + crit_str + "_std over " + folds_tested_str + " eval-sets:")
    print(folds_frames_adv_crit_std)
    print("best folds_frames_delta_" + crit_str + "_mean over " + folds_tested_str + " eval-sets:")
    print(folds_frames_delta_crit_mean)
    print("best folds_frames_delta_" + crit_str + "_std over " + folds_tested_str + " eval-sets:")
    print(folds_frames_delta_crit_std)
    print("best folds_frames_ratio_" + crit_str + "_mean over " + folds_tested_str + " eval-sets:")
    print(folds_frames_ratio_crit_mean)
    print("best folds_frames_ratio_" + crit_str + "_std over " + folds_tested_str + " eval-sets:")
    print(folds_frames_ratio_crit_std)
    print("best folds_frames_delta_ratio_" + crit_str + "_mean over " + folds_tested_str + " eval-sets:")
    print(folds_frames_delta_ratio_crit_mean)
    print("best folds_frames_delta_ratio_" + crit_str + "_std over " + folds_tested_str + " eval-sets:")
    print(folds_frames_delta_ratio_crit_std)


def adv_oos_test_perturb_trajectories(args, train_traj_names, eval_traj_names, test_traj_names,
                                     attack, train_dataloader, motions_target_train_list,
                                     eval_dataloader, motions_target_eval_list,
                                     test_dataloader, motions_target_test_list,
                                     adv_best_pert_dir, train_adv_img_dir, train_adv_pert_dir, train_flowdir, train_pose_dir,
                                     eval_adv_img_dir, eval_adv_pert_dir, eval_flowdir, eval_pose_dir,
                                     test_adv_img_dir, test_adv_pert_dir, test_flowdir, test_pose_dir):
    print("optimizing attacks on trajectories:")
    print(train_traj_names)
    print("evaluate attacks on trajectories:")
    print(eval_traj_names)
    print("testing attacks on trajectories:")
    print(test_traj_names)

    best_pert, eval_clean_loss_list, eval_all_loss_list, eval_all_best_loss_list = \
        attack.perturb(train_dataloader, motions_target_train_list, eps=args.eps, device=args.device,
                       eval_data_loader=eval_dataloader, eval_y_list=motions_target_eval_list)

    adv_eval_loss_list = attack.test_pert(best_pert, eval_dataloader, motions_target_eval_list, device=args.device)

    adv_test_loss_list = attack.test_pert(best_pert, test_dataloader, motions_target_test_list, device=args.device)

    print("eval_clean_loss_list")
    print(eval_clean_loss_list)
    print("eval_all_loss_list")
    print(eval_all_loss_list)
    print("eval_all_best_loss_list")
    print(eval_all_best_loss_list)
    eval_best_loss_list = eval_all_best_loss_list[- 1]
    print("eval_best_loss_list")
    print(eval_best_loss_list)
    print("adv_eval_loss_list")
    print(adv_eval_loss_list)
    print("adv_test_loss_list")
    print(adv_test_loss_list)

    if args.save_best_pert:
        save_image(best_pert[0], adv_best_pert_dir + '/' + 'adv_best_pert.png')

    traj_adv_train_criterions_list = \
        test_adv_trajectories(train_dataloader, args.model, motions_target_train_list, attack, best_pert,
                              args.criterions, args.window_size,
                              args.save_imgs, args.save_flow, args.save_pose,
                              train_adv_img_dir, train_adv_pert_dir, train_flowdir, train_pose_dir,
                              device=args.device)

    traj_adv_eval_criterions_list = \
        test_adv_trajectories(eval_dataloader, args.model, motions_target_eval_list, attack, best_pert,
                              args.criterions, args.window_size,
                              args.save_imgs, args.save_flow, args.save_pose,
                              eval_adv_img_dir, eval_adv_pert_dir, eval_flowdir, eval_pose_dir,
                              device=args.device)

    traj_adv_test_criterions_list = \
        test_adv_trajectories(test_dataloader, args.model, motions_target_test_list, attack, best_pert,
                              args.criterions, args.window_size,
                              args.save_imgs, args.save_flow, args.save_pose,
                              test_adv_img_dir, test_adv_pert_dir, test_flowdir, test_pose_dir,
                              device=args.device)

    del best_pert
    del eval_clean_loss_list
    del eval_all_loss_list
    del eval_all_best_loss_list

    del adv_eval_loss_list
    del adv_test_loss_list
    torch.cuda.empty_cache()
    gc.collect()

    return traj_adv_train_criterions_list, traj_adv_eval_criterions_list, traj_adv_test_criterions_list


def run_attacks_train(args):
    print("Training and testing an adversarial perturbation on the whole dataset")
    print("A single single universal will be produced and then tested on the dataset")
    print("args.attack")
    print(args.attack)
    datastr = args.datastr

    if args.custom_data:
        datastr += '_custom'
    elif args.real_data:
        datastr += '_real_data'
    attack = args.attack_obj

    dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, \
    motions_gt_list, traj_clean_criterions_list, traj_clean_motions = \
        test_clean_multi_inputs(args)

    print("traj_name_list")
    print(traj_name_list)

    motions_target_list = motions_gt_list
    if args.attack_target_clean:
        print("clean VO outputs will be used instead of ground truth, and clean deviation will be reconfigured to 0")
        motions_target_list = traj_clean_motions
        traj_clean_criterions_list = [[[0 for val in clean_crit_list] for clean_crit_list in traj_clean_crit_list]
                                       for traj_clean_crit_list in traj_clean_criterions_list]
    traj_clean_rms_list, traj_clean_mean_partial_rms_list, \
    traj_clean_target_rms_list, traj_clean_target_mean_partial_rms_list = tuple(traj_clean_criterions_list)


    best_pert, clean_loss_list, all_loss_list, all_best_loss_list = \
        attack.perturb(args.testDataloader, motions_target_list, eps=args.eps, device=args.device)

    print("clean_loss_list")
    print(clean_loss_list)
    print("all_loss_list")
    print(all_loss_list)
    print("all_best_loss_list")
    print(all_best_loss_list)
    best_loss_list = all_best_loss_list[- 1]
    print("best_loss_list")
    print(best_loss_list)

    if args.save_best_pert:
        save_image(best_pert[0], args.adv_best_pert_dir + '/' + 'adv_best_pert.png')

    traj_adv_criterions_list = \
        test_adv_trajectories(args.testDataloader, args.model, motions_target_list, attack, best_pert,
                              args.criterions, args.window_size,
                              args.save_imgs, args.save_flow, args.save_pose,
                              args.adv_img_dir, args.adv_pert_dir, args.flowdir, args.pose_dir,
                              device=args.device)
    traj_adv_rms_list, traj_adv_mean_partial_rms_list, \
    traj_adv_target_rms_list, traj_adv_target_mean_partial_rms_list = tuple(traj_adv_criterions_list)

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_mean_partial_rms_list, traj_adv_target_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_rms_list, traj_adv_target_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_mean_partial_rms_list, traj_adv_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_rms_list, traj_adv_rms_list,
                         args.save_csv, args.output_dir, crit_str="rms")


def run_attacks_train_test_random(args):
    print("Training and testing an adversarial perturbation on the whole dataset")
    print("A single single universal will be produced and then tested on the dataset")
    print("args.attack")
    print(args.attack)
    datastr = args.datastr

    if args.custom_data:
        datastr += '_custom'
    elif args.real_data:
        datastr += '_real_data'
    attack = args.attack_obj

    dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, \
    motions_gt_list, traj_clean_criterions_list, traj_clean_motions = \
        test_clean_multi_inputs(args)

    print("traj_name_list")
    print(traj_name_list)

    motions_target_list = motions_gt_list
    if args.attack_target_clean:
        print("clean VO outputs will be used instead of ground truth, and clean deviation will be reconfigured to 0")
        motions_target_list = traj_clean_motions
        traj_clean_criterions_list = [[[0 for val in clean_crit_list] for clean_crit_list in traj_clean_crit_list]
                                       for traj_clean_crit_list in traj_clean_criterions_list]
    traj_clean_rms_list, traj_clean_mean_partial_rms_list, \
    traj_clean_target_rms_list, traj_clean_target_mean_partial_rms_list = tuple(traj_clean_criterions_list)

    samples_traj_adv_criterions_list = []
    for random_sample_idx in range(args.test_random_samples):
        best_pert, clean_loss_list, all_loss_list, all_best_loss_list = \
            attack.perturb(args.testDataloader, motions_target_list, eps=args.eps, device=args.device)

        print("clean_loss_list")
        print(clean_loss_list)
        print("all_loss_list")
        print(all_loss_list)
        print("all_best_loss_list")
        print(all_best_loss_list)
        best_loss_list = all_best_loss_list[- 1]
        print("best_loss_list")
        print(best_loss_list)

        if args.save_best_pert:
            save_image(best_pert[0], args.adv_best_pert_dir + '/' + 'adv_best_pert_random_sample_idx_' +
                       str(random_sample_idx) + '.png')

        sample_traj_adv_criterions_list = \
            test_adv_trajectories(args.testDataloader, args.model, motions_target_list, attack, best_pert,
                                  args.criterions, args.window_size,
                                  args.save_imgs, args.save_flow, args.save_pose,
                                  args.adv_img_dir, args.adv_pert_dir, args.flowdir, args.pose_dir,
                                  device=args.device)
        samples_traj_adv_criterions_list.append(sample_traj_adv_criterions_list)

        del best_pert
        del clean_loss_list
        del all_loss_list
        del all_best_loss_list
        del best_loss_list
        torch.cuda.empty_cache()
        gc.collect()

    traj_adv_criterions_list = avg_criterions_samples(samples_traj_adv_criterions_list)

    del samples_traj_adv_criterions_list
    torch.cuda.empty_cache()
    gc.collect()

    traj_adv_rms_list, traj_adv_mean_partial_rms_list, \
    traj_adv_target_rms_list, traj_adv_target_mean_partial_rms_list = tuple(traj_adv_criterions_list)


    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_mean_partial_rms_list, traj_adv_target_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_rms_list, traj_adv_target_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_mean_partial_rms_list, traj_adv_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_rms_list, traj_adv_rms_list,
                         args.save_csv, args.output_dir, crit_str="rms")


def run_attacks_train_multi_perturb(args):
    print("Training and testing an adversarial perturbation on the whole dataset")
    print("A different perturbation will be produced for each trajectory")
    print("args.attack")
    print(args.attack)
    datastr = args.datastr

    if args.custom_data:
        datastr += '_custom'
    elif args.real_data:
        datastr += '_real_data'
    attack = args.attack_obj

    dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, \
    motions_gt_list, traj_clean_criterions_list, traj_clean_motions = \
        test_clean_multi_inputs(args)

    print("traj_name_list")
    print(traj_name_list)

    motions_target_list = motions_gt_list
    if args.attack_target_clean:
        print("clean VO outputs will be used instead of ground truth, and clean deviation will be reconfigured to 0")
        motions_target_list = traj_clean_motions
        traj_clean_criterions_list = [[[0 for val in clean_crit_list] for clean_crit_list in traj_clean_crit_list]
                                       for traj_clean_crit_list in traj_clean_criterions_list]
    traj_clean_rms_list, traj_clean_mean_partial_rms_list, \
    traj_clean_target_rms_list, traj_clean_target_mean_partial_rms_list = tuple(traj_clean_criterions_list)

    traj_best_pert_list = []
    clean_loss_list = []
    all_loss_list = [[] for k in range(args.attack_k + 1)]
    all_best_loss_list = [[] for k in range(args.attack_k + 1)]
    for traj_idx, traj in enumerate(args.testDataloader):
        traj_best_pert, traj_clean_loss_list, traj_all_loss_list, traj_all_best_loss_list = \
        attack.perturb([traj], [motions_target_list[traj_idx]], eps=args.eps, device=args.device)

        print("traj_clean_loss_list")
        print(traj_clean_loss_list)
        print("traj_all_loss_list")
        print(traj_all_loss_list)
        print("traj_all_best_loss_list")
        print(traj_all_best_loss_list)
        traj_best_loss_list = traj_all_best_loss_list[- 1]
        print("traj_best_loss_list")
        print(traj_best_loss_list)

        traj_best_pert_list.append(traj_best_pert)
        clean_loss_list += traj_clean_loss_list
        [all_loss_list[k].append(traj_all_loss_list[k][0]) for k in range(args.attack_k + 1)]
        [all_best_loss_list[k].append(traj_all_best_loss_list[k][0]) for k in range(args.attack_k + 1)]

        del traj_best_pert
        del traj_clean_loss_list
        del traj_all_loss_list
        del traj_all_best_loss_list
        torch.cuda.empty_cache()


    print("clean_loss_list")
    print(clean_loss_list)
    print("all_loss_list")
    print(all_loss_list)
    print("all_best_loss_list")
    print(all_best_loss_list)
    best_loss_list = all_best_loss_list[- 1]
    print("best_loss_list")
    print(best_loss_list)

    if args.save_best_pert:
        names = ["adv_best_pert_traj_" + traj_name for traj_name in traj_name_list]
        save_img_tensors(traj_best_pert_list, None, args.adv_best_pert_dir, names=names)

    traj_adv_criterions_list = \
        test_adv_trajectories(args.testDataloader, args.model, motions_target_list, attack, traj_best_pert_list,
                              args.criterions, args.window_size,
                              args.save_imgs, args.save_flow, args.save_pose,
                              args.adv_img_dir, args.adv_pert_dir, args.flowdir, args.pose_dir,
                              device=args.device, multi_perturb=True)
    traj_adv_rms_list, traj_adv_mean_partial_rms_list, \
    traj_adv_target_rms_list, traj_adv_target_mean_partial_rms_list = tuple(traj_adv_criterions_list)

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_mean_partial_rms_list, traj_adv_target_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_rms_list, traj_adv_target_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_mean_partial_rms_list, traj_adv_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_rms_list, traj_adv_rms_list,
                         args.save_csv, args.output_dir, crit_str="rms")


def test_clean(args):
    print("Testing the visual odometer on the I1 albedo image compared to the clean I0 albedo image, "
          "I1 is set as the adversarial perturbation")
    return run_attacks_train(args)


def run_attacks_out_of_sample_test(args):
    test_perturb_trajectories = adv_oos_test_perturb_trajectories
    print("Testing the generalization of the adversarial attack to different data folders")
    print("In each iteration, the data folders are split between a train-set, eval-set "
          "and a test-set containing a single data folder")
    print("An adversarial perturbation will be trained on the train-set, "
          "and the best performing perturbation on the eval-set will be chosen")
    print("We report the average performance of the best perturbation over the test-sets")
    print("A single universal perturbation will be produced and then tested on the dataset")
    print("args.attack")
    print(args.attack)
    datastr = args.datastr

    if args.custom_data:
        datastr += '_custom'
    elif args.real_data:
        datastr += '_real_data'

    attack = args.attack_obj

    dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, \
    motions_gt_list, traj_clean_criterions_list, traj_clean_motions = \
        test_clean_multi_inputs(args)

    motions_target_list = motions_gt_list
    if args.attack_target_clean:
        print("clean VO outputs will be used instead of ground truth, and clean deviation will be reconfigured to 0")
        motions_target_list = traj_clean_motions
        traj_clean_criterions_list = [[[0 for val in clean_crit_list] for clean_crit_list in traj_clean_crit_list]
                                       for traj_clean_crit_list in traj_clean_criterions_list]
    traj_clean_rms_list, traj_clean_mean_partial_rms_list, \
    traj_clean_target_rms_list, traj_clean_target_mean_partial_rms_list = tuple(traj_clean_criterions_list)

    folds_num = args.traj_datasets
    folds_indices_list = list(range(folds_num))
    print("evaluating over " + str(folds_num) + " datasets folds")
    print("folds_indices_list")
    print(folds_indices_list)

    folds_frames_clean_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_train_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_train_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_target_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_eval_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_eval_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_target_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_test_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_test_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_test_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_test_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_test_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_test_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_test_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_test_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_test_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_test_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_test_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_test_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_test_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_test_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_test_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_test_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_test_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_test_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_test_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_test_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_test_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_test_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_test_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_test_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_test_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_test_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_test_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_test_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_test_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_test_target_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_test_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_test_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_test_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_test_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_test_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_test_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_test_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_test_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_test_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_test_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    for test_fold_idx in folds_indices_list:
        output_dir, adv_best_pert_dir, \
        train_output_dir, train_flowdir, train_pose_dir, train_img_dir, train_adv_img_dir, train_adv_pert_dir, \
        eval_output_dir, eval_flowdir, eval_pose_dir, eval_img_dir, eval_adv_img_dir, eval_adv_pert_dir, \
        test_output_dir, test_flowdir, test_pose_dir, test_img_dir, test_adv_img_dir, test_adv_pert_dir = \
            out_of_sample_make_folders(args.output_dir, test_fold_idx, args.save_results,
                                       args.save_best_pert, args.save_flow, args.save_pose, args.save_imgs)

        test_indices = [traj_idx for traj_idx in traj_indices if dataset_idx_list[traj_idx] == test_fold_idx]
        test_dataset = args.dataset_class(args.test_dir, processed_data_folder=args.processed_data_dir,
                                                    preprocessed_data=True,
                                                    transform=args.transform,
                                                    data_size=(args.image_height, args.image_width),
                                                    focalx=args.focalx,
                                                    focaly=args.focaly,
                                                    centerx=args.centerx,
                                                    centery=args.centery,
                                                    max_traj_len=args.max_traj_len,
                                                    max_dataset_traj_num=args.max_traj_num,
                                                    max_traj_datasets=args.max_traj_datasets,
                                                    folder_indices_list=[test_fold_idx])
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.worker_num)

        while True:
            eval_fold_idx = random.choice(folds_indices_list)
            if eval_fold_idx != test_fold_idx:
                break
        eval_indices = [traj_idx for traj_idx in traj_indices if dataset_idx_list[traj_idx] == eval_fold_idx]
        eval_dataset = args.dataset_class(args.test_dir, processed_data_folder=args.processed_data_dir,
                                                    preprocessed_data=True,
                                                    transform=args.transform,
                                                    data_size=(args.image_height, args.image_width),
                                                    focalx=args.focalx,
                                                    focaly=args.focaly,
                                                    centerx=args.centerx,
                                                    centery=args.centery,
                                                    max_traj_len=args.max_traj_len,
                                                    max_dataset_traj_num=args.max_traj_num,
                                                    max_traj_datasets=args.max_traj_datasets,
                                                    folder_indices_list=[eval_fold_idx])
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.worker_num)

        train_indices = [traj_idx for traj_idx in traj_indices
                         if (traj_idx not in test_indices and traj_idx not in eval_indices)]
        train_folders_indices = [fold_idx for fold_idx in folds_indices_list
                                 if (fold_idx != test_fold_idx and fold_idx != eval_fold_idx)]
        train_dataset = args.dataset_class(args.test_dir, processed_data_folder=args.processed_data_dir,
                                                    preprocessed_data=True,
                                                    transform=args.transform,
                                                    data_size=(args.image_height, args.image_width),
                                                    focalx=args.focalx,
                                                    focaly=args.focaly,
                                                    centerx=args.centerx,
                                                    centery=args.centery,
                                                    max_traj_len=args.max_traj_len,
                                                    max_dataset_traj_num=args.max_traj_num,
                                                    max_traj_datasets=args.max_traj_datasets,
                                                    folder_indices_list=train_folders_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.worker_num)

        print("test_fold_idx")
        print(test_fold_idx)
        print("eval_fold_idx")
        print(eval_fold_idx)
        print("train_folders_indices")
        print(train_folders_indices)

        train_dataset_indices = masked_list(dataset_idx_list, train_indices)
        train_dataset_names = masked_list(dataset_name_list, train_indices)
        train_traj_names = masked_list(traj_name_list, train_indices)
        motions_target_train_list = masked_list(motions_target_list, train_indices)
        traj_clean_train_rms_list = masked_list(traj_clean_rms_list, train_indices)
        traj_clean_train_mean_partial_rms_list = masked_list(traj_clean_mean_partial_rms_list, train_indices)
        traj_clean_train_target_rms_list = masked_list(traj_clean_target_rms_list, train_indices)
        traj_clean_train_target_mean_partial_rms_list = masked_list(traj_clean_target_mean_partial_rms_list, train_indices)

        eval_dataset_indices = masked_list(dataset_idx_list, eval_indices)
        eval_dataset_names = masked_list(dataset_name_list, eval_indices)
        eval_traj_names = masked_list(traj_name_list, eval_indices)
        motions_target_eval_list = masked_list(motions_target_list, eval_indices)
        traj_clean_eval_rms_list = masked_list(traj_clean_rms_list, eval_indices)
        traj_clean_eval_mean_partial_rms_list = masked_list(traj_clean_mean_partial_rms_list, eval_indices)
        traj_clean_eval_target_rms_list = masked_list(traj_clean_target_rms_list, eval_indices)
        traj_clean_eval_target_mean_partial_rms_list = masked_list(traj_clean_target_mean_partial_rms_list, eval_indices)

        test_dataset_indices = masked_list(dataset_idx_list, test_indices)
        test_dataset_names = masked_list(dataset_name_list, test_indices)
        test_traj_names = masked_list(traj_name_list, test_indices)
        motions_target_test_list = masked_list(motions_target_list, test_indices)
        traj_clean_test_rms_list = masked_list(traj_clean_rms_list, test_indices)
        traj_clean_test_mean_partial_rms_list = masked_list(traj_clean_mean_partial_rms_list, test_indices)
        traj_clean_test_target_rms_list = masked_list(traj_clean_target_rms_list, test_indices)
        traj_clean_test_target_mean_partial_rms_list = masked_list(traj_clean_target_mean_partial_rms_list, test_indices)

        traj_adv_train_criterions_list, traj_adv_eval_criterions_list, traj_adv_test_criterions_list = \
            test_perturb_trajectories(args, train_traj_names, eval_traj_names, test_traj_names,
                                          attack, train_dataloader, motions_target_train_list,
                                          eval_dataloader, motions_target_eval_list,
                                          test_dataloader, motions_target_test_list,
                                          adv_best_pert_dir, train_adv_img_dir, train_adv_pert_dir, train_flowdir,
                                          train_pose_dir,
                                          eval_adv_img_dir, eval_adv_pert_dir, eval_flowdir, eval_pose_dir,
                                          test_adv_img_dir, test_adv_pert_dir, test_flowdir, test_pose_dir)

        traj_adv_train_rms_list, traj_adv_train_mean_partial_rms_list, \
        traj_adv_train_target_rms_list, traj_adv_train_target_mean_partial_rms_list = traj_adv_train_criterions_list
        traj_adv_eval_rms_list, traj_adv_eval_mean_partial_rms_list, \
        traj_adv_eval_target_rms_list, traj_adv_eval_target_mean_partial_rms_list = traj_adv_eval_criterions_list
        traj_adv_test_rms_list, traj_adv_test_mean_partial_rms_list, \
        traj_adv_test_target_rms_list, traj_adv_test_target_mean_partial_rms_list = traj_adv_test_criterions_list


        frames_clean_train_target_mean_partial_rms_mean, frames_clean_train_target_mean_partial_rms_std, \
        frames_adv_train_target_mean_partial_rms_mean, frames_adv_train_target_mean_partial_rms_std, \
        frames_delta_train_target_mean_partial_rms_mean, frames_delta_train_target_mean_partial_rms_std, \
        frames_ratio_train_target_mean_partial_rms_mean, frames_ratio_train_target_mean_partial_rms_std, \
        frames_delta_ratio_train_target_mean_partial_rms_mean, frames_delta_ratio_train_target_mean_partial_rms_std = \
            report_adv_deviation(train_dataset_indices, train_dataset_names, train_traj_names, train_indices,
                                 traj_clean_train_target_mean_partial_rms_list, traj_adv_train_target_mean_partial_rms_list,
                                 args.save_csv, train_output_dir, crit_str="target_mean_partial_rms",
                                 experiment_name="train")
        frames_clean_train_target_rms_mean, frames_clean_train_target_rms_std, \
        frames_adv_train_target_rms_mean, frames_adv_train_target_rms_std, \
        frames_delta_train_target_rms_mean, frames_delta_train_target_rms_std, \
        frames_ratio_train_target_rms_mean, frames_ratio_train_target_rms_std, \
        frames_delta_ratio_train_target_rms_mean, frames_delta_ratio_train_target_rms_std = \
            report_adv_deviation(train_dataset_indices, train_dataset_names, train_traj_names, train_indices,
                                 traj_clean_train_target_rms_list, traj_adv_train_target_rms_list,
                                 args.save_csv, train_output_dir, crit_str="target_rms",
                                 experiment_name="train")

        frames_clean_train_mean_partial_rms_mean, frames_clean_train_mean_partial_rms_std, \
        frames_adv_train_mean_partial_rms_mean, frames_adv_train_mean_partial_rms_std, \
        frames_delta_train_mean_partial_rms_mean, frames_delta_train_mean_partial_rms_std, \
        frames_ratio_train_mean_partial_rms_mean, frames_ratio_train_mean_partial_rms_std, \
        frames_delta_ratio_train_mean_partial_rms_mean, frames_delta_ratio_train_mean_partial_rms_std = \
            report_adv_deviation(train_dataset_indices, train_dataset_names, train_traj_names, train_indices,
                                 traj_clean_train_mean_partial_rms_list, traj_adv_train_mean_partial_rms_list,
                                 args.save_csv, train_output_dir, crit_str="mean_partial_rms",
                                 experiment_name="train")

        frames_clean_train_rms_mean, frames_clean_train_rms_std, \
        frames_adv_train_rms_mean, frames_adv_train_rms_std, \
        frames_delta_train_rms_mean, frames_delta_train_rms_std, \
        frames_ratio_train_rms_mean, frames_ratio_train_rms_std, \
        frames_delta_ratio_train_rms_mean, frames_delta_ratio_train_rms_std = \
            report_adv_deviation(train_dataset_indices, train_dataset_names, train_traj_names, train_indices,
                                 traj_clean_train_rms_list, traj_adv_train_rms_list,
                                 args.save_csv, train_output_dir, crit_str="rms",
                                 experiment_name="train")

        frames_clean_eval_target_mean_partial_rms_mean, frames_clean_eval_target_mean_partial_rms_std, \
        frames_adv_eval_target_mean_partial_rms_mean, frames_adv_eval_target_mean_partial_rms_std, \
        frames_delta_eval_target_mean_partial_rms_mean, frames_delta_eval_target_mean_partial_rms_std, \
        frames_ratio_eval_target_mean_partial_rms_mean, frames_ratio_eval_target_mean_partial_rms_std, \
        frames_delta_ratio_eval_target_mean_partial_rms_mean, frames_delta_ratio_eval_target_mean_partial_rms_std = \
            report_adv_deviation(eval_dataset_indices, eval_dataset_names, eval_traj_names, eval_indices,
                                 traj_clean_eval_target_mean_partial_rms_list, traj_adv_eval_target_mean_partial_rms_list,
                                 args.save_csv, eval_output_dir, crit_str="target_mean_partial_rms",
                                 experiment_name="eval")
        frames_clean_eval_target_rms_mean, frames_clean_eval_target_rms_std, \
        frames_adv_eval_target_rms_mean, frames_adv_eval_target_rms_std, \
        frames_delta_eval_target_rms_mean, frames_delta_eval_target_rms_std, \
        frames_ratio_eval_target_rms_mean, frames_ratio_eval_target_rms_std, \
        frames_delta_ratio_eval_target_rms_mean, frames_delta_ratio_eval_target_rms_std = \
            report_adv_deviation(eval_dataset_indices, eval_dataset_names, eval_traj_names, eval_indices,
                                 traj_clean_eval_target_rms_list, traj_adv_eval_target_rms_list,
                                 args.save_csv, eval_output_dir, crit_str="target_rms",
                                 experiment_name="eval")

        frames_clean_eval_mean_partial_rms_mean, frames_clean_eval_mean_partial_rms_std, \
        frames_adv_eval_mean_partial_rms_mean, frames_adv_eval_mean_partial_rms_std, \
        frames_delta_eval_mean_partial_rms_mean, frames_delta_eval_mean_partial_rms_std, \
        frames_ratio_eval_mean_partial_rms_mean, frames_ratio_eval_mean_partial_rms_std, \
        frames_delta_ratio_eval_mean_partial_rms_mean, frames_delta_ratio_eval_mean_partial_rms_std = \
            report_adv_deviation(eval_dataset_indices, eval_dataset_names, eval_traj_names, eval_indices,
                                 traj_clean_eval_mean_partial_rms_list, traj_adv_eval_mean_partial_rms_list,
                                 args.save_csv, eval_output_dir, crit_str="mean_partial_rms",
                                 experiment_name="eval")

        frames_clean_eval_rms_mean, frames_clean_eval_rms_std, \
        frames_adv_eval_rms_mean, frames_adv_eval_rms_std, \
        frames_delta_eval_rms_mean, frames_delta_eval_rms_std, \
        frames_ratio_eval_rms_mean, frames_ratio_eval_rms_std, \
        frames_delta_ratio_eval_rms_mean, frames_delta_ratio_eval_rms_std = \
            report_adv_deviation(eval_dataset_indices, eval_dataset_names, eval_traj_names, eval_indices,
                                 traj_clean_eval_rms_list, traj_adv_eval_rms_list,
                                 args.save_csv, eval_output_dir, crit_str="rms",
                                 experiment_name="eval")

        frames_clean_test_target_mean_partial_rms_mean, frames_clean_test_target_mean_partial_rms_std, \
        frames_adv_test_target_mean_partial_rms_mean, frames_adv_test_target_mean_partial_rms_std, \
        frames_delta_test_target_mean_partial_rms_mean, frames_delta_test_target_mean_partial_rms_std, \
        frames_ratio_test_target_mean_partial_rms_mean, frames_ratio_test_target_mean_partial_rms_std, \
        frames_delta_ratio_test_target_mean_partial_rms_mean, frames_delta_ratio_test_target_mean_partial_rms_std = \
            report_adv_deviation(test_dataset_indices, test_dataset_names, test_traj_names, test_indices,
                                 traj_clean_test_target_mean_partial_rms_list, traj_adv_test_target_mean_partial_rms_list,
                                 args.save_csv, test_output_dir, crit_str="target_mean_partial_rms",
                                 experiment_name="test")
        frames_clean_test_target_rms_mean, frames_clean_test_target_rms_std, \
        frames_adv_test_target_rms_mean, frames_adv_test_target_rms_std, \
        frames_delta_test_target_rms_mean, frames_delta_test_target_rms_std, \
        frames_ratio_test_target_rms_mean, frames_ratio_test_target_rms_std, \
        frames_delta_ratio_test_target_rms_mean, frames_delta_ratio_test_target_rms_std = \
            report_adv_deviation(test_dataset_indices, test_dataset_names, test_traj_names, test_indices,
                                 traj_clean_test_target_rms_list, traj_adv_test_target_rms_list,
                                 args.save_csv, test_output_dir, crit_str="target_rms",
                                 experiment_name="test")

        frames_clean_test_mean_partial_rms_mean, frames_clean_test_mean_partial_rms_std, \
        frames_adv_test_mean_partial_rms_mean, frames_adv_test_mean_partial_rms_std, \
        frames_delta_test_mean_partial_rms_mean, frames_delta_test_mean_partial_rms_std, \
        frames_ratio_test_mean_partial_rms_mean, frames_ratio_test_mean_partial_rms_std, \
        frames_delta_ratio_test_mean_partial_rms_mean, frames_delta_ratio_test_mean_partial_rms_std = \
            report_adv_deviation(test_dataset_indices, test_dataset_names, test_traj_names, test_indices,
                                 traj_clean_test_mean_partial_rms_list, traj_adv_test_mean_partial_rms_list,
                                 args.save_csv, test_output_dir, crit_str="mean_partial_rms",
                                 experiment_name="test")

        frames_clean_test_rms_mean, frames_clean_test_rms_std, \
        frames_adv_test_rms_mean, frames_adv_test_rms_std, \
        frames_delta_test_rms_mean, frames_delta_test_rms_std, \
        frames_ratio_test_rms_mean, frames_ratio_test_rms_std, \
        frames_delta_ratio_test_rms_mean, frames_delta_ratio_test_rms_std = \
            report_adv_deviation(test_dataset_indices, test_dataset_names, test_traj_names, test_indices,
                                 traj_clean_test_rms_list, traj_adv_test_rms_list,
                                 args.save_csv, test_output_dir, crit_str="rms",
                                 experiment_name="test")


        del train_dataset_indices
        del train_dataset_names
        del train_traj_names
        del motions_target_train_list
        del traj_clean_train_rms_list
        del traj_clean_train_mean_partial_rms_list
        del traj_clean_train_target_rms_list
        del traj_clean_train_target_mean_partial_rms_list

        del eval_dataset_indices
        del eval_dataset_names
        del eval_traj_names
        del motions_target_eval_list
        del traj_clean_eval_rms_list
        del traj_clean_eval_mean_partial_rms_list
        del traj_clean_eval_target_rms_list
        del traj_clean_eval_target_mean_partial_rms_list

        del test_dataset_indices
        del test_dataset_names
        del test_traj_names
        del motions_target_test_list
        del traj_clean_test_rms_list
        del traj_clean_test_mean_partial_rms_list
        del traj_clean_test_target_rms_list
        del traj_clean_test_target_mean_partial_rms_list

        torch.cuda.empty_cache()
        gc.collect()

        collect_and_report_folds_mean_result(folds_frames_clean_train_target_mean_partial_rms_means,
                                        folds_frames_clean_train_target_mean_partial_rms_stds,
                                        folds_frames_adv_train_target_mean_partial_rms_means,
                                        folds_frames_adv_train_target_mean_partial_rms_stds,
                                        folds_frames_delta_train_target_mean_partial_rms_means,
                                        folds_frames_delta_train_target_mean_partial_rms_stds,
                                        folds_frames_ratio_train_target_mean_partial_rms_means,
                                        folds_frames_ratio_train_target_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_train_target_mean_partial_rms_means,
                                        folds_frames_delta_ratio_train_target_mean_partial_rms_stds,
                                        frames_clean_train_target_mean_partial_rms_mean,
                                        frames_clean_train_target_mean_partial_rms_std,
                                        frames_adv_train_target_mean_partial_rms_mean,
                                        frames_adv_train_target_mean_partial_rms_std,
                                        frames_delta_train_target_mean_partial_rms_mean,
                                        frames_delta_train_target_mean_partial_rms_std,
                                        frames_ratio_train_target_mean_partial_rms_mean,
                                        frames_ratio_train_target_mean_partial_rms_std,
                                        frames_delta_ratio_train_target_mean_partial_rms_mean,
                                        frames_delta_ratio_train_target_mean_partial_rms_std,
                                        crit_str="train_target_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_train_target_rms_means,
                                        folds_frames_clean_train_target_rms_stds,
                                        folds_frames_adv_train_target_rms_means,
                                        folds_frames_adv_train_target_rms_stds,
                                        folds_frames_delta_train_target_rms_means,
                                        folds_frames_delta_train_target_rms_stds,
                                        folds_frames_ratio_train_target_rms_means,
                                        folds_frames_ratio_train_target_rms_stds,
                                        folds_frames_delta_ratio_train_target_rms_means,
                                        folds_frames_delta_ratio_train_target_rms_stds,
                                        frames_clean_train_target_rms_mean,
                                        frames_clean_train_target_rms_std,
                                        frames_adv_train_target_rms_mean,
                                        frames_adv_train_target_rms_std,
                                        frames_delta_train_target_rms_mean,
                                        frames_delta_train_target_rms_std,
                                        frames_ratio_train_target_rms_mean,
                                        frames_ratio_train_target_rms_std,
                                        frames_delta_ratio_train_target_rms_mean,
                                        frames_delta_ratio_train_target_rms_std,
                                        crit_str="train_target_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_train_mean_partial_rms_means,
                                        folds_frames_clean_train_mean_partial_rms_stds,
                                        folds_frames_adv_train_mean_partial_rms_means,
                                        folds_frames_adv_train_mean_partial_rms_stds,
                                        folds_frames_delta_train_mean_partial_rms_means,
                                        folds_frames_delta_train_mean_partial_rms_stds,
                                        folds_frames_ratio_train_mean_partial_rms_means,
                                        folds_frames_ratio_train_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_train_mean_partial_rms_means,
                                        folds_frames_delta_ratio_train_mean_partial_rms_stds,
                                        frames_clean_train_mean_partial_rms_mean,
                                        frames_clean_train_mean_partial_rms_std,
                                        frames_adv_train_mean_partial_rms_mean,
                                        frames_adv_train_mean_partial_rms_std,
                                        frames_delta_train_mean_partial_rms_mean,
                                        frames_delta_train_mean_partial_rms_std,
                                        frames_ratio_train_mean_partial_rms_mean,
                                        frames_ratio_train_mean_partial_rms_std,
                                        frames_delta_ratio_train_mean_partial_rms_mean,
                                        frames_delta_ratio_train_mean_partial_rms_std,
                                        crit_str="train_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_train_rms_means,
                                        folds_frames_clean_train_rms_stds,
                                        folds_frames_adv_train_rms_means,
                                        folds_frames_adv_train_rms_stds,
                                        folds_frames_delta_train_rms_means,
                                        folds_frames_delta_train_rms_stds,
                                        folds_frames_ratio_train_rms_means,
                                        folds_frames_ratio_train_rms_stds,
                                        folds_frames_delta_ratio_train_rms_means,
                                        folds_frames_delta_ratio_train_rms_stds,
                                        frames_clean_train_rms_mean,
                                        frames_clean_train_rms_std,
                                        frames_adv_train_rms_mean,
                                        frames_adv_train_rms_std,
                                        frames_delta_train_rms_mean,
                                        frames_delta_train_rms_std,
                                        frames_ratio_train_rms_mean,
                                        frames_ratio_train_rms_std,
                                        frames_delta_ratio_train_rms_mean,
                                        frames_delta_ratio_train_rms_std,
                                        crit_str="train_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_eval_target_mean_partial_rms_means,
                                        folds_frames_clean_eval_target_mean_partial_rms_stds,
                                        folds_frames_adv_eval_target_mean_partial_rms_means,
                                        folds_frames_adv_eval_target_mean_partial_rms_stds,
                                        folds_frames_delta_eval_target_mean_partial_rms_means,
                                        folds_frames_delta_eval_target_mean_partial_rms_stds,
                                        folds_frames_ratio_eval_target_mean_partial_rms_means,
                                        folds_frames_ratio_eval_target_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_eval_target_mean_partial_rms_means,
                                        folds_frames_delta_ratio_eval_target_mean_partial_rms_stds,
                                        frames_clean_eval_target_mean_partial_rms_mean,
                                        frames_clean_eval_target_mean_partial_rms_std,
                                        frames_adv_eval_target_mean_partial_rms_mean,
                                        frames_adv_eval_target_mean_partial_rms_std,
                                        frames_delta_eval_target_mean_partial_rms_mean,
                                        frames_delta_eval_target_mean_partial_rms_std,
                                        frames_ratio_eval_target_mean_partial_rms_mean,
                                        frames_ratio_eval_target_mean_partial_rms_std,
                                        frames_delta_ratio_eval_target_mean_partial_rms_mean,
                                        frames_delta_ratio_eval_target_mean_partial_rms_std,
                                        crit_str="eval_target_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_eval_target_rms_means,
                                folds_frames_clean_eval_target_rms_stds,
                                folds_frames_adv_eval_target_rms_means,
                                folds_frames_adv_eval_target_rms_stds,
                                folds_frames_delta_eval_target_rms_means,
                                folds_frames_delta_eval_target_rms_stds,
                                folds_frames_ratio_eval_target_rms_means,
                                folds_frames_ratio_eval_target_rms_stds,
                                folds_frames_delta_ratio_eval_target_rms_means,
                                folds_frames_delta_ratio_eval_target_rms_stds,
                                frames_clean_eval_target_rms_mean,
                                frames_clean_eval_target_rms_std,
                                frames_adv_eval_target_rms_mean,
                                frames_adv_eval_target_rms_std,
                                frames_delta_eval_target_rms_mean,
                                frames_delta_eval_target_rms_std,
                                frames_ratio_eval_target_rms_mean,
                                frames_ratio_eval_target_rms_std,
                                frames_delta_ratio_eval_target_rms_mean,
                                frames_delta_ratio_eval_target_rms_std,
                                crit_str="eval_target_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_eval_mean_partial_rms_means,
                                        folds_frames_clean_eval_mean_partial_rms_stds,
                                        folds_frames_adv_eval_mean_partial_rms_means,
                                        folds_frames_adv_eval_mean_partial_rms_stds,
                                        folds_frames_delta_eval_mean_partial_rms_means,
                                        folds_frames_delta_eval_mean_partial_rms_stds,
                                        folds_frames_ratio_eval_mean_partial_rms_means,
                                        folds_frames_ratio_eval_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_eval_mean_partial_rms_means,
                                        folds_frames_delta_ratio_eval_mean_partial_rms_stds,
                                        frames_clean_eval_mean_partial_rms_mean,
                                        frames_clean_eval_mean_partial_rms_std,
                                        frames_adv_eval_mean_partial_rms_mean,
                                        frames_adv_eval_mean_partial_rms_std,
                                        frames_delta_eval_mean_partial_rms_mean,
                                        frames_delta_eval_mean_partial_rms_std,
                                        frames_ratio_eval_mean_partial_rms_mean,
                                        frames_ratio_eval_mean_partial_rms_std,
                                        frames_delta_ratio_eval_mean_partial_rms_mean,
                                        frames_delta_ratio_eval_mean_partial_rms_std,
                                        crit_str="eval_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_eval_rms_means,
                                folds_frames_clean_eval_rms_stds,
                                folds_frames_adv_eval_rms_means,
                                folds_frames_adv_eval_rms_stds,
                                folds_frames_delta_eval_rms_means,
                                folds_frames_delta_eval_rms_stds,
                                folds_frames_ratio_eval_rms_means,
                                folds_frames_ratio_eval_rms_stds,
                                folds_frames_delta_ratio_eval_rms_means,
                                folds_frames_delta_ratio_eval_rms_stds,
                                frames_clean_eval_rms_mean,
                                frames_clean_eval_rms_std,
                                frames_adv_eval_rms_mean,
                                frames_adv_eval_rms_std,
                                frames_delta_eval_rms_mean,
                                frames_delta_eval_rms_std,
                                frames_ratio_eval_rms_mean,
                                frames_ratio_eval_rms_std,
                                frames_delta_ratio_eval_rms_mean,
                                frames_delta_ratio_eval_rms_std,
                                crit_str="eval_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_test_target_mean_partial_rms_means,
                                        folds_frames_clean_test_target_mean_partial_rms_stds,
                                        folds_frames_adv_test_target_mean_partial_rms_means,
                                        folds_frames_adv_test_target_mean_partial_rms_stds,
                                        folds_frames_delta_test_target_mean_partial_rms_means,
                                        folds_frames_delta_test_target_mean_partial_rms_stds,
                                        folds_frames_ratio_test_target_mean_partial_rms_means,
                                        folds_frames_ratio_test_target_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_test_target_mean_partial_rms_means,
                                        folds_frames_delta_ratio_test_target_mean_partial_rms_stds,
                                        frames_clean_test_target_mean_partial_rms_mean,
                                        frames_clean_test_target_mean_partial_rms_std,
                                        frames_adv_test_target_mean_partial_rms_mean,
                                        frames_adv_test_target_mean_partial_rms_std,
                                        frames_delta_test_target_mean_partial_rms_mean,
                                        frames_delta_test_target_mean_partial_rms_std,
                                        frames_ratio_test_target_mean_partial_rms_mean,
                                        frames_ratio_test_target_mean_partial_rms_std,
                                        frames_delta_ratio_test_target_mean_partial_rms_mean,
                                        frames_delta_ratio_test_target_mean_partial_rms_std,
                                        crit_str="test_target_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_test_target_rms_means,
                                folds_frames_clean_test_target_rms_stds,
                                folds_frames_adv_test_target_rms_means,
                                folds_frames_adv_test_target_rms_stds,
                                folds_frames_delta_test_target_rms_means,
                                folds_frames_delta_test_target_rms_stds,
                                folds_frames_ratio_test_target_rms_means,
                                folds_frames_ratio_test_target_rms_stds,
                                folds_frames_delta_ratio_test_target_rms_means,
                                folds_frames_delta_ratio_test_target_rms_stds,
                                frames_clean_test_target_rms_mean,
                                frames_clean_test_target_rms_std,
                                frames_adv_test_target_rms_mean,
                                frames_adv_test_target_rms_std,
                                frames_delta_test_target_rms_mean,
                                frames_delta_test_target_rms_std,
                                frames_ratio_test_target_rms_mean,
                                frames_ratio_test_target_rms_std,
                                frames_delta_ratio_test_target_rms_mean,
                                frames_delta_ratio_test_target_rms_std,
                                crit_str="test_target_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_test_mean_partial_rms_means,
                                        folds_frames_clean_test_mean_partial_rms_stds,
                                        folds_frames_adv_test_mean_partial_rms_means,
                                        folds_frames_adv_test_mean_partial_rms_stds,
                                        folds_frames_delta_test_mean_partial_rms_means,
                                        folds_frames_delta_test_mean_partial_rms_stds,
                                        folds_frames_ratio_test_mean_partial_rms_means,
                                        folds_frames_ratio_test_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_test_mean_partial_rms_means,
                                        folds_frames_delta_ratio_test_mean_partial_rms_stds,
                                        frames_clean_test_mean_partial_rms_mean,
                                        frames_clean_test_mean_partial_rms_std,
                                        frames_adv_test_mean_partial_rms_mean,
                                        frames_adv_test_mean_partial_rms_std,
                                        frames_delta_test_mean_partial_rms_mean,
                                        frames_delta_test_mean_partial_rms_std,
                                        frames_ratio_test_mean_partial_rms_mean,
                                        frames_ratio_test_mean_partial_rms_std,
                                        frames_delta_ratio_test_mean_partial_rms_mean,
                                        frames_delta_ratio_test_mean_partial_rms_std,
                                        crit_str="test_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_test_rms_means,
                                folds_frames_clean_test_rms_stds,
                                folds_frames_adv_test_rms_means,
                                folds_frames_adv_test_rms_stds,
                                folds_frames_delta_test_rms_means,
                                folds_frames_delta_test_rms_stds,
                                folds_frames_ratio_test_rms_means,
                                folds_frames_ratio_test_rms_stds,
                                folds_frames_delta_ratio_test_rms_means,
                                folds_frames_delta_ratio_test_rms_stds,
                                frames_clean_test_rms_mean,
                                frames_clean_test_rms_std,
                                frames_adv_test_rms_mean,
                                frames_adv_test_rms_std,
                                frames_delta_test_rms_mean,
                                frames_delta_test_rms_std,
                                frames_ratio_test_rms_mean,
                                frames_ratio_test_rms_std,
                                frames_delta_ratio_test_rms_mean,
                                frames_delta_ratio_test_rms_std,
                                crit_str="test_rms")

    report_mean_results(folds_frames_clean_train_target_mean_partial_rms_means,
                                    folds_frames_clean_train_target_mean_partial_rms_stds,
                                    folds_frames_adv_train_target_mean_partial_rms_means,
                                    folds_frames_adv_train_target_mean_partial_rms_stds,
                                    folds_frames_delta_train_target_mean_partial_rms_means,
                                    folds_frames_delta_train_target_mean_partial_rms_stds,
                                    folds_frames_ratio_train_target_mean_partial_rms_means,
                                    folds_frames_ratio_train_target_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_train_target_mean_partial_rms_means,
                                    folds_frames_delta_ratio_train_target_mean_partial_rms_stds,
                                    crit_str="train_target_mean_partial_rms")

    report_mean_results(folds_frames_clean_train_target_rms_means,
                                    folds_frames_clean_train_target_rms_stds,
                                    folds_frames_adv_train_target_rms_means,
                                    folds_frames_adv_train_target_rms_stds,
                                    folds_frames_delta_train_target_rms_means,
                                    folds_frames_delta_train_target_rms_stds,
                                    folds_frames_ratio_train_target_rms_means,
                                    folds_frames_ratio_train_target_rms_stds,
                                    folds_frames_delta_ratio_train_target_rms_means,
                                    folds_frames_delta_ratio_train_target_rms_stds,
                                    crit_str="train_target_rms")

    report_mean_results(folds_frames_clean_train_mean_partial_rms_means,
                                    folds_frames_clean_train_mean_partial_rms_stds,
                                    folds_frames_adv_train_mean_partial_rms_means,
                                    folds_frames_adv_train_mean_partial_rms_stds,
                                    folds_frames_delta_train_mean_partial_rms_means,
                                    folds_frames_delta_train_mean_partial_rms_stds,
                                    folds_frames_ratio_train_mean_partial_rms_means,
                                    folds_frames_ratio_train_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_train_mean_partial_rms_means,
                                    folds_frames_delta_ratio_train_mean_partial_rms_stds,
                                    crit_str="train_mean_partial_rms")

    report_mean_results(folds_frames_clean_train_rms_means,
                                    folds_frames_clean_train_rms_stds,
                                    folds_frames_adv_train_rms_means,
                                    folds_frames_adv_train_rms_stds,
                                    folds_frames_delta_train_rms_means,
                                    folds_frames_delta_train_rms_stds,
                                    folds_frames_ratio_train_rms_means,
                                    folds_frames_ratio_train_rms_stds,
                                    folds_frames_delta_ratio_train_rms_means,
                                    folds_frames_delta_ratio_train_rms_stds,
                                    crit_str="train_rms")

    report_mean_results(folds_frames_clean_eval_target_mean_partial_rms_means,
                                    folds_frames_clean_eval_target_mean_partial_rms_stds,
                                    folds_frames_adv_eval_target_mean_partial_rms_means,
                                    folds_frames_adv_eval_target_mean_partial_rms_stds,
                                    folds_frames_delta_eval_target_mean_partial_rms_means,
                                    folds_frames_delta_eval_target_mean_partial_rms_stds,
                                    folds_frames_ratio_eval_target_mean_partial_rms_means,
                                    folds_frames_ratio_eval_target_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_eval_target_mean_partial_rms_means,
                                    folds_frames_delta_ratio_eval_target_mean_partial_rms_stds,
                                    crit_str="eval_target_mean_partial_rms")

    report_mean_results(folds_frames_clean_eval_target_rms_means,
                                    folds_frames_clean_eval_target_rms_stds,
                                    folds_frames_adv_eval_target_rms_means,
                                    folds_frames_adv_eval_target_rms_stds,
                                    folds_frames_delta_eval_target_rms_means,
                                    folds_frames_delta_eval_target_rms_stds,
                                    folds_frames_ratio_eval_target_rms_means,
                                    folds_frames_ratio_eval_target_rms_stds,
                                    folds_frames_delta_ratio_eval_target_rms_means,
                                    folds_frames_delta_ratio_eval_target_rms_stds,
                                    crit_str="eval_target_rms")

    report_mean_results(folds_frames_clean_eval_mean_partial_rms_means,
                                    folds_frames_clean_eval_mean_partial_rms_stds,
                                    folds_frames_adv_eval_mean_partial_rms_means,
                                    folds_frames_adv_eval_mean_partial_rms_stds,
                                    folds_frames_delta_eval_mean_partial_rms_means,
                                    folds_frames_delta_eval_mean_partial_rms_stds,
                                    folds_frames_ratio_eval_mean_partial_rms_means,
                                    folds_frames_ratio_eval_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_eval_mean_partial_rms_means,
                                    folds_frames_delta_ratio_eval_mean_partial_rms_stds,
                                    crit_str="eval_mean_partial_rms")

    report_mean_results(folds_frames_clean_eval_rms_means,
                                    folds_frames_clean_eval_rms_stds,
                                    folds_frames_adv_eval_rms_means,
                                    folds_frames_adv_eval_rms_stds,
                                    folds_frames_delta_eval_rms_means,
                                    folds_frames_delta_eval_rms_stds,
                                    folds_frames_ratio_eval_rms_means,
                                    folds_frames_ratio_eval_rms_stds,
                                    folds_frames_delta_ratio_eval_rms_means,
                                    folds_frames_delta_ratio_eval_rms_stds,
                                    crit_str="eval_rms")

    report_mean_results(folds_frames_clean_test_target_mean_partial_rms_means,
                                    folds_frames_clean_test_target_mean_partial_rms_stds,
                                    folds_frames_adv_test_target_mean_partial_rms_means,
                                    folds_frames_adv_test_target_mean_partial_rms_stds,
                                    folds_frames_delta_test_target_mean_partial_rms_means,
                                    folds_frames_delta_test_target_mean_partial_rms_stds,
                                    folds_frames_ratio_test_target_mean_partial_rms_means,
                                    folds_frames_ratio_test_target_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_test_target_mean_partial_rms_means,
                                    folds_frames_delta_ratio_test_target_mean_partial_rms_stds,
                                    crit_str="test_target_mean_partial_rms")

    report_mean_results(folds_frames_clean_test_target_rms_means,
                                    folds_frames_clean_test_target_rms_stds,
                                    folds_frames_adv_test_target_rms_means,
                                    folds_frames_adv_test_target_rms_stds,
                                    folds_frames_delta_test_target_rms_means,
                                    folds_frames_delta_test_target_rms_stds,
                                    folds_frames_ratio_test_target_rms_means,
                                    folds_frames_ratio_test_target_rms_stds,
                                    folds_frames_delta_ratio_test_target_rms_means,
                                    folds_frames_delta_ratio_test_target_rms_stds,
                                    crit_str="test_target_rms")

    report_mean_results(folds_frames_clean_test_mean_partial_rms_means,
                                    folds_frames_clean_test_mean_partial_rms_stds,
                                    folds_frames_adv_test_mean_partial_rms_means,
                                    folds_frames_adv_test_mean_partial_rms_stds,
                                    folds_frames_delta_test_mean_partial_rms_means,
                                    folds_frames_delta_test_mean_partial_rms_stds,
                                    folds_frames_ratio_test_mean_partial_rms_means,
                                    folds_frames_ratio_test_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_test_mean_partial_rms_means,
                                    folds_frames_delta_ratio_test_mean_partial_rms_stds,
                                    crit_str="test_mean_partial_rms")

    report_mean_results(folds_frames_clean_test_rms_means,
                                    folds_frames_clean_test_rms_stds,
                                    folds_frames_adv_test_rms_means,
                                    folds_frames_adv_test_rms_stds,
                                    folds_frames_delta_test_rms_means,
                                    folds_frames_delta_test_rms_stds,
                                    folds_frames_ratio_test_rms_means,
                                    folds_frames_ratio_test_rms_stds,
                                    folds_frames_delta_ratio_test_rms_means,
                                    folds_frames_delta_ratio_test_rms_stds,
                                    crit_str="test_rms")


def run_attacks_open_loop_opt(args):
    print("Optimizing an adversarial perturbation in open loop format")
    print("In each iteration, the data folders are split between a train-set"
          "and a eval-set containing a single data folder")
    print("An adversarial perturbation will be trained on the train-set, "
          "and the best performing perturbation on the eval-set will be noted")
    print("The best performing perturbation over all data-sets will than be reported")
    print("A single universal perturbation will be produced and then tested on the dataset")
    print("args.attack")
    print(args.attack)
    datastr = args.datastr

    if args.custom_data:
        datastr += '_custom'
    elif args.real_data:
        datastr += '_real_data'
    attack = args.attack_obj

    dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, \
    motions_gt_list, traj_clean_criterions_list, traj_clean_motions = \
        test_clean_multi_inputs(args)

    motions_target_list = motions_gt_list
    if args.attack_target_clean:
        print("clean VO outputs will be used instead of ground truth, and clean deviation will be reconfigured to 0")
        motions_target_list = traj_clean_motions
        traj_clean_criterions_list = [[[0 for val in clean_crit_list] for clean_crit_list in traj_clean_crit_list]
                                       for traj_clean_crit_list in traj_clean_criterions_list]
    traj_clean_rms_list, traj_clean_mean_partial_rms_list, \
    traj_clean_target_rms_list, traj_clean_target_mean_partial_rms_list = tuple(traj_clean_criterions_list)

    folds_num = args.traj_datasets
    folds_indices_list = list(range(folds_num))
    eval_folds_num = args.open_loop_folders
    if eval_folds_num > folds_num:
        eval_folds_num = folds_num
    eval_folds_indices_list = list(range(eval_folds_num))
    print("optimizing attack with " + str(eval_folds_num) + " datasets folds as an optional evaluation set")
    print("folds_indices_list")
    print(folds_indices_list)
    print("eval_folds_indices_list")
    print(eval_folds_indices_list)

    folds_frames_clean_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_train_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_train_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_target_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_train_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_eval_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_eval_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_target_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_target_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_target_rms_stds = [[] for i in range(args.traj_len)]

    folds_frames_clean_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_clean_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_adv_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_ratio_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_target_mean_partial_rms_means = [[] for i in range(args.traj_len)]
    folds_frames_delta_ratio_eval_target_mean_partial_rms_stds = [[] for i in range(args.traj_len)]

    for eval_fold_idx in eval_folds_indices_list:
        output_dir, adv_best_pert_dir, \
        train_output_dir, train_flowdir, train_pose_dir, train_img_dir, train_adv_img_dir, train_adv_pert_dir, \
        eval_output_dir, eval_flowdir, eval_pose_dir, eval_img_dir, eval_adv_img_dir, eval_adv_pert_dir = \
            open_loop_make_folders(args.output_dir, eval_fold_idx,
                                       args.save_results, args.save_best_pert, args.save_flow, args.save_pose, args.save_imgs)
        print("adv_best_pert_dir")
        print(adv_best_pert_dir)

        eval_indices = [traj_idx for traj_idx in traj_indices if dataset_idx_list[traj_idx] == eval_fold_idx]
        eval_dataset = args.dataset_class(args.test_dir, processed_data_folder=args.processed_data_dir,
                                                    preprocessed_data=True,
                                                    transform=args.transform,
                                                    data_size=(args.image_height, args.image_width),
                                                    focalx=args.focalx,
                                                    focaly=args.focaly,
                                                    centerx=args.centerx,
                                                    centery=args.centery,
                                                    max_traj_len=args.max_traj_len,
                                                    max_dataset_traj_num=args.max_traj_num,
                                                    max_traj_datasets=args.max_traj_datasets,
                                                    folder_indices_list=[eval_fold_idx])
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.worker_num)


        train_indices = [traj_idx for traj_idx in traj_indices
                         if (traj_idx not in eval_indices)]
        train_folders_indices = [fold_idx for fold_idx in folds_indices_list
                                 if fold_idx != eval_fold_idx]
        train_dataset = args.dataset_class(args.test_dir, processed_data_folder=args.processed_data_dir,
                                                    preprocessed_data=True,
                                                    transform=args.transform,
                                                    data_size=(args.image_height, args.image_width),
                                                    focalx=args.focalx,
                                                    focaly=args.focaly,
                                                    centerx=args.centerx,
                                                    centery=args.centery,
                                                    max_traj_len=args.max_traj_len,
                                                    max_dataset_traj_num=args.max_traj_num,
                                                    max_traj_datasets=args.max_traj_datasets,
                                                    folder_indices_list=train_folders_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.worker_num)

        print("eval_fold_idx")
        print(eval_fold_idx)
        print("eval_indices")
        print(eval_indices)
        print("train_folders_indices")
        print(train_folders_indices)
        print("train_indices")
        print(train_indices)

        train_dataset_indices = masked_list(dataset_idx_list, train_indices)
        train_dataset_names = masked_list(dataset_name_list, train_indices)
        train_traj_names = masked_list(traj_name_list, train_indices)
        motions_target_train_list = masked_list(motions_target_list, train_indices)
        traj_clean_train_rms_list = masked_list(traj_clean_rms_list, train_indices)
        traj_clean_train_mean_partial_rms_list = masked_list(traj_clean_mean_partial_rms_list, train_indices)
        traj_clean_train_target_rms_list = masked_list(traj_clean_target_rms_list, train_indices)
        traj_clean_train_target_mean_partial_rms_list = masked_list(traj_clean_target_mean_partial_rms_list, train_indices)

        eval_dataset_indices = masked_list(dataset_idx_list, eval_indices)
        eval_dataset_names = masked_list(dataset_name_list, eval_indices)
        eval_traj_names = masked_list(traj_name_list, eval_indices)
        motions_target_eval_list = masked_list(motions_target_list, eval_indices)
        traj_clean_eval_rms_list = masked_list(traj_clean_rms_list, eval_indices)
        traj_clean_eval_mean_partial_rms_list = masked_list(traj_clean_mean_partial_rms_list, eval_indices)
        traj_clean_eval_target_rms_list = masked_list(traj_clean_target_rms_list, eval_indices)
        traj_clean_eval_target_mean_partial_rms_list = masked_list(traj_clean_target_mean_partial_rms_list, eval_indices)

        print("optimizing attacks on trajectories:")
        print(train_traj_names)
        print("evaluate attacks on trajectories:")
        print(eval_traj_names)

        best_pert, eval_clean_loss_list, eval_all_loss_list, eval_all_best_loss_list = \
            attack.perturb(train_dataloader, motions_target_train_list, eps=args.eps, device=args.device,
                           eval_data_loader=eval_dataloader, eval_y_list=motions_target_eval_list)

        adv_eval_loss_list = attack.test_pert(best_pert, eval_dataloader, motions_target_eval_list, device=args.device)

        print("eval_clean_loss_list")
        print(eval_clean_loss_list)
        print("eval_all_loss_list")
        print(eval_all_loss_list)
        print("eval_all_best_loss_list")
        print(eval_all_best_loss_list)
        eval_best_loss_list = eval_all_best_loss_list[- 1]
        print("eval_best_loss_list")
        print(eval_best_loss_list)
        print("adv_eval_loss_list")
        print(adv_eval_loss_list)

        if args.save_best_pert:
            save_image(best_pert[0], adv_best_pert_dir + '/' + 'adv_best_pert.png')

        traj_adv_train_criterions_list = \
            test_adv_trajectories(train_dataloader, args.model, motions_target_train_list, attack, best_pert,
                                  args.criterions, args.window_size,
                                  args.save_imgs, args.save_flow, args.save_pose,
                                  train_adv_img_dir, train_adv_pert_dir, train_flowdir, train_pose_dir,
                                  device=args.device)
        traj_adv_train_rms_list, traj_adv_train_mean_partial_rms_list, \
        traj_adv_train_target_rms_list, traj_adv_train_target_mean_partial_rms_list = traj_adv_train_criterions_list

        traj_adv_eval_criterions_list = \
            test_adv_trajectories(eval_dataloader, args.model, motions_target_eval_list, attack, best_pert,
                                  args.criterions, args.window_size,
                                  args.save_imgs, args.save_flow, args.save_pose,
                                  eval_adv_img_dir, eval_adv_pert_dir, eval_flowdir, eval_pose_dir,
                                  device=args.device)
        traj_adv_eval_rms_list, traj_adv_eval_mean_partial_rms_list, \
        traj_adv_eval_target_rms_list, traj_adv_eval_target_mean_partial_rms_list = traj_adv_eval_criterions_list


        frames_clean_train_target_mean_partial_rms_mean, frames_clean_train_target_mean_partial_rms_std, \
        frames_adv_train_target_mean_partial_rms_mean, frames_adv_train_target_mean_partial_rms_std, \
        frames_delta_train_target_mean_partial_rms_mean, frames_delta_train_target_mean_partial_rms_std, \
        frames_ratio_train_target_mean_partial_rms_mean, frames_ratio_train_target_mean_partial_rms_std, \
        frames_delta_ratio_train_target_mean_partial_rms_mean, frames_delta_ratio_train_target_mean_partial_rms_std = \
            report_adv_deviation(train_dataset_indices, train_dataset_names, train_traj_names, train_indices,
                                 traj_clean_train_target_mean_partial_rms_list, traj_adv_train_target_mean_partial_rms_list,
                                 args.save_csv, train_output_dir, crit_str="target_mean_partial_rms",
                                 experiment_name="train")
        frames_clean_train_target_rms_mean, frames_clean_train_target_rms_std, \
        frames_adv_train_target_rms_mean, frames_adv_train_target_rms_std, \
        frames_delta_train_target_rms_mean, frames_delta_train_target_rms_std, \
        frames_ratio_train_target_rms_mean, frames_ratio_train_target_rms_std, \
        frames_delta_ratio_train_target_rms_mean, frames_delta_ratio_train_target_rms_std = \
            report_adv_deviation(train_dataset_indices, train_dataset_names, train_traj_names, train_indices,
                                 traj_clean_train_target_rms_list, traj_adv_train_target_rms_list,
                                 args.save_csv, train_output_dir, crit_str="target_rms",
                                 experiment_name="train")

        frames_clean_train_mean_partial_rms_mean, frames_clean_train_mean_partial_rms_std, \
        frames_adv_train_mean_partial_rms_mean, frames_adv_train_mean_partial_rms_std, \
        frames_delta_train_mean_partial_rms_mean, frames_delta_train_mean_partial_rms_std, \
        frames_ratio_train_mean_partial_rms_mean, frames_ratio_train_mean_partial_rms_std, \
        frames_delta_ratio_train_mean_partial_rms_mean, frames_delta_ratio_train_mean_partial_rms_std = \
            report_adv_deviation(train_dataset_indices, train_dataset_names, train_traj_names, train_indices,
                                 traj_clean_train_mean_partial_rms_list, traj_adv_train_mean_partial_rms_list,
                                 args.save_csv, train_output_dir, crit_str="mean_partial_rms",
                                 experiment_name="train")

        frames_clean_train_rms_mean, frames_clean_train_rms_std, \
        frames_adv_train_rms_mean, frames_adv_train_rms_std, \
        frames_delta_train_rms_mean, frames_delta_train_rms_std, \
        frames_ratio_train_rms_mean, frames_ratio_train_rms_std, \
        frames_delta_ratio_train_rms_mean, frames_delta_ratio_train_rms_std = \
            report_adv_deviation(train_dataset_indices, train_dataset_names, train_traj_names, train_indices,
                                 traj_clean_train_rms_list, traj_adv_train_rms_list,
                                 args.save_csv, train_output_dir, crit_str="rms",
                                 experiment_name="train")

        frames_clean_eval_target_mean_partial_rms_mean, frames_clean_eval_target_mean_partial_rms_std, \
        frames_adv_eval_target_mean_partial_rms_mean, frames_adv_eval_target_mean_partial_rms_std, \
        frames_delta_eval_target_mean_partial_rms_mean, frames_delta_eval_target_mean_partial_rms_std, \
        frames_ratio_eval_target_mean_partial_rms_mean, frames_ratio_eval_target_mean_partial_rms_std, \
        frames_delta_ratio_eval_target_mean_partial_rms_mean, frames_delta_ratio_eval_target_mean_partial_rms_std = \
            report_adv_deviation(eval_dataset_indices, eval_dataset_names, eval_traj_names, eval_indices,
                                 traj_clean_eval_target_mean_partial_rms_list, traj_adv_eval_target_mean_partial_rms_list,
                                 args.save_csv, eval_output_dir, crit_str="target_mean_partial_rms",
                                 experiment_name="eval")
        frames_clean_eval_target_rms_mean, frames_clean_eval_target_rms_std, \
        frames_adv_eval_target_rms_mean, frames_adv_eval_target_rms_std, \
        frames_delta_eval_target_rms_mean, frames_delta_eval_target_rms_std, \
        frames_ratio_eval_target_rms_mean, frames_ratio_eval_target_rms_std, \
        frames_delta_ratio_eval_target_rms_mean, frames_delta_ratio_eval_target_rms_std = \
            report_adv_deviation(eval_dataset_indices, eval_dataset_names, eval_traj_names, eval_indices,
                                 traj_clean_eval_target_rms_list, traj_adv_eval_target_rms_list,
                                 args.save_csv, eval_output_dir, crit_str="target_rms",
                                 experiment_name="eval")

        frames_clean_eval_mean_partial_rms_mean, frames_clean_eval_mean_partial_rms_std, \
        frames_adv_eval_mean_partial_rms_mean, frames_adv_eval_mean_partial_rms_std, \
        frames_delta_eval_mean_partial_rms_mean, frames_delta_eval_mean_partial_rms_std, \
        frames_ratio_eval_mean_partial_rms_mean, frames_ratio_eval_mean_partial_rms_std, \
        frames_delta_ratio_eval_mean_partial_rms_mean, frames_delta_ratio_eval_mean_partial_rms_std = \
            report_adv_deviation(eval_dataset_indices, eval_dataset_names, eval_traj_names, eval_indices,
                                 traj_clean_eval_mean_partial_rms_list, traj_adv_eval_mean_partial_rms_list,
                                 args.save_csv, eval_output_dir, crit_str="mean_partial_rms",
                                 experiment_name="eval")

        frames_clean_eval_rms_mean, frames_clean_eval_rms_std, \
        frames_adv_eval_rms_mean, frames_adv_eval_rms_std, \
        frames_delta_eval_rms_mean, frames_delta_eval_rms_std, \
        frames_ratio_eval_rms_mean, frames_ratio_eval_rms_std, \
        frames_delta_ratio_eval_rms_mean, frames_delta_ratio_eval_rms_std = \
            report_adv_deviation(eval_dataset_indices, eval_dataset_names, eval_traj_names, eval_indices,
                                 traj_clean_eval_rms_list, traj_adv_eval_rms_list,
                                 args.save_csv, eval_output_dir, crit_str="rms",
                                 experiment_name="eval")


        del train_dataset_indices
        del train_dataset_names
        del train_traj_names
        del motions_target_train_list
        del traj_clean_train_rms_list
        del traj_clean_train_mean_partial_rms_list
        del traj_clean_train_target_rms_list
        del traj_clean_train_target_mean_partial_rms_list

        del eval_dataset_indices
        del eval_dataset_names
        del eval_traj_names
        del motions_target_eval_list
        del traj_clean_eval_rms_list
        del traj_clean_eval_mean_partial_rms_list
        del traj_clean_eval_target_rms_list
        del traj_clean_eval_target_mean_partial_rms_list

        del best_pert
        del eval_clean_loss_list
        del eval_all_loss_list
        del eval_all_best_loss_list

        del adv_eval_loss_list
        torch.cuda.empty_cache()
        gc.collect()

        collect_and_report_folds_mean_result(folds_frames_clean_train_target_mean_partial_rms_means,
                                        folds_frames_clean_train_target_mean_partial_rms_stds,
                                        folds_frames_adv_train_target_mean_partial_rms_means,
                                        folds_frames_adv_train_target_mean_partial_rms_stds,
                                        folds_frames_delta_train_target_mean_partial_rms_means,
                                        folds_frames_delta_train_target_mean_partial_rms_stds,
                                        folds_frames_ratio_train_target_mean_partial_rms_means,
                                        folds_frames_ratio_train_target_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_train_target_mean_partial_rms_means,
                                        folds_frames_delta_ratio_train_target_mean_partial_rms_stds,
                                        frames_clean_train_target_mean_partial_rms_mean,
                                        frames_clean_train_target_mean_partial_rms_std,
                                        frames_adv_train_target_mean_partial_rms_mean,
                                        frames_adv_train_target_mean_partial_rms_std,
                                        frames_delta_train_target_mean_partial_rms_mean,
                                        frames_delta_train_target_mean_partial_rms_std,
                                        frames_ratio_train_target_mean_partial_rms_mean,
                                        frames_ratio_train_target_mean_partial_rms_std,
                                        frames_delta_ratio_train_target_mean_partial_rms_mean,
                                        frames_delta_ratio_train_target_mean_partial_rms_std,
                                        crit_str="train_target_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_train_target_rms_means,
                                        folds_frames_clean_train_target_rms_stds,
                                        folds_frames_adv_train_target_rms_means,
                                        folds_frames_adv_train_target_rms_stds,
                                        folds_frames_delta_train_target_rms_means,
                                        folds_frames_delta_train_target_rms_stds,
                                        folds_frames_ratio_train_target_rms_means,
                                        folds_frames_ratio_train_target_rms_stds,
                                        folds_frames_delta_ratio_train_target_rms_means,
                                        folds_frames_delta_ratio_train_target_rms_stds,
                                        frames_clean_train_target_rms_mean,
                                        frames_clean_train_target_rms_std,
                                        frames_adv_train_target_rms_mean,
                                        frames_adv_train_target_rms_std,
                                        frames_delta_train_target_rms_mean,
                                        frames_delta_train_target_rms_std,
                                        frames_ratio_train_target_rms_mean,
                                        frames_ratio_train_target_rms_std,
                                        frames_delta_ratio_train_target_rms_mean,
                                        frames_delta_ratio_train_target_rms_std,
                                        crit_str="train_target_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_train_mean_partial_rms_means,
                                        folds_frames_clean_train_mean_partial_rms_stds,
                                        folds_frames_adv_train_mean_partial_rms_means,
                                        folds_frames_adv_train_mean_partial_rms_stds,
                                        folds_frames_delta_train_mean_partial_rms_means,
                                        folds_frames_delta_train_mean_partial_rms_stds,
                                        folds_frames_ratio_train_mean_partial_rms_means,
                                        folds_frames_ratio_train_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_train_mean_partial_rms_means,
                                        folds_frames_delta_ratio_train_mean_partial_rms_stds,
                                        frames_clean_train_mean_partial_rms_mean,
                                        frames_clean_train_mean_partial_rms_std,
                                        frames_adv_train_mean_partial_rms_mean,
                                        frames_adv_train_mean_partial_rms_std,
                                        frames_delta_train_mean_partial_rms_mean,
                                        frames_delta_train_mean_partial_rms_std,
                                        frames_ratio_train_mean_partial_rms_mean,
                                        frames_ratio_train_mean_partial_rms_std,
                                        frames_delta_ratio_train_mean_partial_rms_mean,
                                        frames_delta_ratio_train_mean_partial_rms_std,
                                        crit_str="train_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_train_rms_means,
                                        folds_frames_clean_train_rms_stds,
                                        folds_frames_adv_train_rms_means,
                                        folds_frames_adv_train_rms_stds,
                                        folds_frames_delta_train_rms_means,
                                        folds_frames_delta_train_rms_stds,
                                        folds_frames_ratio_train_rms_means,
                                        folds_frames_ratio_train_rms_stds,
                                        folds_frames_delta_ratio_train_rms_means,
                                        folds_frames_delta_ratio_train_rms_stds,
                                        frames_clean_train_rms_mean,
                                        frames_clean_train_rms_std,
                                        frames_adv_train_rms_mean,
                                        frames_adv_train_rms_std,
                                        frames_delta_train_rms_mean,
                                        frames_delta_train_rms_std,
                                        frames_ratio_train_rms_mean,
                                        frames_ratio_train_rms_std,
                                        frames_delta_ratio_train_rms_mean,
                                        frames_delta_ratio_train_rms_std,
                                        crit_str="train_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_eval_target_mean_partial_rms_means,
                                        folds_frames_clean_eval_target_mean_partial_rms_stds,
                                        folds_frames_adv_eval_target_mean_partial_rms_means,
                                        folds_frames_adv_eval_target_mean_partial_rms_stds,
                                        folds_frames_delta_eval_target_mean_partial_rms_means,
                                        folds_frames_delta_eval_target_mean_partial_rms_stds,
                                        folds_frames_ratio_eval_target_mean_partial_rms_means,
                                        folds_frames_ratio_eval_target_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_eval_target_mean_partial_rms_means,
                                        folds_frames_delta_ratio_eval_target_mean_partial_rms_stds,
                                        frames_clean_eval_target_mean_partial_rms_mean,
                                        frames_clean_eval_target_mean_partial_rms_std,
                                        frames_adv_eval_target_mean_partial_rms_mean,
                                        frames_adv_eval_target_mean_partial_rms_std,
                                        frames_delta_eval_target_mean_partial_rms_mean,
                                        frames_delta_eval_target_mean_partial_rms_std,
                                        frames_ratio_eval_target_mean_partial_rms_mean,
                                        frames_ratio_eval_target_mean_partial_rms_std,
                                        frames_delta_ratio_eval_target_mean_partial_rms_mean,
                                        frames_delta_ratio_eval_target_mean_partial_rms_std,
                                        crit_str="eval_target_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_eval_target_rms_means,
                                folds_frames_clean_eval_target_rms_stds,
                                folds_frames_adv_eval_target_rms_means,
                                folds_frames_adv_eval_target_rms_stds,
                                folds_frames_delta_eval_target_rms_means,
                                folds_frames_delta_eval_target_rms_stds,
                                folds_frames_ratio_eval_target_rms_means,
                                folds_frames_ratio_eval_target_rms_stds,
                                folds_frames_delta_ratio_eval_target_rms_means,
                                folds_frames_delta_ratio_eval_target_rms_stds,
                                frames_clean_eval_target_rms_mean,
                                frames_clean_eval_target_rms_std,
                                frames_adv_eval_target_rms_mean,
                                frames_adv_eval_target_rms_std,
                                frames_delta_eval_target_rms_mean,
                                frames_delta_eval_target_rms_std,
                                frames_ratio_eval_target_rms_mean,
                                frames_ratio_eval_target_rms_std,
                                frames_delta_ratio_eval_target_rms_mean,
                                frames_delta_ratio_eval_target_rms_std,
                                crit_str="eval_target_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_eval_mean_partial_rms_means,
                                        folds_frames_clean_eval_mean_partial_rms_stds,
                                        folds_frames_adv_eval_mean_partial_rms_means,
                                        folds_frames_adv_eval_mean_partial_rms_stds,
                                        folds_frames_delta_eval_mean_partial_rms_means,
                                        folds_frames_delta_eval_mean_partial_rms_stds,
                                        folds_frames_ratio_eval_mean_partial_rms_means,
                                        folds_frames_ratio_eval_mean_partial_rms_stds,
                                        folds_frames_delta_ratio_eval_mean_partial_rms_means,
                                        folds_frames_delta_ratio_eval_mean_partial_rms_stds,
                                        frames_clean_eval_mean_partial_rms_mean,
                                        frames_clean_eval_mean_partial_rms_std,
                                        frames_adv_eval_mean_partial_rms_mean,
                                        frames_adv_eval_mean_partial_rms_std,
                                        frames_delta_eval_mean_partial_rms_mean,
                                        frames_delta_eval_mean_partial_rms_std,
                                        frames_ratio_eval_mean_partial_rms_mean,
                                        frames_ratio_eval_mean_partial_rms_std,
                                        frames_delta_ratio_eval_mean_partial_rms_mean,
                                        frames_delta_ratio_eval_mean_partial_rms_std,
                                        crit_str="eval_mean_partial_rms")

        collect_and_report_folds_mean_result(folds_frames_clean_eval_rms_means,
                                folds_frames_clean_eval_rms_stds,
                                folds_frames_adv_eval_rms_means,
                                folds_frames_adv_eval_rms_stds,
                                folds_frames_delta_eval_rms_means,
                                folds_frames_delta_eval_rms_stds,
                                folds_frames_ratio_eval_rms_means,
                                folds_frames_ratio_eval_rms_stds,
                                folds_frames_delta_ratio_eval_rms_means,
                                folds_frames_delta_ratio_eval_rms_stds,
                                frames_clean_eval_rms_mean,
                                frames_clean_eval_rms_std,
                                frames_adv_eval_rms_mean,
                                frames_adv_eval_rms_std,
                                frames_delta_eval_rms_mean,
                                frames_delta_eval_rms_std,
                                frames_ratio_eval_rms_mean,
                                frames_ratio_eval_rms_std,
                                frames_delta_ratio_eval_rms_mean,
                                frames_delta_ratio_eval_rms_std,
                                crit_str="eval_rms")

    report_best_results(folds_frames_clean_train_target_mean_partial_rms_means,
                                    folds_frames_clean_train_target_mean_partial_rms_stds,
                                    folds_frames_adv_train_target_mean_partial_rms_means,
                                    folds_frames_adv_train_target_mean_partial_rms_stds,
                                    folds_frames_delta_train_target_mean_partial_rms_means,
                                    folds_frames_delta_train_target_mean_partial_rms_stds,
                                    folds_frames_ratio_train_target_mean_partial_rms_means,
                                    folds_frames_ratio_train_target_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_train_target_mean_partial_rms_means,
                                    folds_frames_delta_ratio_train_target_mean_partial_rms_stds,
                                    crit_str="train_target_mean_partial_rms")

    report_best_results(folds_frames_clean_train_target_rms_means,
                                    folds_frames_clean_train_target_rms_stds,
                                    folds_frames_adv_train_target_rms_means,
                                    folds_frames_adv_train_target_rms_stds,
                                    folds_frames_delta_train_target_rms_means,
                                    folds_frames_delta_train_target_rms_stds,
                                    folds_frames_ratio_train_target_rms_means,
                                    folds_frames_ratio_train_target_rms_stds,
                                    folds_frames_delta_ratio_train_target_rms_means,
                                    folds_frames_delta_ratio_train_target_rms_stds,
                                    crit_str="train_target_rms")

    report_best_results(folds_frames_clean_train_mean_partial_rms_means,
                                    folds_frames_clean_train_mean_partial_rms_stds,
                                    folds_frames_adv_train_mean_partial_rms_means,
                                    folds_frames_adv_train_mean_partial_rms_stds,
                                    folds_frames_delta_train_mean_partial_rms_means,
                                    folds_frames_delta_train_mean_partial_rms_stds,
                                    folds_frames_ratio_train_mean_partial_rms_means,
                                    folds_frames_ratio_train_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_train_mean_partial_rms_means,
                                    folds_frames_delta_ratio_train_mean_partial_rms_stds,
                                    crit_str="train_mean_partial_rms")

    report_best_results(folds_frames_clean_train_rms_means,
                                    folds_frames_clean_train_rms_stds,
                                    folds_frames_adv_train_rms_means,
                                    folds_frames_adv_train_rms_stds,
                                    folds_frames_delta_train_rms_means,
                                    folds_frames_delta_train_rms_stds,
                                    folds_frames_ratio_train_rms_means,
                                    folds_frames_ratio_train_rms_stds,
                                    folds_frames_delta_ratio_train_rms_means,
                                    folds_frames_delta_ratio_train_rms_stds,
                                    crit_str="train_rms")

    report_best_results(folds_frames_clean_eval_target_mean_partial_rms_means,
                                    folds_frames_clean_eval_target_mean_partial_rms_stds,
                                    folds_frames_adv_eval_target_mean_partial_rms_means,
                                    folds_frames_adv_eval_target_mean_partial_rms_stds,
                                    folds_frames_delta_eval_target_mean_partial_rms_means,
                                    folds_frames_delta_eval_target_mean_partial_rms_stds,
                                    folds_frames_ratio_eval_target_mean_partial_rms_means,
                                    folds_frames_ratio_eval_target_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_eval_target_mean_partial_rms_means,
                                    folds_frames_delta_ratio_eval_target_mean_partial_rms_stds,
                                    crit_str="eval_target_mean_partial_rms")

    report_best_results(folds_frames_clean_eval_target_rms_means,
                                    folds_frames_clean_eval_target_rms_stds,
                                    folds_frames_adv_eval_target_rms_means,
                                    folds_frames_adv_eval_target_rms_stds,
                                    folds_frames_delta_eval_target_rms_means,
                                    folds_frames_delta_eval_target_rms_stds,
                                    folds_frames_ratio_eval_target_rms_means,
                                    folds_frames_ratio_eval_target_rms_stds,
                                    folds_frames_delta_ratio_eval_target_rms_means,
                                    folds_frames_delta_ratio_eval_target_rms_stds,
                                    crit_str="eval_target_rms")

    report_best_results(folds_frames_clean_eval_mean_partial_rms_means,
                                    folds_frames_clean_eval_mean_partial_rms_stds,
                                    folds_frames_adv_eval_mean_partial_rms_means,
                                    folds_frames_adv_eval_mean_partial_rms_stds,
                                    folds_frames_delta_eval_mean_partial_rms_means,
                                    folds_frames_delta_eval_mean_partial_rms_stds,
                                    folds_frames_ratio_eval_mean_partial_rms_means,
                                    folds_frames_ratio_eval_mean_partial_rms_stds,
                                    folds_frames_delta_ratio_eval_mean_partial_rms_means,
                                    folds_frames_delta_ratio_eval_mean_partial_rms_stds,
                                    crit_str="eval_mean_partial_rms")

    report_best_results(folds_frames_clean_eval_rms_means,
                                    folds_frames_clean_eval_rms_stds,
                                    folds_frames_adv_eval_rms_means,
                                    folds_frames_adv_eval_rms_stds,
                                    folds_frames_delta_eval_rms_means,
                                    folds_frames_delta_eval_rms_stds,
                                    folds_frames_ratio_eval_rms_means,
                                    folds_frames_ratio_eval_rms_stds,
                                    folds_frames_delta_ratio_eval_rms_means,
                                    folds_frames_delta_ratio_eval_rms_stds,
                                    crit_str="eval_rms")


def main():
    args = get_args()
    if args.attack is None:
        return test_clean(args)
    if args.attack_open_loop:
        return run_attacks_open_loop_opt(args)
    if args.attack_oos:
        return run_attacks_out_of_sample_test(args)
    if (args.attack_name == 'random' or args.attack_name == 'random_perm') and args.test_random:
        return run_attacks_train_test_random(args)
    if args.attack_multi_perturb:
        return run_attacks_train_multi_perturb(args)
    return run_attacks_train(args)


if __name__ == '__main__':
    main()
