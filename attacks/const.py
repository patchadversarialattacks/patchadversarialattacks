import numpy as np
import torch
from attacks.attack import Attack
import time
import cv2
from Datasets.tartanTrajFlowDataset import extract_traj_data


class Const(Attack):
    def __init__(
            self,
            model,
            test_criterion,
            data_shape,
            norm='Linf',
            pert_padding=(0, 0),
            pert_path=None,
            pert_transform=None,
            default_pert_I1=False):
        super(Const, self).__init__(model, criterion=None, test_criterion=test_criterion,
                                    norm=norm, data_shape=data_shape, stochastic=False,
                                    sample_window_size=None, sample_window_stride=None,
                                     frames_exp_factor=0, pert_padding=pert_padding)
        self.set_pertubation(pert_path, pert_transform, default_pert_I1)

    def set_pertubation(self, pert_path=None, pert_transform=None, default_pert_I1=False):
        self.pert = None
        self.default_pert_I1 = default_pert_I1
        if pert_path is not None:
            self.pert = cv2.cvtColor(cv2.imread(pert_path), cv2.COLOR_BGR2RGB)
            if pert_transform is None:
                self.pert = torch.tensor(self.pert)
            else:
                self.pert = pert_transform({'img': self.pert})['img']

    def perturb(self, data_loader, y_list, eps,
                                   targeted=False, device=None, eval_data_loader=None, eval_y_list=None):

        print("computing output on given pertubation, normalized according to parameters:")
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))

        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)

        print("producing pertubation")
        opt_start_time = time.time()

        if self.pert is None:
            pert = torch.zeros_like(best_pert)
            if self.default_pert_I1:
                pert = torch.ones_like(best_pert)
        else:
            pert = self.pert.to(device)
            pert = self.project(pert, eps)

        print("evaluating perturbation")
        eval_start_time = time.time()

        with torch.no_grad():
            eval_loss_tot, eval_loss_list = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list, device)

            best_pert = pert.clone().detach()
            best_loss_list = eval_loss_list
            best_loss_sum = eval_loss_tot
            all_loss.append(eval_loss_list)
            all_best_loss.append(best_loss_list)
            del eval_loss_tot
            del eval_loss_list
            torch.cuda.empty_cache()

        eval_runtime = time.time() - eval_start_time
        print("evaluation finished, evaluation runtime: " + str(eval_runtime))
        traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)
        print("trajectories best loss mean list:")
        print(str(traj_best_loss_mean_list))
        print("trajectories clean loss mean list:")
        print(str(traj_clean_loss_mean_list))
        print("trajectories best loss sum:")
        print(str(best_loss_sum))
        print("trajectories clean loss sum:")
        print(str(clean_loss_sum))
        opt_runtime = time.time() - opt_start_time
        print("optimization restart finished, optimization runtime: " + str(opt_runtime))
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss

    def perturb_model_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta, scale, mask1, mask2,
                      perspective1, perspective2, device=None):
        pert_warp1, pert_warp2 = self.warp_pert(pert, perspective1, perspective2, device)
        traj_pert_l0_ratio = [(pert_warp1[frame_idx][mask1[frame_idx]].count_nonzero() / frame.numel()).item()
                        for frame_idx, frame in enumerate(img1_I0)]
        traj_pert_l0_ratio += [(pert_warp2[-1][mask2[-1]].count_nonzero() / img2_I0[-1].numel()).item()]
        avg_traj_pert_l0_ratio = np.mean(traj_pert_l0_ratio)


        print("traj_pert_l0_ratio")
        print(traj_pert_l0_ratio)
        print("avg_traj_pert_l0_ratio")
        print(avg_traj_pert_l0_ratio)

        if device is None:
            img1_adv = img1_I0.clone().detach()
            img2_adv = img2_I0.clone().detach()
            img1_adv[mask1] += img1_delta[mask1] * pert_warp1[mask1]
            img2_adv[mask2] += img2_delta[mask2] * pert_warp2[mask2]
            output_adv = self.model(img1_adv, img2_adv, intrinsic_I0, scale)
        else:
            img1_adv = img1_I0.clone().detach().to(device)
            img2_adv = img2_I0.clone().detach().to(device)
            img1_adv[mask1] += img1_delta[mask1].to(device) * pert_warp1[mask1]
            img2_adv[mask2] += img2_delta[mask2].to(device) * pert_warp2[mask2]
            output_adv = self.model(img1_adv, img2_adv, intrinsic_I0.to(device), scale.to(device))
        del pert_warp1
        del pert_warp2
        torch.cuda.empty_cache()
        return img1_adv, img2_adv, output_adv, traj_pert_l0_ratio

    def attack_eval(self, pert, data_shape, eval_data_loader, eval_y_list, device):
        with torch.no_grad():
            loss_list = []
            loss_sum_list = []
            traj_pert_l0_ratio_list = [[] for idx in range(data_shape[0])]
            pert_expand = pert.expand(data_shape[0], -1, -1, -1)
            for data_idx, data in enumerate(eval_data_loader):
                dataset_idx, dataset_name, traj_name, traj_len, \
                img1_I0, img2_I0, intrinsic_I0, \
                img1_I1, img2_I1, intrinsic_I1, \
                img1_delta, img2_delta, \
                motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
                print("traj_name")
                print(traj_name)
                mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
                pert_expand = pert_expand.detach()
                img1_adv, img2_adv, output_adv, traj_pert_l0_ratio = self.perturb_model(pert_expand, img1_I0, img2_I0,
                                                                    intrinsic_I0,
                                                                    img1_I1, img2_I1,
                                                                    scale,
                                                                    mask1, mask2,
                                                                    perspective1,
                                                                    perspective2,
                                                                    device)
                loss = self.test_criterion(output_adv, scale.to(device),
                                           eval_y_list[data_idx].to(device), patch_pose.to(device))

                loss *= self.frame_weights.to(device).view(-1)
                loss_sum = loss.sum(dim=0)
                loss_sum_item = loss_sum.item()
                print("loss_sum_item")
                print(loss_sum_item)
                loss_sum_list.append(loss_sum_item)
                loss_list.append(loss.detach().cpu().tolist())
                [traj_pert_l0_ratio_list[idx].append(traj_pert_l0_ratio[idx]) for idx in range(data_shape[0])]

                del img1_I0
                del img2_I0
                del intrinsic_I0
                del img1_I1
                del img2_I1
                del intrinsic_I1
                del img1_delta
                del img2_delta
                del motions_gt
                del scale
                del pose_quat_gt
                del patch_pose
                del mask
                del perspective
                del img1_adv
                del img2_adv
                del output_adv
                del loss
                del loss_sum
                torch.cuda.empty_cache()

            loss_tot = np.sum(loss_sum_list)
            [traj_pert_l0_ratio_list[idx].append(traj_pert_l0_ratio[idx]) for idx in range(data_shape[0])]
            traj_pert_l0_ratio_means = [np.mean(frame_pert_l0_ratio_list) for frame_pert_l0_ratio_list in traj_pert_l0_ratio_list]
            traj_pert_l0_ratio_stds = [np.std(frame_pert_l0_ratio_list) for frame_pert_l0_ratio_list in traj_pert_l0_ratio_list]
            pert_l0_ratio_mean = np.mean(traj_pert_l0_ratio_list)
            pert_l0_ratio_std = np.std(traj_pert_l0_ratio_list)
            print("traj_pert_l0_ratio_means")
            print(traj_pert_l0_ratio_means)
            print("traj_pert_l0_ratio_stds")
            print(traj_pert_l0_ratio_stds)
            print("pert_l0_ratio_mean")
            print(pert_l0_ratio_mean)
            print("pert_l0_ratio_std")
            print(pert_l0_ratio_std)
            del loss_sum_list
            del traj_pert_l0_ratio_list
            del traj_pert_l0_ratio_means
            del traj_pert_l0_ratio_stds
            torch.cuda.empty_cache()

            return loss_tot, loss_list
