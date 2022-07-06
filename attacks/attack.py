import torch
from torch.nn import functional as F
import kornia.geometry as kgm
import kornia.filters as kf
from Datasets.tartanTrajFlowDataset import extract_traj_data
from loss import test_model
import numpy as np


class Attack:
    def __init__(self, model, criterion, test_criterion, norm, data_shape, stochastic=False,
                 sample_window_size=None, sample_window_stride=None, frames_exp_factor=0,
                 pert_padding=(0, 0)):
        self.model = model
        self.criterion = criterion
        self.test_criterion = test_criterion
        self.norm = norm
        self.p = float(self.norm[1:])
        self.data_len = data_shape[0]
        self.data_size = (data_shape[1], data_shape[2])
        self.stochastic = stochastic

        self.sample_window_size = sample_window_size
        self.sample_window_stride = sample_window_stride
        if sample_window_stride is None:
            self.sample_window_stride = sample_window_size

        self.calc_sample_grad_aux = self.calc_sample_grad_single
        self.perturb_model = self.perturb_model_single
        if sample_window_size is not None:
            self.calc_sample_grad_aux = self.calc_sample_grad_split
            self.perturb_model = self.perturb_model_split

        frames_exp_factors = torch.tensor([frames_exp_factor * data_idx / self.data_len
                                                for data_idx in range(self.data_len + 1)],
                                           dtype=torch.float32)
        self.frame_weights = torch.exp(frames_exp_factors).view(-1, 1, 1, 1)
        self.pert_padding = pert_padding

    def random_initialization(self, pert, eps):
        if self.norm == 'Linf':
            return torch.empty_like(pert).uniform_(1 - eps, eps)
        else:
            return torch.empty_like(pert).normal_(0, eps * eps).clamp_(0, 1)

    def normalize_grad(self, grad):
        if self.norm == 'Linf':
            return grad.sign()
        else:
            return F.normalize(grad.view(grad.shape[0], -1), p=self.p, dim=-1).view(grad.shape)

    def project(self, pert, eps):
        if self.norm == 'Linf':
            pert = torch.clamp(pert, 1-eps, eps)
        else:
            pert = F.normalize(pert.view(pert.shape[0], -1),
                               p=self.p, dim=-1).view(pert.shape) * eps
        return pert

    def warp_pert(self, pert, perspective1, perspective2, device=None):
        if self.pert_padding[0] > 0 or self.pert_padding[1] > 0:
            pert = F.pad(input=pert, pad=(self.pert_padding[1], self.pert_padding[1],
                                          self.pert_padding[0], self.pert_padding[0],
                                          0, 0,
                                          0, 0),
                         mode='constant', value=0)
        pert_blur = kf.box_blur(pert, (9, 9))
        if device is not None:
            pert_warp1 = kgm.warp_perspective(pert_blur, perspective1.to(device), dsize=self.data_size)
            pert_warp2 = kgm.warp_perspective(pert_blur, perspective2.to(device), dsize=self.data_size)
            return pert_warp1, pert_warp2
        pert_warp1 = kgm.warp_perspective(pert_blur, perspective1, dsize=self.data_size)
        pert_warp2 = kgm.warp_perspective(pert_blur, perspective2, dsize=self.data_size)
        return pert_warp1, pert_warp2

    def prep_data(self, mask, perspective):
        mask1 = mask[0:-1]
        mask2 = mask[1:]
        perspective1 = perspective[0:-1]
        perspective2 = perspective[1:]
        return mask1, mask2, perspective1, perspective2

    def apply_pert(self, pert, img1_I0, img2_I0, img1_delta, img2_delta, mask, perspective, device=None):
        with torch.no_grad():
            pert_expand = pert.expand(self.data_len, -1, -1, -1)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
            pert_warp1, pert_warp2 = self.warp_pert(pert_expand, perspective1, perspective2, device)
            if device is None:
                img1_adv = img1_I0.clone().detach()
                img2_adv = img2_I0.clone().detach()
                img1_adv[mask1] += img1_delta[mask1] * pert_warp1[mask1]
                img2_adv[mask2] += img2_delta[mask2] * pert_warp2[mask2]
            else:
                img1_adv = img1_I0.clone().detach().to(device)
                img2_adv = img2_I0.clone().detach().to(device)
                img1_adv[mask1] += img1_delta[mask1].to(device) * pert_warp1[mask1]
                img2_adv[mask2] += img2_delta[mask2].to(device) * pert_warp2[mask2]
            del pert_warp1
            del pert_warp2
            torch.cuda.empty_cache()
            return img1_adv, img2_adv

    def test_pert(self, pert, eval_data_loader, eval_y_list, device=None):
        with torch.no_grad():
            pert_expand = pert.expand(self.data_len, -1, -1, -1)
            loss_list = []
            for data_idx, data in enumerate(eval_data_loader):
                dataset_idx, dataset_name, traj_name, traj_len, \
                img1_I0, img2_I0, intrinsic_I0, \
                img1_I1, img2_I1, intrinsic_I1, \
                img1_delta, img2_delta, \
                motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
                img1_adv, img2_adv, output_adv, loss = self.test_pert_sample(pert_expand, img1_I0, img2_I0, intrinsic_I0,
                                                           img1_delta, img2_delta,
                                                           scale, mask, perspective,
                                                           eval_y_list[data_idx],
                                                           patch_pose,
                                                           device=device)
                loss_list.append(loss)

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
                torch.cuda.empty_cache()

        return loss_list

    def test_pert_sample(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                   scale, mask, perspective, y, target_pose, device=None):
        mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
        pert = pert.detach()
        img1_adv, img2_adv, output_adv_device = self.perturb_model(pert, img1_I0, img2_I0,
                                                            intrinsic_I0,
                                                            img1_delta,
                                                            img2_delta,
                                                            scale,
                                                            mask1,
                                                            mask2,
                                                            perspective1,
                                                            perspective2,
                                                            device)
        loss_device = self.test_criterion(output_adv_device, scale.to(device), y.to(device), target_pose.to(device))
        loss = loss_device.detach().cpu().tolist()
        output_adv = (output_adv_device[0].detach().cpu(), output_adv_device[1].detach().cpu())
        del output_adv_device
        del loss_device
        torch.cuda.empty_cache()
        return img1_adv, img2_adv, output_adv, loss

    def test_clean_multi_input(self, eval_data_loader, eval_y_list, device):
        clean_output_list = []
        clean_loss_list = []
        for data_idx, data in enumerate(eval_data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)

            data_shape = img1_I0.shape
            dtype = img1_I0.dtype
            with torch.no_grad():
                clean_output, clean_loss = test_model(self.model, self.test_criterion,
                                                      img1_I0, img2_I0, intrinsic_I0,
                                                      scale, eval_y_list[data_idx], patch_pose,
                                                      window_size=self.sample_window_size,
                                                      device=device)
                clean_loss *= self.frame_weights.view(-1)
            clean_output_list.append(clean_output)
            clean_loss_list.append(clean_loss)

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

        return clean_output_list, clean_loss_list, data_shape, dtype

    def compute_train_eval_clean_output(self, data_loader, y_list, eval_data_loader, eval_y_list, device=None):
        print("computing clean output and clean loss over training samples")
        clean_output_list, clean_loss_list, data_shape, dtype = self.test_clean_multi_input(data_loader, y_list, device)
        clean_flow_list = [clean_output[1] for clean_output in clean_output_list]
        del clean_output_list
        torch.cuda.empty_cache()
        if eval_data_loader is None:
            print("training samples will be used for evaluating perturbations")
            eval_clean_loss_list = clean_loss_list
            eval_data_loader = data_loader
            eval_y_list = y_list

        else:
            print("computing clean output and clean loss over evaluation samples")
            eval_clean_output_list, eval_clean_loss_list, _, _ = \
                self.test_clean_multi_input(eval_data_loader, eval_y_list, device)
            del eval_clean_output_list
            del clean_loss_list
            torch.cuda.empty_cache()
        return clean_flow_list, eval_clean_loss_list, eval_data_loader, eval_y_list, data_shape, dtype

    def compute_clean_baseline(self, data_loader, y_list, eval_data_loader, eval_y_list, device=None):

        clean_flow_list, eval_clean_loss_list, eval_data_loader, eval_y_list, data_shape, dtype = \
            self.compute_train_eval_clean_output(data_loader, y_list, eval_data_loader, eval_y_list, device=device)

        best_pert = torch.zeros(1, data_shape[1], data_shape[2], data_shape[3], device=device,
                                dtype=dtype).to(device)

        self.frame_weights = self.frame_weights.to(dtype)
        best_loss_list = [loss.detach().cpu().tolist() for loss in eval_clean_loss_list]
        best_loss_sum = np.sum([loss.sum().item() for loss in eval_clean_loss_list])
        all_loss = [best_loss_list]
        all_best_loss = [best_loss_list]
        del eval_clean_loss_list
        torch.cuda.empty_cache()
        eval_clean_loss_list = best_loss_list
        traj_clean_loss_mean_list = np.mean(eval_clean_loss_list, axis=0)
        clean_loss_sum = best_loss_sum

        return data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
               eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
               best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss

    def calc_sample_grad(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        grad = self.calc_sample_grad_aux(pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device)
        with torch.no_grad():
            grad *= self.frame_weights.to(device)[1:]
        return grad

    def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        raise NotImplementedError('calc_sample_grad_single method not defined!')

    def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        raise NotImplementedError('calc_sample_grad_split method not defined!')

    def perturb_model_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta, scale, mask1, mask2,
                      perspective1, perspective2, device=None):
        pert_warp1, pert_warp2 = self.warp_pert(pert, perspective1, perspective2, device)
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
        return img1_adv, img2_adv, output_adv

    def perturb_model_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta, scale, mask1, mask2,
                      perspective1, perspective2, device=None):

        sample_data_ind = list(range(img1_I0.shape[0] + 1))
        window_start_list = sample_data_ind[0::self.sample_window_size]
        window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_size]
        if window_end_list[-1] != sample_data_ind[-1]:
            window_end_list.append(sample_data_ind[-1])
        img1_adv_window_device_list = []
        img2_adv_window_device_list = []
        motions_window_device_list = []
        flow_window_device_list = []
        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]

            pert_window = pert[window_start:window_end].clone().detach()
            img1_I0_window = img1_I0[window_start:window_end].clone().detach()
            img2_I0_window = img2_I0[window_start:window_end].clone().detach()
            intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
            img1_delta_window = img1_delta[window_start:window_end].clone().detach()
            img2_delta_window = img2_delta[window_start:window_end].clone().detach()
            scale_window = scale[window_start:window_end].clone().detach()
            perspective1_window = perspective1[window_start:window_end].clone().detach()
            perspective2_window = perspective2[window_start:window_end].clone().detach()
            mask1_window = mask1[window_start:window_end].clone().detach()
            mask2_window = mask2[window_start:window_end].clone().detach()

            img1_adv_window_device, img2_adv_window_device, (motions_adv_window_device, flow_adv_window_device)\
                = self.perturb_model_single(pert_window, img1_I0_window, img2_I0_window, intrinsic_I0_window,
                                            img1_delta_window, img2_delta_window, scale_window,
                                            mask1_window, mask2_window, perspective1_window, perspective2_window,
                                            device=device)


            img1_adv_window_device_list.append(img1_adv_window_device)
            img2_adv_window_device_list.append(img2_adv_window_device)
            motions_window_device_list.append(motions_adv_window_device)
            flow_window_device_list.append(flow_adv_window_device)

            del pert_window
            del img1_I0_window
            del img2_I0_window
            del intrinsic_I0_window
            del scale_window
            del perspective1_window
            del perspective2_window
            del mask1_window
            del mask2_window
            torch.cuda.empty_cache()

        img1_adv_device = torch.cat(img1_adv_window_device_list, dim=0)
        img2_adv_device = torch.cat(img2_adv_window_device_list, dim=0)
        motions_device = torch.cat(motions_window_device_list, dim=0)
        flow_device = torch.cat(flow_window_device_list, dim=0)

        del img1_adv_window_device_list
        del img2_adv_window_device_list
        del motions_window_device_list
        del flow_window_device_list
        torch.cuda.empty_cache()

        return img1_adv_device, img2_adv_device, (motions_device, flow_device)

    def attack_eval(self, pert, data_shape, eval_data_loader, eval_y_list, device):
        with torch.no_grad():
            loss_list = []
            loss_sum_list = []
            pert_expand = pert.expand(data_shape[0], -1, -1, -1)
            for data_idx, data in enumerate(eval_data_loader):
                dataset_idx, dataset_name, traj_name, traj_len, \
                img1_I0, img2_I0, intrinsic_I0, \
                img1_I1, img2_I1, intrinsic_I1, \
                img1_delta, img2_delta, \
                motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
                mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
                pert_expand = pert_expand.detach()
                img1_adv, img2_adv, output_adv = self.perturb_model(pert_expand, img1_I0, img2_I0,
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
                loss_sum_list.append(loss_sum.item())
                loss_list.append(loss.detach().cpu().tolist())

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
            del loss_sum_list
            torch.cuda.empty_cache()

            return loss_tot, loss_list

    def gradient_ascent_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, a_abs, eps, device=None):

        pert_expand = pert.expand(data_shape[0], -1, -1, -1).to(device)
        grad_tot = torch.zeros_like(pert, requires_grad=False)

        for data_idx, data in enumerate(data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
            grad = self.calc_sample_grad(pert_expand, img1_I0, img2_I0, intrinsic_I0,
                                         img1_delta, img2_delta,
                                         scale, y_list[data_idx], clean_flow_list[data_idx], patch_pose,
                                         perspective1, perspective2,
                                         mask1, mask2, device=device)
            grad = grad.sum(dim=0, keepdims=True).detach()

            with torch.no_grad():
                if self.stochastic:
                    grad = self.normalize_grad(grad)
                    pert += multiplier * a_abs * grad
                    pert = self.project(pert, eps)
                else:
                    grad_tot += grad

            del grad
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
            torch.cuda.empty_cache()

        if not self.stochastic:
            with torch.no_grad():
                grad = self.normalize_grad(grad_tot)
                pert += multiplier * a_abs * grad
                pert = self.project(pert, eps)

        return pert

    def perturb(self, data_loader, y_list, eps,
                                   targeted=False, device=None, eval_data_loader=None, eval_y_list=None):
        raise NotImplementedError('perturb method not defined!')
