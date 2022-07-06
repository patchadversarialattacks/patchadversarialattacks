import numpy as np
import torch
from attacks.attack import Attack
import time
from tqdm import tqdm
import cv2


class RandomPerm(Attack):
    def __init__(
            self,
            model,
            test_criterion,
            data_shape,
            norm='Linf',
            n_iter=20,
            n_restarts=1,
            frames_exp_factor=0,
            pert_padding=(0, 0),
            hist_ref_pert_path=None,
            ref_pert_transform=None,
            disable_default_pert=True):
        super(RandomPerm, self).__init__(model, criterion=None, test_criterion=test_criterion,
                                         norm=norm, data_shape=data_shape, stochastic=False,
                                         sample_window_size=None, sample_window_stride=None,
                                         frames_exp_factor=frames_exp_factor, pert_padding=pert_padding)

        self.n_iter = n_iter * n_restarts

        self.hist_ref_pert = None
        if hist_ref_pert_path is not None:
            self.hist_ref_pert = cv2.cvtColor(cv2.imread(hist_ref_pert_path), cv2.COLOR_BGR2RGB)
            if ref_pert_transform is None:
                self.hist_ref_pert = torch.tensor(self.hist_ref_pert)
            else:
                self.hist_ref_pert = ref_pert_transform({'img': self.hist_ref_pert})['img']
        self.disable_default_pert = disable_default_pert

    def perturb(self, data_loader, y_list, eps,
                                   targeted=False, device=None, eval_data_loader=None, eval_y_list=None):

        print("computing random attack with parameters:")
        print("attack iterations: " + str(self.n_iter))
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))

        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)
        if self.disable_default_pert:
            del best_pert
            torch.cuda.empty_cache()
            best_pert = None

        print("starting attack optimization")
        opt_start_time = time.time()

        for k in tqdm(range(self.n_iter)):
            print(" attack random sampling iteration: " + str(k))
            iter_start_time = time.time()

            pert = self.sample_rand_perm(device)
            pert = self.project(pert, eps)

            step_runtime = time.time() - iter_start_time
            print(" sampling finished, iteration runtime: " + str(step_runtime))

            print(" evaluating perturbation")
            eval_start_time = time.time()

            with torch.no_grad():
                eval_loss_tot, eval_loss_list = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                                 device)

                if eval_loss_tot > best_loss_sum or best_pert is None:
                    best_pert = pert.clone().detach()
                    best_loss_list = eval_loss_list
                    best_loss_sum = eval_loss_tot
                all_loss.append(eval_loss_list)
                all_best_loss.append(best_loss_list)
                del eval_loss_tot
                del eval_loss_list
                torch.cuda.empty_cache()

            eval_runtime = time.time() - eval_start_time
            print(" evaluation finished, evaluation runtime: " + str(eval_runtime))
            traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)
            print(" current trajectories best loss mean list:")
            print(" " + str(traj_best_loss_mean_list))
            print(" trajectories clean loss mean list:")
            print(" " + str(traj_clean_loss_mean_list))
            print(" current trajectories best loss sum:")
            print(" " + str(best_loss_sum))
            print(" trajectories clean loss sum:")
            print(" " + str(clean_loss_sum))
        opt_runtime = time.time() - opt_start_time
        print("optimization restart finished, optimization runtime: " + str(opt_runtime))
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss

    def sample_rand_perm(self, device):
        idx_flat = torch.randperm(self.hist_ref_pert.numel(), device=device)
        return self.hist_ref_pert.view(-1)[idx_flat].view(self.hist_ref_pert.shape).to(device=device)
