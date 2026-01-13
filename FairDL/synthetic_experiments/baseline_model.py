################################################################################################################################################
#This script defines the baseline (i.e. "Projection") model used for Synthetic Experiments and described in the Paper
################################################################################################################################################
import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np
from configs import nn_architecture, synthetic_xp_params

class BaselineRegressionModel(nn.Module):
    '''
    Baseline model that trains in a typical fashion (ignoring constraint set). Raw test-time predictions then
    predicted onto constraint set afterward
    '''
    def __init__(self,val_loader):
        super(BaselineRegressionModel, self).__init__()
        
        #define architecture from config, and initialize
        self.ffnn = nn_architecture.get_ffnn_structure()
        self._initialize_weights()
        self.val_loader = val_loader

    def _initialize_weights(self):
        """
        Kaiming initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                    
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                
                #initialize biases
                if m.bias is not None:
                     nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        """
        Define forward pass for baseline model
        """
        return self.ffnn(x)


    def fair_projection(self,raw_0,raw_1,lb,ub):
        """
        Convex problem that projects baseline model's raw predictions onto constraint set
        @param raw_0: raw predictions from the model for group 0 for the minibatch
        @param raw_1: raw predictions from the model for group 1 for the minibatch
        @param lb: lower bound for any given prediction
        @param ub: upper bound for any given prediction

        @return y_hat0.value: decision variable of problem (i.e. transformed predictions for group 0)
        @return y_hat1.value: decision variable of problem (i.e. transformed predictions for group 1)
        """

        #define decision variables
        yhat_0 = cp.Variable(len(raw_0))
        yhat_1 = cp.Variable(len(raw_1))

        #define constraints
        constr = [yhat_0 >= lb, yhat_1 >= lb, yhat_0 <= ub, yhat_1 <= ub]

        #ensures average prediction in minibatch differs by no more than get_slack() value. (split abs val constraint into two)
        constr += [cp.sum(yhat_0)/len(raw_0) - cp.sum(yhat_1)/len(raw_1) <= synthetic_xp_params.get_slack()]
        constr += [-cp.sum(yhat_0)/len(raw_0) + cp.sum(yhat_1)/len(raw_1) <= synthetic_xp_params.get_slack()]

        #define problem and solve
        objective = cp.Minimize(cp.sum_squares(yhat_0 - raw_0) + cp.sum_squares(yhat_1 - raw_1))
        problem = cp.Problem(objective, constr)
        problem.solve(solver=cp.SCS, verbose=False)

        return yhat_0.value, yhat_1.value

    def get_test_loss_large_batch(self, base_model, test_loader, lb, ub):
        '''
        Helper function to get the test set loss for the baseline regression model if b_inf > b_tau
        
        @param base_model: instance of class BaselineRegressionModel
        @param test_loader: Torch DataLoader for the test set
        @param lb: lower bound for any given prediction
        @param ub: upper bound for any given prediction

        @returns mseloss_orig.item(): Test set loss of raw predictions
        @return mseloss.item(): Test set loss of transformed predictions
        '''

        #initialize data containers
        base_model.eval()
        all_raw_0 = []
        all_raw_1 = []
        all_fair_0 = []
        all_fair_1 = []
        indicator_list = []
        all_targets = []

        with torch.no_grad():

            #if insufficient observations in group 0 or 1, skip the batch
            for xb, yb in test_loader: 
                if (xb[:, 0] == 0).sum() <1 or (xb[:, 0] == 1).sum() <1:
                    continue

                pred = base_model(xb).squeeze(1)

                #find obs belonging to group 0
                indicator_0_mask = (xb[:, 0] == 0)
                
                #append the info to the  data containers holding overall info
                indicator_list.append(indicator_0_mask)
                all_targets.append(yb.float())

                #get raw predictions for each group and append to the data containers holding overall info across batches
                pred_0 = pred[indicator_0_mask]
                pred_1 = pred[~indicator_0_mask]
                if pred_0.numel() > 0:
                    all_raw_0.append(pred_0)
                if pred_1.numel() > 0:
                    all_raw_1.append(pred_1)

                new_y0, new_y1 = base_model.fair_projection(pred_0, pred_1,lb,ub)

                all_fair_0.append(torch.from_numpy(new_y0))
                all_fair_1.append(torch.from_numpy(new_y1))


        #Clean up data containers that contain info across all batches (raw predictions and group assignment and true target)
        all_raw_0 = torch.cat(all_raw_0) if all_raw_0 else torch.tensor([])
        all_raw_1 = torch.cat(all_raw_1) if all_raw_1 else torch.tensor([])
        all_fair_0 = torch.cat(all_fair_0) if all_fair_0 else torch.tensor([])
        all_fair_1 = torch.cat(all_fair_1) if all_fair_1 else torch.tensor([])


        if len(all_fair_0) > 0 and len(all_fair_1) > 0:
            mean_group_0 = all_fair_0.mean().item()  # Use torch.mean()
            mean_group_1 = all_fair_1.mean().item()  # Use torch.mean()
            aggregate_fairness_gap = abs(mean_group_0 - mean_group_1)
            aggregate_fairness_satisfied = aggregate_fairness_gap <= .05
        else:
            # If we don't have both groups, fairness is undefined
            aggregate_fairness_gap = float('nan')
            aggregate_fairness_satisfied = False


        indicator_all = torch.cat(indicator_list)
        targets_all = torch.cat(all_targets)
        #get overall vectors of all raw and fair preds (ensure same type/shape as targets desipte redundancy)
        fair_preds = torch.zeros_like(targets_all, dtype=torch.float32)
        raw_preds = torch.zeros_like(targets_all, dtype=torch.float32)
        fair_preds[indicator_all] = torch.tensor(all_fair_0, dtype=torch.float32)
        fair_preds[~indicator_all] = torch.tensor(all_fair_1, dtype=torch.float32)
        raw_preds[indicator_all] = torch.tensor(all_raw_0, dtype=torch.float32)
        raw_preds[~indicator_all] = torch.tensor(all_raw_1, dtype=torch.float32)

        #get loss values
        loss_fn = torch.nn.MSELoss()
        mseloss = loss_fn(fair_preds, targets_all)
        mseloss_orig = loss_fn(raw_preds, targets_all)
            
        print(f"Original prediction test loss: {mseloss_orig.item():.4f}")
        print(f"Fair prediction test loss: {mseloss.item():.4f}")

        return mseloss_orig.item(), mseloss.item(),aggregate_fairness_satisfied, aggregate_fairness_gap
