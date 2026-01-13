#################################################################################################################################################
##This script defines the fairness model (i.e. "F-Layer") used for Synthetic Experiments and Described in the Paper
#################################################################################################################################################

import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from configs import nn_architecture, synthetic_xp_params


class FairModelcvxpy(nn.Module):
    """
    Model trained with a declarative convex layer for fairness
    """

    def __init__(self, lb, ub, val_loader):
        super(FairModelcvxpy, self).__init__()

        #define architecture from config, and initialize
        self.ffnn = nn_architecture.get_ffnn_structure()
        self._initialize_weights()

        #set high number of max epochs as a safety net
        self.max_epochs = 15000
        self.lb = lb
        self.ub = ub
        
        #Primal-dual variables for training (when b_train < b_tau)
        self.lambda_dual_train = 0.0  #Dual variable for training
        self.eta_0_train = .1  #Initial dual step size for training.
        self.dual_update_count_train = 0  #Track training dual updates
        
        #Primal-dual variables for inference (when b_infer < b_tau)
        self.lambda_dual = 0.0  #Initialize dual variable
        self.eta_0 = .5  #Initial dual step size
        self.dual_update_count = 0  #Track number of dual updates for adaptive step size
        
        #Tracking for convergence analysis
        self.cumulative_samples = 0
        self.cumulative_weighted_violation = 0.0
        self.lambda_max = 0.0

        self.val_loader = val_loader

        self.slack = np.full(self.max_epochs, synthetic_xp_params.get_slack())
        self.slack_current = self.slack[0]
        self.slack_current = torch.tensor([self.slack_current], dtype=torch.float32)

    def _initialize_weights(self):
        """
        Kaiming initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def get_adaptive_eta(self):
        """
        Compute adaptive step size using 1/sqrt(t) schedule for inference
        """
        return self.eta_0 / np.sqrt(max(1, self.dual_update_count))
    
    def get_adaptive_eta_train(self):
        """
        Compute adaptive step size using 1/sqrt(t) schedule for training
        """
        return self.eta_0_train / np.sqrt(max(1, self.dual_update_count_train))

    def forward(self, x, inference=False):
        """
        Define forward pass for fairness model

        @param x: inputs to get predictions for
        @param inference (bool): if true, forward pass is for inference. if not, forward pass is for training

        @returns predictions for input x
        """

        y_hat = self.ffnn(x)

        #get predictions belonging to each group in the minibatch
        indicator_0 = x[:, 0] == 0
        raw_0 = y_hat[indicator_0]
        raw_1 = y_hat[~indicator_0]
        raw_0 = raw_0.squeeze(1)
        raw_1 = raw_1.squeeze(1)

        #re-declare the layer each forward pass if slack schedule is not constant
        if isinstance(self.slack_current, int) or isinstance(self.slack_current, float):
            self.slack_current = torch.tensor([self.slack_current], dtype=torch.float32)

        #INFERENCE: Use primal-dual algorithm when b_infer < b_tau
        if inference:
            b_infer = len(x)
            b_tau = synthetic_xp_params.b_tau()
            
            if b_infer < b_tau:
                #Use adaptive step size
                eta_current = self.get_adaptive_eta()
                
                #Use primal-dual formulation
                self.cvxpylayer = self.create_primal_dual_layer(
                    len(raw_0), len(raw_1), self.lb, self.ub, self.lambda_dual, b_infer
                )
                
                epsilon = synthetic_xp_params.get_slack()
                ytilde_0, ytilde_1 = self.cvxpylayer(
                    raw_0, raw_1, torch.tensor([self.lambda_dual], dtype=torch.float32)
                )
                
                #Merge predictions
                ytilde = torch.empty_like(y_hat)
                ytilde[indicator_0] = ytilde_0.unsqueeze(1)
                ytilde[~indicator_0] = ytilde_1.unsqueeze(1)
                
                #Compute fairness violation for dual update
                with torch.no_grad():
                    mean_0 = ytilde_0.mean()
                    mean_1 = ytilde_1.mean()
                    fairness_gap = torch.abs(mean_0 - mean_1).item()
                    violation = b_infer * (fairness_gap - epsilon)
                    
                    #Dual update with adaptive step size
                    self.lambda_dual = max(0.0, self.lambda_dual + eta_current * violation)
                    self.dual_update_count += 1
                    
                    #Track statistics
                    self.cumulative_samples += b_infer
                    self.cumulative_weighted_violation += violation
                    self.lambda_max = max(self.lambda_max, self.lambda_dual)
                
                return ytilde
            else:
                #Regular inference when b_infer >= b_tau
                self.cvxpylayer = self.create_fair_layer(
                    self.slack_current, len(raw_0), len(raw_1), self.lb, self.ub
                )
                ytilde_0, ytilde_1 = self.cvxpylayer(raw_0, raw_1, self.slack_current)
                ytilde = torch.empty_like(y_hat)
                ytilde[indicator_0] = ytilde_0.unsqueeze(1)
                ytilde[~indicator_0] = ytilde_1.unsqueeze(1)
                return ytilde

        #TRAINING: Use primal-dual algorithm when b_train < b_tau
        b_train = len(x)
        b_tau = synthetic_xp_params.b_tau()
        
        if b_train < b_tau:
            #Use primal-dual formulation for training
            eta_current_train = self.get_adaptive_eta_train()
            
            #Use primal-dual formulation
            self.cvxpylayer = self.create_primal_dual_layer(
                len(raw_0), len(raw_1), self.lb, self.ub, self.lambda_dual_train, b_train
            )
            
            epsilon = synthetic_xp_params.get_slack()
            ytilde_0, ytilde_1 = self.cvxpylayer(
                raw_0, raw_1, torch.tensor([self.lambda_dual_train], dtype=torch.float32)
            )
            
            #Merge predictions
            ytilde = torch.empty_like(y_hat)
            ytilde[indicator_0] = ytilde_0.unsqueeze(1)
            ytilde[~indicator_0] = ytilde_1.unsqueeze(1)
            
            #Compute fairness violation for dual update (no grad since this is for dual variable)
            with torch.no_grad():
                mean_0 = ytilde_0.mean()
                mean_1 = ytilde_1.mean()
                fairness_gap = torch.abs(mean_0 - mean_1).item()
                violation = b_train * (fairness_gap - epsilon)
                
                #Dual update with adaptive step size
                self.lambda_dual_train = max(0.0, self.lambda_dual_train + eta_current_train * violation)
                self.dual_update_count_train += 1
            
            return ytilde
        else:
            #Regular training when b_train >= b_tau
            self.cvxpylayer = self.create_fair_layer(
                self.slack_current, len(raw_0), len(raw_1), self.lb, self.ub
            )
            ytilde_0, ytilde_1 = self.cvxpylayer(raw_0, raw_1, self.slack_current)
            ytilde = torch.empty_like(y_hat)
            ytilde[indicator_0] = ytilde_0.unsqueeze(1)
            ytilde[~indicator_0] = ytilde_1.unsqueeze(1)
            return ytilde

    def create_primal_dual_layer(self, raw0len, raw1len, lb, ub, lambda_dual, batch_size):
        """
        Creates the primal-dual fairness layer for training/inference when batch_size < b_tau.
        
        Solves: min_y ||y - raw||^2 + lambda * batch_size * |mean(y_0) - mean(y_1)|
        subject to: lb <= y <= ub
        
        Note: We drop the constant term (lambda * batch_size * epsilon) from the objective
        since it doesn't affect the optimal solution.
        
        @param raw0len: number of group 0 predictions
        @param raw1len: number of group 1 predictions
        @param lb: lower bound constraint
        @param ub: upper bound constraint
        @param lambda_dual: current dual variable value
        @param batch_size: training/inference batch size
        
        @return cvxpylayer: CvxpyLayer instance
        """
        
        #Decision variables
        yhat_0 = cp.Variable(raw0len)
        yhat_1 = cp.Variable(raw1len)
        
        #Parameters
        raw_0 = cp.Parameter(raw0len)
        raw_1 = cp.Parameter(raw1len)
        lambda_param = cp.Parameter(1, nonneg=True)
        
        #Auxiliary variable for absolute value |mean(y_0) - mean(y_1)|
        fairness_gap = cp.Variable(1, nonneg=True)
        
        #Box constraints
        constraints = [
            yhat_0 >= lb,
            yhat_1 >= lb,
            yhat_0 <= ub,
            yhat_1 <= ub
        ]
        
        #Fairness gap constraints: fairness_gap >= |mean(y_0) - mean(y_1)|
        mean_diff = cp.sum(yhat_0) / raw0len - cp.sum(yhat_1) / raw1len
        constraints += [
            fairness_gap >= mean_diff,
            fairness_gap >= -mean_diff
        ]
        
        #Objective: ||y - raw||^2 + lambda * batch_size * fairness_gap
        objective = cp.Minimize(
            cp.sum_squares(yhat_0 - raw_0) + cp.sum_squares(yhat_1 - raw_1)
            + lambda_param[0] * batch_size * fairness_gap
        )
        
        problem = cp.Problem(objective, constraints)
        
        cvxpylayer = CvxpyLayer(
            problem,
            parameters=[raw_0, raw_1, lambda_param],
            variables=[yhat_0, yhat_1]
        )
        
        return cvxpylayer

    def create_fair_layer(
        self, slack_current, raw0len, raw1len, lb, ub, use_past_predictions=False
    ):
        """
        Creates the fairness layer for the fair model. Uses cvxpylayers notation.
        Used when batch_size >= b_tau (standard fairness constraints).
        
        @param slack_current: fairness tolerance (epsilon)
        @param raw0len: length of raw predictions for group 0
        @param raw1len: length of raw predictions for group 1
        @param lb: lower bound for predictions
        @param ub: upper bound for predictions
        @param use_past_predictions: if True, include past predictions in fairness constraint (legacy, not used with primal-dual)

        @return cvxpylayer: a CvxpyLayer instance
        """
        
        #Standard fairness layer (no past predictions with primal-dual approach)
        yhat_0 = cp.Variable(raw0len)
        yhat_1 = cp.Variable(raw1len)
        raw_0 = cp.Parameter(raw0len)
        raw_1 = cp.Parameter(raw1len)
        slack_current = cp.Parameter(1)

        constr = [yhat_0 >= lb, yhat_1 >= lb, yhat_0 <= ub, yhat_1 <= ub]
        constr += [
            cp.sum(yhat_0) / raw0len - cp.sum(yhat_1) / raw1len <= slack_current
        ]
        constr += [
            -cp.sum(yhat_0) / raw0len + cp.sum(yhat_1) / raw1len <= slack_current
        ]
        objective = cp.Minimize(
            cp.sum_squares(yhat_0 - raw_0) + cp.sum_squares(yhat_1 - raw_1)
        )
        problem = cp.Problem(objective, constr)

        cvxpylayer = CvxpyLayer(
            problem,
            parameters=[raw_0, raw_1, slack_current],
            variables=[yhat_0, yhat_1],
        )

        return cvxpylayer
    
    def get_aggregate_fairness_violation(self, loader):
        """
        Computes the aggregate fairness violation over the entire dataset in loader

        @return: tuple (aggregate_violation, avg_fairness_gap, theoretical_bound, max_dual_seen, mean_group_0, mean_group_1)
        """
        model = self
        model.eval()
        
        #Reset dual state for fresh evaluation
        original_lambda = self.lambda_dual
        original_count = self.dual_update_count
        original_samples = self.cumulative_samples
        original_violation = self.cumulative_weighted_violation
        original_lambda_max = self.lambda_max
        
        self.lambda_dual = 0.0
        self.dual_update_count = 0
        self.cumulative_samples = 0
        self.cumulative_weighted_violation = 0.0
        self.lambda_max = 0.0
        
        epsilon = synthetic_xp_params.get_slack()
        total_samples = 0
        total_weighted_violation = 0.0
        
        #Track all predictions by group for aggregate means
        all_predictions_group_0 = []
        all_predictions_group_1 = []
        
        with torch.no_grad():
            for xb, yb in loader:
                if (xb[:, 0] == 0).sum() < 1 or (xb[:, 0] == 1).sum() < 1:
                    continue
                
                pred = model(xb, inference=True)
                
                indicator_0 = xb[:, 0] == 0
                
                #Collect predictions by group
                preds_group_0 = pred[indicator_0].squeeze(-1).cpu().tolist()
                preds_group_1 = pred[~indicator_0].squeeze(-1).cpu().tolist()
                all_predictions_group_0.extend(preds_group_0)
                all_predictions_group_1.extend(preds_group_1)
                
                #Compute batch-level fairness gap (for violation tracking)
                mean_0 = pred[indicator_0].mean()
                mean_1 = pred[~indicator_0].mean()
                fairness_gap = torch.abs(mean_0 - mean_1).item()
                
                batch_size = len(xb)
                total_samples += batch_size
                total_weighted_violation += batch_size * (fairness_gap - epsilon)
        
        #Save the lambda_max from THIS validation run
        max_dual_seen = self.lambda_max
        
        #Compute aggregate means across ALL predictions
        if len(all_predictions_group_0) > 0 and len(all_predictions_group_1) > 0:
            mean_group_0 = np.mean(all_predictions_group_0)
            mean_group_1 = np.mean(all_predictions_group_1)
            aggregate_fairness_gap = abs(mean_group_0 - mean_group_1)
        else:
            mean_group_0 = float('nan')
            mean_group_1 = float('nan')
            aggregate_fairness_gap = float('nan')
        
        avg_violation = total_weighted_violation / total_samples if total_samples > 0 else 0.0
        
        #Use the lambda_max we just computed
        theoretical_bound = epsilon + (max_dual_seen / (self.get_adaptive_eta() * total_samples)) if total_samples > 0 and self.eta_0 > 0 else float('inf')
        
        #Restore original dual state
        self.lambda_dual = original_lambda
        self.dual_update_count = original_count
        self.cumulative_samples = original_samples
        self.cumulative_weighted_violation = original_violation
        self.lambda_max = original_lambda_max
        
        model.train()
        
        return avg_violation, aggregate_fairness_gap, theoretical_bound, max_dual_seen, mean_group_0, mean_group_1

    def get_test_loss(self, model, loader):
        """
        Function to get test loss. Uses primal-dual algorithm for small batch inference.

        @param model: model to test
        @param loader: torch DataLoader that gets inference data

        @return loss: MSE loss on the data inside the loader object
        """

        criterion = nn.MSELoss()
        model.eval()
        loss = 0.0
        total_passes = 0
        
        #Reset dual variable at start of test
        self.lambda_dual = 0.0
        self.dual_update_count = 0
        self.cumulative_samples = 0
        self.cumulative_weighted_violation = 0.0
        self.lambda_max = 0.0

        with torch.no_grad():
            for xb, yb in loader:
                if (xb[:, 0] == 0).sum() < 1 or (xb[:, 0] == 1).sum() < 1:
                    continue

                total_passes += 1
                
                #Use primal-dual algorithm for small batches
                pred = model(xb, inference=True)
                loss += criterion(pred.squeeze(1), yb.float())

        loss = loss / total_passes
        return loss
    