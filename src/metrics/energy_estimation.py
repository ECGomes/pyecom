# IMPORTS

import math

import numpy as np
import sklearn.metrics
from .metrics_base import BaseMetric
from .utils import aux_get_size, aux_error_checking


class EEMetric(BaseMetric):

    def __special_metrics__(self, metrics_list):
        unique_metrics = metrics_list.copy()
        if 'cep' in unique_metrics:
            unique_metrics = unique_metrics[unique_metrics != 'cep']

            unique_metrics = np.append(unique_metrics, 'cep_c')
            unique_metrics = np.append(unique_metrics, 'cep_co')
            unique_metrics = np.append(unique_metrics, 'cep_cu')
            unique_metrics = np.append(unique_metrics, 'cep_o')
            unique_metrics = np.append(unique_metrics, 'cep_ozero')
            unique_metrics = np.append(unique_metrics, 'cep_u')
            unique_metrics = np.append(unique_metrics, 'cep_total')
            unique_metrics = np.append(unique_metrics, 'cep_cp')
            unique_metrics = np.append(unique_metrics, 'cep_cop')
            unique_metrics = np.append(unique_metrics, 'cep_cup')
            unique_metrics = np.append(unique_metrics, 'cep_cx')
            unique_metrics = np.append(unique_metrics, 'cep_op')
            unique_metrics = np.append(unique_metrics, 'cep_ozp')
            unique_metrics = np.append(unique_metrics, 'cep_up')

        return unique_metrics

    # Definition of metrics
    def cmd_re(self, state_gt, state_pred):
        '''
        Calculates the relative error of the energy estimation:
        RE = (sum(energy_gt) - sum(energy_pred)) / sum(energy_gt)
        '''
        if aux_error_checking(state_gt, state_pred):
            return

        temp_numerator = (np.sum(state_gt) - np.sum(state_pred))
        temp_denominator = np.sum(state_gt)

        if temp_denominator != 0:
            return temp_numerator / temp_denominator
        else:
            return 0

    def cmd_cv_rmsd(self, state_gt, state_pred):
        """
        Calculates the 1 - Covariance of the RMSD
        inv_cv = 1 - (RMSE / mean(ground_truth))
        """

        if aux_error_checking(state_gt, state_pred):
            return

        if np.mean(state_gt) == 0:
            return 0

        temp_cv_rmsd = 1 - (np.sqrt(sklearn.metrics.mean_squared_error(state_gt, state_pred)) / np.mean(state_gt))

        return temp_cv_rmsd

    def cmd_rmse(self, state_gt, state_pred):
        """
        Calculates the root mean squared values
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_rmse = np.sqrt(sklearn.metrics.mean_squared_error(state_gt, state_pred))

        return temp_rmse

    def cmd_abse(self, state_gt, state_pred):
        """
        Calculate the Average Error of the ground truth vs predicted values
        ae = sum()
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_ae = 0
        temp_length = aux_get_size(state_gt)
        for i in np.arange(0, temp_length, 1):
            temp_ae += np.abs(state_gt[i] - state_pred[i])

        return temp_ae / temp_length

    def cmd_mae(self, state_gt, state_pred):
        """
        Calculates the Mean Absolute Error of the ground truth vs predicted values
        """

        if aux_error_checking(state_gt, state_pred):
            return

        return sklearn.metrics.mean_absolute_error(state_gt, state_pred)

    def cmd_ae(self, state_gt, state_pred):
        '''
        Calculate the Average Error of the ground truth vs predicted values
        ae = sum()
        '''

        if aux_error_checking(state_gt, state_pred):
            return

        temp_ae = 0
        temp_length = aux_get_size(state_gt)
        for i in np.arange(0, temp_length, 1):
            temp_ae += (state_pred[i] - state_gt[i])

        return temp_ae / temp_length

    def cmd_sde(self, state_gt, state_pred):
        """
        Calculate the standard deviation error
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_length = aux_get_size(state_gt)

        temp_placeholder = 0
        temp_delta_mean = self.cmd_ae(state_gt, state_pred)

        for i in np.arange(0, temp_length, 1):
            temp_placeholder += ((np.abs(state_pred[i] - state_gt[i]) - temp_delta_mean) ** 2)

        temp_sde = np.sqrt(temp_placeholder / temp_length)

        return temp_sde

    def cmd_rsquared(self, state_gt, state_pred):
        """
        Calculates the r-squared value
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_rsquared = sklearn.metrics.r2_score(state_gt, state_pred)

        return temp_rsquared

    def cmd_psde(self, state_gt, state_pred):
        """
        Percent Standard Deviation Explained: 1 - sqrt(1 - r-squared)
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_rsquared = self.cmd_rsquared(state_gt, state_pred)
        temp_psde = 1 - np.sqrt(1 - temp_rsquared)

        return temp_psde

    def cmd_ee(self, state_gt, state_pred):
        """
        Calculates the energy error of the predictions versus the ground truth values
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_length = aux_get_size(state_gt)
        temp_numerator = 0
        for i in np.arange(0, temp_length, 1):
            temp_numerator += np.abs(state_pred[i] - state_gt[i])

        temp_denominator = np.sum(state_gt)

        if temp_denominator == 0:
            return 0

        temp_ee = temp_numerator / temp_denominator

        return temp_ee

    def cmd_eav1(self, state_gt, state_pred, alpha_param=1.4):
        """
        Calculate energy accuracy
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_ee = self.cmd_ee(state_gt, state_pred)
        temp_eav1 = math.exp(-alpha_param * temp_ee)

        return temp_eav1

    def cmd_mr_micro(self, state_gt, state_pred):
        """
        Micro Match rate of energy
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_numerator = 0
        temp_denominator = 0

        temp_length = aux_get_size(state_gt)
        for i in np.arange(0, temp_length, 1):
            temp_numerator += min(state_gt[i], state_pred[i])
            temp_denominator += max(state_gt[i], state_pred[i])

        if temp_denominator == 0:
            return 0

        temp_mr = temp_numerator / temp_denominator

        return temp_mr

    def cmd_mr_feel(self, state_gt, state_pred):
        """
        Macro Match rate FEEL of energy. Outputs Macro MR, UnderMR and OverMR
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_mr = 0
        temp_mr_count = 0

        temp_undermatch_rate = 0
        temp_overmatch_rate = 0

        temp_numerator = 0
        temp_denominator = 0

        temp_length = aux_get_size(state_gt)
        for i in np.arange(0, temp_length, 1):

            if state_gt[i] == 0 and state_pred[i] > 0:
                temp_overmatch_rate += 1
            elif state_gt[i] > 0 and state_pred[i] == 0:
                temp_undermatch_rate += 1
            elif state_gt[i] == 0 and state_pred[i] == 0:
                temp_mr += 1
                temp_mr_count += 1
            else:
                temp_numerator = min(state_gt[i], state_pred[i])
                temp_denominator = max(state_gt[i], state_pred[i])

                temp_mr += (temp_numerator / temp_denominator)
                temp_mr_count += 1

        temp_undermatch_rate /= temp_length
        temp_overmatch_rate /= temp_length

        if temp_mr_count > 0:
            temp_mr = temp_mr / temp_mr_count
        else:
            temp_mr = 0

        return temp_mr, temp_undermatch_rate, temp_overmatch_rate

    def cmd_cep(self, state_gt, state_pred):
        """
        Custom metric to return the correctly explained under, over and correct estimations
        """

        if aux_error_checking(state_gt, state_pred):
            return

        total = 0

        cu = 0
        co = 0
        c = 0

        ozero = 0
        o = 0
        u = 0

        temp_length = aux_get_size(state_gt)
        for i in np.arange(0, temp_length, 1):
            total += state_gt[i]

            # Correctly assigned
            if state_gt[i] == state_pred[i]:
                c += state_gt[i]

            # Underestimation
            elif state_gt[i] > state_pred[i]:
                u += (state_gt[i] - state_pred[i])
                cu += state_pred[i]

            # Overestimation
            elif state_gt[i] < state_pred[i]:
                if state_gt[i] > 0:
                    o += (state_pred[i] - state_gt[i])
                else:
                    ozero += (state_pred[i] - state_gt[i])
                co += state_gt[i]

        cp, cop, cup, cx, op, ozp, up = 0, 0, 0, 0, 0, 0, 0

        if total > 0:
            cp = c / total
            cop = co / total
            cup = cu / total
            cx = (c + co + cu) / total
            op = o / total
            ozp = ozero / total
            up = u / total
        else:
            cp = 0
            cop = 0
            cozp = 0
            cup = 0
            if (c + co + cu) == total:
                cx = 1
            else:
                cx = 0
            op = 0
            up = 0
            ozp = ozero / state_gt.shape[0]

        return c, co, cu, o, ozero, u, total, cp, cop, cup, cx, op, ozp, up

    def cmd_cep_c(self, state_gt, state_pred):
        """
        Returns the C value of the CEP metric
        """
        return self.cmd_cep(state_gt, state_pred)[0]

    def cmd_cep_co(self, state_gt, state_pred):
        """
        Returns the CO value of the CEP metric
        """
        return self.cmd_cep(state_gt, state_pred)[1]

    def cmd_cep_cu(self, state_gt, state_pred):
        """
        Returns the CU value of the CEP metric
        """
        return self.cmd_cep(state_gt, state_pred)[2]

    def cmd_cep_o(self, state_gt, state_pred):
        """
        Returns the O value of the CEP metric
        """
        return self.cmd_cep(state_gt, state_pred)[3]

    def cmd_cep_ozero(self, state_gt, state_pred):
        """
        Returns the OZero value of the CEP metric
        """
        return self.cmd_cep(state_gt, state_pred)[4]

    def cmd_cep_u(self, state_gt, state_pred):
        """
        Returns the U value of the CEP metric
        """
        return self.cmd_cep(state_gt, state_pred)[5]

    def cmd_cep_total(self, state_gt, state_pred):
        """
        Returns the total energy for other calculations
        """
        return self.cmd_cep(state_gt, state_pred)[6]

    def cmd_cep_cp(self, state_gt, state_pred):
        """
        Returns the percentage of correctly estimated energy
        """
        return self.cmd_cep(state_gt, state_pred)[7]

    def cmd_cep_cop(self, state_gt, state_pred):
        """
        Returns the percentage of correctly estimated energy with overestimation
        """
        return self.cmd_cep(state_gt, state_pred)[8]

    def cmd_cep_cup(self, state_gt, state_pred):
        """
        Returns the percentage of correctly estimated energy with underestmation
        """
        return self.cmd_cep(state_gt, state_pred)[9]

    def cmd_cep_cx(self, state_gt, state_pred):
        """
        Returns the percentage of correctly estimated energy with other components
        """
        return self.cmd_cep(state_gt, state_pred)[10]

    def cmd_cep_op(self, state_gt, state_pred):
        """
        Returns the percentage of overtimation
        """
        return self.cmd_cep(state_gt, state_pred)[11]

    def cmd_cep_ozp(self, state_gt, state_pred):
        """
        Returns the percentage of overestimation on GT == 0
        """
        return self.cmd_cep(state_gt, state_pred)[12]

    def cmd_cep_up(self, state_gt, state_pred):
        """
        Returns the percentage of underestimation
        """
        return self.cmd_cep(state_gt, state_pred)[13]

    def cmd_MRuMR_macro(self, state_gt, state_pred):
        """
        Macro Match and Un-Match rate of energy
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_mr = 0
        temp_mr_count = 0

        temp_unmatch_rate = 0
        temp_unmatch_rate_zero_counts = 0

        temp_length = aux_get_size(state_gt)
        for i in np.arange(0, temp_length, 1):

            if state_gt[i] > 0:
                temp_numerator = min(state_gt[i], state_pred[i])
                temp_denominator = max(state_gt[i], state_pred[i])

                temp_mr += (temp_numerator / temp_denominator)
                temp_mr_count += 1
            else:
                temp_unmatch_rate_zero_counts += 1

                if state_pred[i] > 0:
                    temp_unmatch_rate += 1

        if temp_unmatch_rate_zero_counts > 0:
            temp_unmatch_rate /= temp_unmatch_rate_zero_counts
        else:
            temp_unamtch_rate = 0

        if temp_mr_count > 0:
            temp_mr = temp_mr / temp_mr_count
        else:
            temp_mr = 0

        return temp_mr, temp_unmatch_rate

    def cmd_sem(self, state_gt):
        """
        Standard error of the mean
        """

        temp_length = aux_get_size(state_gt)
        temp_std = np.std(state_gt)

        temp_sem = temp_std / np.sqrt(temp_length)

        return temp_sem

    def cmd_fee(self, state_gt, state_pred):
        """
        Calculates the fraction of energy explained
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_numerator = np.sum(state_pred)
        temp_denominator = np.sum(state_gt)

        if temp_denominator <= 0:
            return np.nan

        return temp_numerator / temp_denominator

    def cmd_teca(self, state_gt, state_pred):
        """
        Total Energy Correctly Assigned
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_placeholder = 0
        for i in np.arange(0, state_gt.shape[0], 1):

            temp_appliance = 0
            for j in np.arange(0, state_gt.shape[1], 1):
                temp_appliance += np.abs(state_pred[i, j] - state_gt[i, j])

            temp_placeholder += temp_appliance
            # temp_totalEnergy += np.sum()

        temp_numerator = 1 - temp_placeholder
        temp_denominator = 2 * (np.sum(state_gt))

        if temp_denominator == 0:
            return 0

        return temp_numerator / temp_denominator

    def cmd_etea(self, state_gt, state_pred):
        """
        Error in total energy assigned
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_sum_gt = np.sum(state_gt)
        temp_sum_pred = np.sum(state_pred)

        return np.abs(temp_sum_pred - temp_sum_gt)

    def cmd_dev(self, state_gt, state_pred):
        """
        Ratio between ETEA and the actual energy consumed
        :param state_gt: Ground truth values
        :param state_pred: Predicted values
        :return: Metric Value
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_sum_gt = np.sum(state_gt)
        temp_etea = self.cmd_etea(state_gt, state_pred)

        if temp_sum_gt == 0:
            return 0

        return temp_etea / temp_sum_gt

    def metrics_fteac(state_gt, state_pred):
        """
        Not implemented
        :param state_gt:
        :param state_pred:
        :return:
        """
        return