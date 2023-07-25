# IMPORTS

import numpy as np

from .nilm_metrics import NILMMetrics


class HybridMetrics(NILMMetrics):

    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def __special_metrics__(self, metrics_list):
        unique_metrics = metrics_list.copy()
        if 'conf_mat' in unique_metrics:
            unique_metrics = unique_metrics[unique_metrics != 'conf_mat']

            unique_metrics = np.append(unique_metrics, 'atp')
            unique_metrics = np.append(unique_metrics, 'itp')
            unique_metrics = np.append(unique_metrics, 'tn')
            unique_metrics = np.append(unique_metrics, 'fp')
            unique_metrics = np.append(unique_metrics, 'fn')

        return unique_metrics

    def cmd_conf_mat(self, state_gt, state_pred):

        if isinstance(state_gt, list):
            if len(state_gt) != len(state_pred):
                print('Ground truth and predicted arrays must be of the same size')
                return
        elif state_gt.shape[0] != state_pred.shape[0]:
            print('Ground truth and predicted arrays must be of the same size')
            return

        iterator = []
        if isinstance(state_gt, list):
            iterator = np.arange(0, len(state_gt), 1)
        else:
            iterator = np.arange(0, state_gt.shape[0], 1)

        atp, itp, tn, fp, fn = 0, 0, 0, 0, 0
        for i in iterator:
            if (state_gt[i] == 0) and (state_pred[i] == 0):
                tn += 1
            elif (state_gt[i] == 0) and (state_pred[i] > 0):
                fp += 1
            elif (state_gt[i] > 0) and (state_pred[i] == 0):
                fn += 1
            elif (state_gt[i] > 0) and (state_pred[i] > 0):
                error = np.abs(state_gt[i] - state_pred[i]) / state_gt[i]
                if error >= self.threshold:
                    itp += 1
                else:
                    atp += 1

        conf_matrix = {'ATP': atp, 'ITP': itp, 'TN': tn, 'FP': fp, 'FN': fn}

        return conf_matrix

    def cmd_atp(self, state_gt, state_pred):
        return self.cmd_conf_mat(state_gt, state_pred)['ATP']

    def cmd_itp(self, state_gt, state_pred):
        return self.cmd_conf_mat(state_gt, state_pred)['ITP']

    def cmd_tn(self, state_gt, state_pred):
        return self.cmd_conf_mat(state_gt, state_pred)['TN']

    def cmd_fp(self, state_gt, state_pred):
        return self.cmd_conf_mat(state_gt, state_pred)['FP']

    def cmd_fn(self, state_gt, state_pred):
        return self.cmd_conf_mat(state_gt, state_pred)['FN']

    def cmd_precision(self, state_gt, state_pred):

        temp_conf_mat = self.cmd_conf_mat(state_gt, state_pred)
        temp_precision = 0
        try:
            temp_precision = temp_conf_mat['ATP'] / (temp_conf_mat['ATP'] + temp_conf_mat['ITP'] + temp_conf_mat['FP'])
        except ZeroDivisionError:
            temp_precision = np.nan

        return temp_precision

    def cmd_recall(self, state_gt, state_pred):

        temp_conf_mat = self.cmd_conf_mat(state_gt, state_pred)
        temp_recall = 0
        try:
            temp_recall = temp_conf_mat['ATP'] / (temp_conf_mat['ATP'] + temp_conf_mat['ITP'] + temp_conf_mat['FN'])
        except ZeroDivisionError:
            temp_recall = np.nan

        return temp_recall

    def cmd_fscore(self, state_gt, state_pred, beta=1):
        """
        Calculate f-beta score
        """

        temp_precision = self.cmd_precision(state_gt, state_pred)
        temp_recall = self.cmd_recall(state_gt, state_pred)

        temp_fscore = 0
        try:
            temp_fscore = (1 + beta ** 2) * (temp_precision * temp_recall) / (
                    (beta ** 2 * temp_precision) + temp_recall)
        except ZeroDivisionError:
            temp_fscore = np.nan

        return temp_fscore
