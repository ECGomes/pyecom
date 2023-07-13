# IMPORTS

import math

import numpy as np
import sklearn.metrics
from .metrics_base import MetricsBase


class MetricsED(MetricsBase):

    def __special_metrics__(self, metrics_list):
        unique_metrics = metrics_list.copy()
        if 'conf_mat' in unique_metrics:
            unique_metrics = unique_metrics[unique_metrics != 'conf_mat']

            unique_metrics = np.append(unique_metrics, 'tp')
            unique_metrics = np.append(unique_metrics, 'tn')
            unique_metrics = np.append(unique_metrics, 'fp')
            unique_metrics = np.append(unique_metrics, 'fn')

        return unique_metrics

    def cmd_conf_mat(self, state_gt, state_pred):

        tp, tn, fp, fn = 0, 0, 0, 0

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

        for i in iterator:
            if (state_gt[i] == 0) and (state_pred[i] == 0):
                tn += 1
            elif (state_gt[i] == 0) and (state_pred[i] == 1):
                fp += 1
            elif (state_gt[i] == 1) and (state_pred[i] == 0):
                fn += 1
            elif (state_gt[i] == 1) and (state_pred[i] == 1):
                tp += 1

        conf_matrix = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

        return conf_matrix

    def cmd_tp(self, state_gt, state_pred):
        return self.cmd_conf_mat(state_gt, state_pred)['TP']

    def cmd_tn(self, state_gt, state_pred):
        return self.cmd_conf_mat(state_gt, state_pred)['TN']

    def cmd_fp(self, state_gt, state_pred):
        return self.cmd_conf_mat(state_gt, state_pred)['FP']

    def cmd_fn(self, state_gt, state_pred):
        return self.cmd_conf_mat(state_gt, state_pred)['FN']

    def cmd_precision(self, state_gt, state_pred):
        """
        Calculate precision metric
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_precision = 0
        try:
            temp_precision = temp_conf_matrix['TP'] / (temp_conf_matrix['TP'] + temp_conf_matrix['FP'])
        except ZeroDivisionError:
            temp_precision = np.nan

        return temp_precision

    def cmd_recall(self, state_gt, state_pred):
        """
        Calculate recall metric
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_recall = 0
        try:
            temp_recall = temp_conf_matrix['TP'] / (temp_conf_matrix['TP'] + temp_conf_matrix['FN'])
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
            temp_fscore = (1 + beta ** 2) * (temp_precision * temp_recall) / ((beta ** 2 * temp_precision) + temp_recall)
        except ZeroDivisionError:
            temp_fscore = np.nan

        return temp_fscore

    def cmd_fpr(self, state_gt, state_pred):
        """
        Calculate the false positive rate
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_fpr = 0
        try:
            temp_fpr = temp_conf_matrix['FP'] / (temp_conf_matrix['FP'] + temp_conf_matrix['TN'])
        except ZeroDivisionError:
            temp_fpr = np.nan

        return temp_fpr

    def cmd_fd(self, state_gt, state_pred):
        '''
        Calculate the false detections
        '''

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_fd = temp_conf_matrix['FP'] + temp_conf_matrix['FN']

        return temp_fd

    def cmd_der(self, state_gt, state_pred):
        """
        Calculates the detection error rate: (FP + FN) / (TP + FN)
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_der = 0
        try:
            temp_der = (temp_conf_matrix['FP'] + temp_conf_matrix['FN']) / (temp_conf_matrix['TP'] + temp_conf_matrix['FN'])
        except ZeroDivisionError:
            temp_der = np.nan

        return temp_der

    def cmd_dea(self, state_gt, state_pred):
        """
        Calculates the detection accuracy: (TP + TN) / ((TP + FN) + FP - FN)
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_dea = 0
        try:
            temp_dea = (temp_conf_matrix['TP'] + temp_conf_matrix['TN']) / (temp_conf_matrix['TP'] + temp_conf_matrix['FP'])
        except ZeroDivisionError:
            temp_dea = np.nan

        return temp_dea

    def cmd_dia(self, state_gt, state_pred):
        """
        Calculate the disaggregation accuracy: (TP + TN) / ((TP + FN) + FP - FN - FP)
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_dia = 0
        try:
            temp_dia = (temp_conf_matrix['TP'] + temp_conf_matrix['TN']) / temp_conf_matrix['TP']
        except ZeroDivisionError:
            temp_dia = np.nan

        return temp_dia

    def cmd_oa(self, state_gt, state_pred):
        """
        Calculate the overall accuracy: (TP + TN) / (TP + FN)
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_oa = 0
        try:
            temp_oa = (temp_conf_matrix['TP'] + temp_conf_matrix['TN']) / (temp_conf_matrix['TP'] + temp_conf_matrix['FN'])
        except ZeroDivisionError:
            temp_oa = np.nan

        return temp_oa

    def cmd_tpp(self, state_gt, state_pred):
        """
        Calculates the true positive percentage: TP / (TP + FN)
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_tpp = 0
        try:
            temp_tpp = temp_conf_matrix['TP'] / (temp_conf_matrix['TP'] + temp_conf_matrix['FN'])
        except ZeroDivisionError:
            temp_tpp = np.nan

        return temp_tpp

    def cmd_fpp(self, state_gt, state_pred):
        """
        Calculates the false positive percentage: FP / (TP + FN)
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_fpp = 0
        try:
            temp_fpp = temp_conf_matrix['FP'] / (temp_conf_matrix['TP'] + temp_conf_matrix['FN'])
        except ZeroDivisionError:
            temp_fpp = np.nan

        return temp_fpp

    def cmd_hamming_loss(self, state_gt, state_pred):
        """
        Calculate the hamming loss using sklearn
        """

        temp_hamming = sklearn.metrics.hamming_loss(y_true=state_gt, y_pred=state_pred)

        return temp_hamming

    def cmd_tpr(self, state_gt, state_pred):
        """
        Calculate true positive rate (recall): TP / (TP + FN)
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        try:
            temp_tpr = temp_conf_matrix['TP'] / (temp_conf_matrix['TP'] + temp_conf_matrix['FN'])
        except ZeroDivisionError:
            temp_tpr = np.nan

        return temp_tpr

    def cmd_tnr(self, state_gt, state_pred):
        """
        Calculate true negative rate (inverse recall): TN / (TN + FP)
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_tnr = 0
        try:
            temp_tnr = temp_conf_matrix['TN'] / (temp_conf_matrix['TN'] + temp_conf_matrix['FP'])
        except ZeroDivisionError:
            temp_tnr = np.nan

        return temp_tnr

    def cmd_bm(self, state_gt, state_pred):
        """
        Calculates the informedness: TPR + TNR - 1
        """

        temp_tpr = self.cmd_tpr(state_gt, state_pred)
        temp_tnr = self.cmd_tnr(state_gt, state_pred)

        temp_bm = temp_tpr + temp_tnr - 1

        return temp_bm

    def cmd_gm(self, state_gt, state_pred):
        """
        Calculates the geometric mean: sqrt(TPR * TNR)
        """

        temp_tpr = self.cmd_tpr(state_gt, state_pred)
        temp_tnr = self.cmd_tnr(state_gt, state_pred)

        temp_gm = np.sqrt(temp_tpr * temp_tnr)

        return temp_gm

    def cmd_ppv(self, state_gt, state_pred):
        """
        Calculates positive predictive value: Precision
        """

        temp_ppv = self.cmd_precision(state_gt, state_pred)

        return temp_ppv

    def cmd_npv(self, state_gt, state_pred):
        """
        Calculates negative predictive value, inverse precision: NPV = TN / (TN + FN)
        """

        temp_conf_matrix = self.cmd_conf_mat(state_gt, state_pred)
        temp_npv = 0
        try:
            temp_npv = temp_conf_matrix['TN'] / (temp_conf_matrix['TN'] + temp_conf_matrix['FN'])
        except ZeroDivisionError:
            temp_npv = np.nan

        return temp_npv

    def cmd_mk(self, state_gt, state_pred):
        """
        Calculates the markedness: MK = PPV + NPV - 1
        """

        temp_ppv = self.cmd_ppv(state_gt, state_pred)
        temp_npv = self.cmd_npv(state_gt, state_pred)
        temp_mk = temp_ppv + temp_npv - 1

        return temp_mk

    def cmd_mcc(self, state_gt, state_pred):
        """
        Calculates the Mathews Correlation Coefficient: MCC = sqrt(BM * MK)
        """

        temp_bm = self.cmd_bm(state_gt, state_pred)
        temp_mk = self.cmd_mk(state_gt, state_pred)
        temp_mcc = np.sqrt((temp_bm * temp_mk))

        return temp_mcc

    def cmd_smcc(self, state_gt, state_pred):
        """
        Calculates the Standardized MCC: SMCC = (1 - MCC) / 2
        """

        temp_mcc = self.cmd_mcc(state_gt, state_pred)
        temp_smcc = (1 + temp_mcc) / 2

        return temp_smcc

    def cmd_accuracy(self, state_gt, state_pred):
        """
        Calculates the Accuracy: (TP + TN ) / (TP + FP + TN + FN)
        """

        temp_conf_mat = self.cmd_conf_mat(state_gt, state_pred)
        temp_acc = 0
        try:
            temp_num = (temp_conf_mat['TP'] + temp_conf_mat['TN'])
            temp_dem = (temp_conf_mat['TP'] + temp_conf_mat['FP'] + temp_conf_mat['TN'] + temp_conf_mat['FN'])
            temp_acc =  temp_num / temp_dem
        except ZeroDivisionError:
            temp_acc = np.nan

        return temp_acc

    def cmd_er(self, state_gt, state_pred):
        """
        Calculates the Error Rate: ER = 1 - Accuracy
        """

        temp_acc = self.cmd_accuracy(state_gt, state_pred)
        return 1 - temp_acc

    def cmd_dps_pr(self, state_gt, state_pred):
        """
        Calculates the Distance to Perfect Score regarding Precision and Recall
        DPS-PR = P**2 + R**2 - 2*(P + R) + 2
        """

        temp_precision = self.cmd_precision(state_gt, state_pred)
        temp_recall = self.cmd_recall(state_gt, state_pred)

        temp_dps = temp_precision**2 + temp_recall**2 - 2 * (temp_precision + temp_recall) + 2

        return temp_dps