from .utils import aux_group_interval
import pandas as pd
import numpy as np


class BaseMetric:

    def __init__(self):
        return

    def checkFunction(self, name):
        fn = getattr(self, 'cmd_' + name, None)
        if fn is not None:
            return True
        else:
            print('Undefined metric call')
            return False

    def callFunction(self, name, gt, pred):
        fn = getattr(self, 'cmd_' + name, None)
        if fn is not None:
            return fn(gt, pred)
        else:
            print('Undefined metric call')
            return

    def check_metrics(self, df_gt, df_pred, metric_list, start_date='None', end_date='None', interval='None'):
        """
        :param df_gt: Ground-truth pandas DataFrame
        :param df_pred: Predictions on a pandas DataFrame
        :param start_date: Starting date for metric calculations
        :param end_date: Ending date for metric calculations
        :param metric_list: Set of metrics to calculate
        :param interval: Interval to use while calculating metrics: 'None', 'Min', 'Hour', 'Day', 'Week', 'Year'
        :return: Dictionary containing metrics for the interval specified
        """

        # Check available columns based on the GT frame
        col_list = self.__check_columns__(df_gt, df_pred)

        # Get the data within the time frame
        temp_gt, temp_pred = self.__extract_datatime__(col_list, df_gt, df_pred, end_date, start_date)

        sampled_time = aux_group_interval(temp_gt, interval)

        # Go through a list of metrics to calculate
        unique_metrics_v1 = np.unique(np.array(metric_list))
        unique_metrics_v1 = self.__special_metrics__(unique_metrics_v1)
        unique_metrics_v2 = self.__remove_undefinedMetrics__(unique_metrics_v1)

        # Calculate the metrics
        column_results = self.__calculate_metrics__(col_list, sampled_time, temp_gt, temp_pred,
                                                    unique_metrics_v2, interval)
        results = self.__cleanup_results(column_results)

        return results

    def __check_columns__(self, df_gt, df_pred):
        col_list = []

        for col in df_pred.columns:
            if col in df_gt.columns:
                col_list.append(col)

        return col_list

    def __extract_datatime__(self, col_list, df_gt, df_pred, end_date, start_date):
        temp_gt = df_gt[col_list]
        temp_pred = df_pred[col_list]
        temp_start = 'None'
        temp_end = 'None'
        if start_date != 'None':
            temp_start = pd.to_datetime(start_date)
        if end_date != 'None':
            temp_end = pd.to_datetime(end_date)
        if start_date == 'None' and end_date == 'None':
            pass

        elif start_date == 'None' and end_date != 'None':
            temp_gt = temp_gt[temp_gt.index < temp_end]
            temp_pred = temp_pred[temp_pred.index < temp_end]

        elif start_date != 'None' and end_date == 'None':
            temp_gt = temp_gt[temp_gt.index >= temp_start]
            temp_pred = temp_pred[temp_pred.index >= temp_start]

        elif start_date != 'None' and end_date != 'None':
            temp_gt = temp_gt[(temp_gt.index >= temp_start) & (temp_gt.index < temp_end)]
            temp_pred = temp_pred[(temp_pred.index >= temp_start) & (temp_pred.index < temp_end)]
        return temp_gt, temp_pred

    def __special_metrics__(self, metrics_list):
        return metrics_list

    def __calculate_metrics__(self, col_list, sampled_time, temp_gt, temp_pred, metric_list, interval):
        column_results = {}
        for col in col_list:

            metrics_results = {}
            for calc in metric_list:

                results_timeframe = {}
                for timeframe in sampled_time:

                    if interval == 'Week':
                        if len(timeframe) > 1:
                            results_timeframe[timeframe[0]] = self.callFunction(calc,
                                                                                temp_gt[col][timeframe[0]:timeframe[1]],
                                                                                temp_pred[col][timeframe[0]:timeframe[1]])
                        else:
                            pass
                    else:
                        results_timeframe[timeframe] = self.callFunction(calc,
                                                                         temp_gt[col][timeframe],
                                                                         temp_pred[col][timeframe])

                metrics_results[calc] = results_timeframe

            metrics_results = pd.DataFrame(metrics_results)
            metrics_results.index = pd.to_datetime(metrics_results.index)

            column_results[col] = metrics_results
        return column_results

    def __remove_undefinedMetrics__(self, metric_list):
        new_list = metric_list.copy()
        for metric in metric_list:
            if self.checkFunction(metric):
                pass
            else:
                new_list = new_list[new_list != metric]
        return new_list

    def __cleanup_results(self, results):
        return results
