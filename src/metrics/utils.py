import pandas as pd


def aux_group_interval(data, options):
    if options == 'None':
        temp_dates = data.index

        temp_list = []
        for i in temp_dates:
            temp_list.append(i)

        return temp_list
    elif options == 'Min':
        temp_dates = data.groupby([data.index.year,
                                   data.index.month,
                                   data.index.day,
                                   data.index.hour,
                                   data.index.minute])

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}-{:02d}-{:02d} {:02d}:{:02d}'.format(i[0], i[1], i[2], i[3], i[4]))

        return temp_list
    elif options == 'Hour':
        temp_dates = data.groupby([data.index.year,
                                   data.index.month,
                                   data.index.day,
                                   data.index.hour])

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}-{:02d}-{:02d} {:02d}'.format(i[0], i[1], i[2], i[3]))

        return temp_list
    elif options == 'Day':
        temp_dates = data.groupby([data.index.year,
                                   data.index.month,
                                   data.index.day])

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}-{:02d}-{:02d}'.format(i[0], i[1], i[2]))

        return temp_list
    elif options == 'Week':
        temp_dates = data.groupby(pd.Grouper(freq='W-Mon'))

        temp_list = []

        last_date = list(temp_dates.groups.keys())[len(list(temp_dates.groups.keys())) - 2]
        last_date = last_date.strftime('%Y-%m-%d')

        for i in list(temp_dates.groups.keys())[:-1]:
            temp_start = i.strftime('%Y-%m-%d')

            if temp_start != last_date:
                temp_end = i + pd.DateOffset(weeks=1) - pd.DateOffset(days=1)
                temp_end = temp_end.strftime('%Y-%m-%d')

                temp_list.append((temp_start, temp_end))

            else:
                temp_list.append((temp_start,))

        return temp_list
    elif options == 'Month':
        temp_dates = data.groupby([data.index.year,
                                   data.index.month])

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}-{:02d}'.format(i[0], i[1]))

        return temp_list
    elif options == 'Year':
        temp_dates = data.groupby(data.index.year)

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}'.format(i))

        return temp_list
    else:
        print('Option not valid!')
        return


def aux_get_size(state_gt):
    """
    Gets the size of list/array for iteration
    """

    temp_length = 0
    if isinstance(state_gt, list):
        temp_length = len(state_gt)
    else:
        temp_length = state_gt.shape[0]

    return temp_length


def aux_error_checking(state_gt, state_pred):
    """
    Checks for list/array size incompatibility
    """

    if isinstance(state_gt, list):
        if len(state_gt) != len(state_pred):
            print('Ground truth and predicted arrays must be of the same size')
            return True
        else:
            return False
    elif state_gt.shape[0] != state_pred.shape[0]:
        print('Ground truth and predicted arrays must be of the same size')
        return True
    else:
        return False
