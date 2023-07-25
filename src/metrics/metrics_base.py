
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
            if pred is None:
                return fn(gt)
            return fn(gt, pred)
        else:
            print('Undefined metric call')
            return
