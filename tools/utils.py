class global_vars(object):
    def __init__(self):
        self.losstype = 0

    def update_vars(self, i):
        self.losstype = i


losstype = global_vars()