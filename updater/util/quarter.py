
class Quarter:
    def __init__(self, quarter):
        self.quarter = quarter

    def get_prev_quarter(self):
        if self.quarter[-1] == '1':
            return str(int(self.quarter[:4]) - 1)+'q4'
        else:
            return self.quarter[:5] + str(int(self.quarter[-1]) - 1)

    def get_next_quarter(self):
        if self.quarter[-1] == '4':
            return str(int(self.quarter[:4]) + 1) + 'q1'
        else:
            return self.quarter[:5] + str(int(self.quarter[-1]) + 1)

    def get_quarter(self):
        return self.quarter

    def get_year(self):
        return self.quarter[:4]

    def get_nq(self):
        return self.quarter[-1]+'q'

    def get_qn(self):
        return self.quarter[-2:]
