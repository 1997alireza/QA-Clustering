class Cluster:

    def __init__(self, title=""):
        self.title = title
        self._records = []

    def __str__(self):
        res = "    " + self.title + ":\n"
        res += "    " + '-' * (len(self.title) + 1) + '\n'
        for record in self._records:
            res += str(record) + '\n'
        res += '\n'
        return res

    def get_records(self):
        return self._records[:]

    def add_record(self, record):
        self._records.append(record)

    def print(self, top_n=-1, just_raw_answers=False):
        if top_n < 0 or top_n > len(self._records):
            top_n = len(self._records)
        print("    " + self.title + ":\n")
        print("    " + '-' * (len(self.title) + 1) + '\n')
        for i in range(top_n):
            if just_raw_answers:
                print(self._records[i].a_raw + '\n')
            else:
                print(str(self._records[i]) + '\n')
        print('\n')


class LDACluster(Cluster):

    def __init__(self, title=""):
        Cluster.__init__(self, title)
        self._possibilities = []

    def add_record(self, record):
        self._records.append(record)
        self._possibilities.append(.0)

    def add_record_psb(self, record, possibility):
        for i in range(len(self._records)):
            if self._possibilities[i] < possibility:
                self._records.insert(i, record)
                self._possibilities.insert(i, possibility)
                return
        self._records.append(record)
        self._possibilities.append(possibility)
