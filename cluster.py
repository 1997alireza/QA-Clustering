class Cluster:

    def __init__(self, title=""):
        self.title = title
        self.records = []

    def __str__(self):
        res = "    " + self.title + ":\n"
        res += "    " + '-' * (len(self.title) + 1) + '\n'
        for record in self.records:
            res += str(record) + '\n'
        res += '\n'
        return res

    def print(self, top_n=-1):
        if top_n < 0 or top_n > len(self.records):
            top_n = len(self.records)
        print("    " + self.title + ":\n")
        print("    " + '-' * (len(self.title) + 1) + '\n')
        for i in range(top_n):
            print(str(self.recrod[i]) + '\n')
        print('\n')
