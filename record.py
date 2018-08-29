class Record:

    def __init__(self, question_raw, answer_raw, question_pre, answer_pre):
        self.q_raw = question_raw
        self.a_raw = answer_raw
        self.q_pre = question_pre
        self.a_pre = answer_pre

    def __str__(self):
        return "(Q raw: " + self.q_raw + ")" + \
               "(A raw: " + self.a_raw + ")" + \
               "(Q pre: " + self.q_pre + ")" + \
               "(A pre: " + self.a_pre + ")"
