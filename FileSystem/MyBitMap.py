

LEAF_BIT = 32
MAX_LEVEL = 5
MAX_INNER_NUM = 67
BIAS = 5

class MyBitMap:
    def __init__(self, cap, k):
        self.size = (cap >> BIAS)

    def getMask(self, k):
        s = 0
