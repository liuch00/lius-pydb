
import numpy as np
from .rid import RID

class Record:

    def __init__(self, rid: RID, record: np.ndarray):
        self.record = record
        self.rid = rid
