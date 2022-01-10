class RID:
    def __init__(self, page_value, slot_value):
        self._page = page_value
        self._slot = slot_value

    @property
    def page(self):
        return self._page

    @property
    def slot(self):
        return self._slot

    def __str__(self):
        return f'{{page: {self.page}, slot: {self.slot}}}'

    def __hash__(self):
        return hash((self._page, self._slot))

    def __eq__(self, other):
        if other is None:
            return False
        else:
            ans = self._page == other.page and self._slot == other.slot
            return ans
