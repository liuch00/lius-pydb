class RID:
    def __init__(self, page_value, slot_value):
        self._page = page_value
        self._slot = slot_value

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self, value):
        if value < 0:
            raise ValueError('score must >= 0')
        self._page = value

    @property
    def slot(self):
        return self._slot

    @page.setter
    def slot(self, value):
        if value < 0:
            raise ValueError('score must >= 0')
        self._slot = value

    def __str__(self):
        return f'{{page: {self.page}, slot: {self.slot}}}'

    def __eq__(self, other):
        return self._page == other.page and self._slot == other.slot

    def __hash__(self):
        return hash((self._page, self._slot))