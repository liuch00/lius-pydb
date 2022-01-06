from lookup_element import Reducer, Term, LookupOutput, Join


class Join:
    def __init__(self, res_map: dict, term, union=None):
        self.res_map = res_map
        self._term = term
        self._join_map = {}
        self._union = union

    def create_pair(self, term: Term):
        if term.aim_table is not None:
            if term.table != term.aim_table:
                if term.operator != '=':
                    raise TypeError('Join create_pair error!')
                pairs = (term.table, term.col), (term.aim_table, term.aim_col)
                sorted_pairs = zip(*sorted(pairs))
                res = tuple(sorted_pairs)
                return res
            else:
                return None, None
        else:
            return None, None

    def union_create(self):
        for key, element in map(self.create_pair, self._term):
            if element is None:
                continue
            else:
                if key in self._join_map:
                    element_0 = element[0]
                    element_1 = element[1]
                    self._join_map[key][0].append(element_0)
                    self._join_map[key][1].append(element_1)
                else:
                    element_0 = [element[0]]
                    element_1 = [element[1]]
                    self._join_map[key] = (element_0, element_1)

    def union_search(self, element):
        if element != self._union[element]:
            self._union[element] = self.union_search(self._union[element])
        return self._union[element]

    def union_merge(self, element_1, element_2):
        father_1 = self.union_search(element=element_1)
        father_2 = self.union_search(element=element_2)
        self._union[father_1] = father_2

    def get_output(self):
        res = None
        self.union_create()
        for each_pair in self._join_map:
            each_pair_0 = each_pair[0]
            outside: LookupOutput = self.res_map[each_pair_0]
            each_pair_1 = each_pair[1]
            inside: LookupOutput = self.res_map[each_pair_1]
            outside_joined = tuple(each_pair_0 + ".")
            for each_0 in self._join_map[each_pair][0]:
                outside_joined += tuple(each_0)
            inside_joined = tuple(each_pair_1 + ".")
            for each_1 in self._join_map[each_pair][1]:
                inside_joined += tuple(each_1)
            new_res = nested_loops_join(outside, inside, outside_joined, inside_joined)
            self.union_merge(each_pair_0, each_pair_1)
            new_key = self.union_search(each_pair_0)
            self.res_map[new_key] = new_res
            res = new_res
        return res
