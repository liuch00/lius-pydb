from lookup_element import Term, LookupOutput


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
            new_res = self.loop_join(outside, inside, outside_joined, inside_joined)
            self.union_merge(each_pair_0, each_pair_1)
            new_key = self.union_search(each_pair_0)
            self.res_map[new_key] = new_res
            res = new_res
        return res

    @staticmethod
    def get_values(value: tuple, block: tuple):
        res = ()
        for item in block:
            res = res + (value[item])
        return res

    def create_join_value(self, outside: tuple, outside_joined: tuple):
        row_index = 0
        join_value_map = {}
        for item in outside:
            val = self.get_values(item, outside_joined)
            if val in join_value_map:
                join_value_map[val].append(row_index)
            else:
                join_value_map[val] = [row_index]
            row_index = row_index + 1
        return join_value_map

    @staticmethod
    def check_join(len_outside_joined, len_inside_joined):
        if len_outside_joined != len_inside_joined:
            raise ValueError("join error!")

    def loop_join_data(self, outside: tuple, inside: tuple, outside_joined: tuple, inside_joined: tuple):
        len_outside = len(outside)
        len_inside = len(inside)
        len_outside_joined = len(outside_joined)
        len_inside_joined = len(inside_joined)
        self.check_join(len_outside_joined=len_outside_joined, len_inside_joined=len_inside_joined)
        if len_outside == 0 or len_inside == 0:
            return None, None, None
        len_outside_0 = len(outside[0])
        len_inside_0 = len(inside[0])
        outside_left = tuple(i for i in range(len_outside_0) if i not in outside_joined)
        inside_left = tuple(i for i in range(len_inside_0) if i not in inside_joined)
        res = []
        join_value = self.create_join_value(outside=outside, outside_joined=outside_joined)
        for inside_index in range(len_inside):
            tmp_value = self.get_values(inside[inside_index], inside_joined)
            if tmp_value in join_value:
                in_tmp_value = self.get_values(inside[inside_index], inside_joined)
                outside_list = join_value[in_tmp_value]
                for outside_index in outside_list:
                    t1 = self.get_values(outside[outside_index], outside_left)
                    t2 = self.get_values(inside[inside_index], inside_left)
                    t3 = self.get_values(outside[outside_index], outside_joined)
                    res.append(t1 + t2 + t3)
        return res, outside_left, inside_left

    def loop_join(self, outside: LookupOutput, inside: LookupOutput, outside_joined: tuple, inside_joined: tuple):
        len_out_data = outside.size()
        len_in_data = inside.size()
        if len_out_data > len_in_data:
            # swap in out
            tmp = outside
            outside = inside
            inside = tmp
            tmp_joined = outside_joined
            outside_joined = inside_joined
            inside_joined = tmp_joined
        outside_data = outside.data
        outside_head = outside.headers
        outside_joined_id = tuple(outside.header_id(item) for item in outside_joined)
        inside_data = inside.data
        inside_head = inside.headers
        inside_joined_id = tuple(inside.header_id(item) for item in inside_joined)

        joined_data, outside_left, inside_left = self.loop_join_data(outside=outside_data, inside=inside_data,
                                                                     outside_joined=outside_joined_id,
                                                                     inside_joined=inside_joined_id)
        if joined_data is None:
            res = LookupOutput(headers=[], data=[])
        else:
            h1 = self.get_values(outside_head, outside_left)
            h2 = self.get_values(inside_head, inside_left)
            head = h1 + h2 + outside_joined
            res = LookupOutput(headers=head, data=joined_data)
        for outside_h, inside_h in zip(outside_joined, inside_joined):
            res.insert_alias(inside_h, outside_h)
        for alias in outside.alias_map:
            out_t = outside.alias_map[alias]
            res.insert_alias(alias, out_t)
        for alias in inside.alias_map:
            in_t = inside.alias_map[alias]
            res.insert_alias(alias, in_t)
        return res
