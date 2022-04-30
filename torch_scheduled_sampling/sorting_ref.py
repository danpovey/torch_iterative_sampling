#!/usr/bin/env python3

import random
from typing import List


def is_sorted(x: List[int]):
    return x == sorted(x)


def sort_unique(x: List[int]):
    assert len(set(x)) == len(x)
    x_old = list(x)
    for i in range(len(x)):
        total_lt_x = 0
        for j in range(len(x)):
            if x_old[j] < x_old[i]:
                total_lt_x += 1
        x[total_lt_x] = x_old[i]
    return x

def merge_lists(new_sublist_size: int,
                max_elements_needed: int,
                x: List[int]):
    """
    Merge adjacent pairs of sorted sublists of size `new_sublist_size//2` into
    sorted sublists of size `new_sublist_size`.  Assume that for each
    pair of lists that we merge, the 2nd one is pointwise >= the 1st one.
    """
    x_old = list(x)
    old_sublist_size = new_sublist_size // 2
    for i in range(len(x)):
        x_val = x_old[i]

        offset_in_old = i & (old_sublist_size-1)  # i % old_sublist_size

        new_sublist_start = (i & ~(new_sublist_size - 1))
        is_rhs = (i & old_sublist_size)
        other_list_start = new_sublist_start | (is_rhs ^ old_sublist_size)

        search_offset = other_list_start
        search_begin = 0
        search_end = min(max_elements_needed, old_sublist_size) + 1

        # x_val_mod is about >= vs >.  For stable sort, we treat rhs values
        # as larger than LHS values.  We need the final position of values to be
        # deterministic and in agreement between lhs and rhs.
        #
        # This assumes integer inputs.
        x_val_mod = x_val + (is_rhs != 0)

        while search_begin + 1 < search_end:
            mid = (search_begin + search_end) // 2
            print(f"i={i}, other_list_start={other_list_start}, Comparing {x_val_mod} vs {x_old[search_offset + mid - 1]}")
            if x_val_mod > x_old[search_offset + mid - 1]:
                search_begin = mid
            else:
                search_end = mid
        new_pos = new_sublist_start + offset_in_old + search_begin
        print(f"{new_pos} = {new_sublist_start} + {offset_in_old} + {search_begin}")
        x[new_pos] = x_val




def _test_merge():
    for _ in range(1000):
        max_elements_needed = 8
        N = 16
        x = [ random.randint(0, 100) for _ in range(N) ]

        for list_size in [2,4,8,16]:
            print("list_size=", list_size)
            x_copy = list(x)
            merge_lists(list_size, 16, x)
            for i in range(0, N, list_size):
                size = min(max_elements_needed, list_size)
                assert x[i:i+size] == list(sorted(x_copy[i:i+list_size]))[:size]

def _test_sort_unique():
    for i in range(1000):
        N = 16
        x = [ random.randint(0, 100) for _ in range(N) ]
        if len(set(x)) == len(x):  # if unique..
            print("x = ", x)
            print("sort_unique(x) = ", sort_unique(x))
            assert sort_unique(x) == sorted(x)

if __name__ == '__main__':
    _test_sort_unique()
    _test_merge()
