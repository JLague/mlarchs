import math
from functools import reduce


def get_feature_size(in_size, kernel_size, stride=1, padding=0):
    return math.floor((in_size + 2*padding - kernel_size) / stride) + 1

def get_out_size(start_size, ops):
    return reduce(lambda a,b : get_feature_size(a, *b), ops, start_size)


if __name__ == '__main__':
    start_size = 32

    # (kernel_size, stride, padding)
    # ops = [
    #     (11, 4),
    #     (3, 2),
    #     (5, 1, 2),
    #     (3, 2),
    #     (3, 1, 1),
    #     (3, 1, 1),
    #     (3, 1, 1),
    #     (3, 2),
    # ]

    ops = [
        (3, 2, 1),
        (3, 2),
        (3, 1, 2),
        (3, 2),
        (3, 1, 1),
        (3, 1, 1),
        (3, 1, 1),
        (3, 2),
    ]

    print(get_out_size(start_size, ops))
