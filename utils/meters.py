# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import time

from collections import OrderedDict
from numbers import Number


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val  # * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n


def format_meters(stats):
    postfix = OrderedDict(stats)
    for key in postfix.keys():
        if key == "loss":
            postfix[key] = "{:.4f}".format(postfix[key].avg)
        elif key == "correct":
            postfix[key] = "{:.2%}".format(postfix[key].avg)
        elif key == "tokens":
            postfix[key] = "{:.0f} ({:.1f})".format(
                postfix[key].sum, postfix[key].avg
            )
        elif key == "ori_tokens":
            postfix[key] = "{:.0f} ({:.1f})".format(
                postfix[key].sum, postfix[key].avg
            )
        # Number: limit the length of the string
        elif isinstance(postfix[key], Number):
            postfix[key] = "{:g}".format(postfix[key])
        # Meter: display both current and average value
        elif isinstance(postfix[key], AverageMeter):
            postfix[key] = "{:.2f} ({:.2f})".format(
                postfix[key].sum, postfix[key].avg
            )
        elif isinstance(postfix[key], TimeMeter):
            postfix[key] = "{:.0f}s".format(postfix[key].elapsed_time)
        elif isinstance(postfix[key], StopwatchMeter):
            postfix[key] = "{:.0f}s".format(postfix[key].sum)
        # Else for any other type, try to get the string conversion
        elif not isinstance(postfix[key], str):
            postfix[key] = str(postfix[key])
        # Else if it's a string, don't need to preprocess anything
    return postfix
