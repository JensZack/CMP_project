"""
Contains tests for the heat diffusion solver,
Including MMS tests on the taurus grid
"""
import dataclasses
from typing import Callable

import matplotlib.pyplot as plt



def gen_mms_cubic():
    """
    generate the MMS object for x ** 3 + y ** 3
    :return:
    """
    mms = MMS(
        lambda x, y: x ** 3 + y ** 3,
        lambda x, y: 6 * x + 6 * y
    )
    return mms

