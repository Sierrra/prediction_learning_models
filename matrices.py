# -*- coding: utf-8 -*-
"""
This module contains pay matrix samples
"""

from pandas import DataFrame

matching_pen_pl1 = DataFrame({
    0: [4.0, 0.0 ],
    1: [0.0, 4.0 ],})

matching_pen_pl2 = DataFrame({
    0: [0.0, 4.0 ],
    1: [4.0, 0.0 ],})

salmonII_matrrix_pl1 = DataFrame({
    0: [2.0, 2.0, 0.0, 0.0 ],
    1: [0.0, 2.0, 0.0, 2.0 ],
    2: [2.0, 0.0, 2.0, 0.0 ],
    3: [0.0, 0.0, 2.0, 2.0 ],})

salmonII_matrrix_pl2 = DataFrame({
    0: [0.0, 0.0, 2.0, 2.0 ],
    1: [2.0, 0.0, 2.0, 0.0 ],
    2: [0.0, 2.0, 0.0, 2.0 ],
    3: [2.0, 2.0, 0.0, 0.0 ],})

salmonIII_matrrix_pl1 = DataFrame({
    0: [2.0, 0.0, 0.0, 0.0, 0.0, 2.0 ],
    1: [0.0, 0.0, 2.0, 2.0, 2.0, 2.0 ],
    2: [0.0, 2.0, 0.0, 0.0, 2.0, 0.0 ],
    3: [0.0, 2.0, 2.0, 0.0, 0.0, 0.0 ],
    4: [0.0, 2.0, 0.0, 2.0, 2.0, 0.0 ],
    5: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0 ],})

salmonIII_matrrix_pl2 = DataFrame({
    0: [0.0, 2.0, 2.0, 2.0, 2.0, 0.0 ],
    1: [2.0, 2.0, 0.0, 0.0, 0.0, 0.0 ],
    2: [2.0, 0.0, 2.0, 2.0, 0.0, 2.0 ],
    3: [2.0, 0.0, 0.0, 2.0, 2.0, 2.0 ],
    4: [2.0, 0.0, 2.0, 0.0, 0.0, 2.0 ],
    5: [2.0, 2.0, 0.0, 2.0, 0.0, 0.0 ],})



matrix_set = [[matching_pen_pl1, matching_pen_pl2], [salmonII_matrrix_pl1, salmonII_matrrix_pl2], [salmonIII_matrrix_pl1, salmonIII_matrrix_pl2]]