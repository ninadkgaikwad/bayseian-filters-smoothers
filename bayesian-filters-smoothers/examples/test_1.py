import os
import sys

module_path = os.path.abspath(os.path.join('./bayesian-filters-smoothers/bayesian_filters_smoothers'))

print(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)

import bayesian_filters_smoothers as bfs

a=bfs.addnum(10,10)

print(a)

a=bfs.subnum(10,10)

print(a)

