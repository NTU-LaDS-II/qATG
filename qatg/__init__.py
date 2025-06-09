"""
qatg.

An open-source quantum automatic test generator, compatible with Qiskit.
"""

__version__ = "0.8.3"
__author__ = 'LaDS'
__credits__ = 'The Laboratory of Dependable Systems (LaDS), National Taiwan University'

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))

from qatgUtil import *
from qatgFault import QATGFault
from qatgConfiguration import QATGConfiguration
from qatgMain import QATG
