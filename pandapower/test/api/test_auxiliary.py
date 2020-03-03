# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
import gc
import copy
import numpy as np
import pandas as pd

from pandapower.auxiliary import get_indices

import pandapower as pp
import pandapower.networks
import pandapower.control
import pandapower.timeseries


class MemoryLeakDemo:
    """
    Dummy class to demonstrate memory leaks
    """
    def __init__(self, net):
        self.net = net
        # it is interesting, that if "self" is just an attribute of net, there are no problems
        # if "self" is saved in a DataFrame, it causes a memory leak
        net['memory_leak_demo'] = pd.DataFrame(data=[self], columns=['object'])


def test_get_indices():
    a = [i+100 for i in range(10)]
    lookup = {idx: pos for pos, idx in enumerate(a)}
    lookup["before_fuse"] = a

    # First without fused buses no magic here
    # after fuse
    result = get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 7])

    # before fuse
    result = get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])

    # Same setup EXCEPT we have fused buses now (bus 102 and 107 are fused)
    lookup[107] = lookup[102]

    # after fuse
    result = get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 2])

    # before fuse
    result = get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])


def test_net_deepcopy():
    net = pp.networks.example_simple()
    net.line_geodata.loc[0, 'coords'] = [[0,1], [1,2]]

    pp.control.ContinuousTapControl(net, tid=0, vm_set_pu=1)
    ds = pp.timeseries.DFData(pd.DataFrame(data=[[0,1,2], [3,4,5]]))
    pp.control.ConstControl(net, element='load', variable='p_mw', element_index=[0], profile_name=[0], data_source=ds)

    net1 = copy.deepcopy(net)
    assert net1.controller.object.at[0].net is net1
    assert net1.controller.object.at[1].net is net1

    assert not net1.controller.object.at[0].net is net
    assert not net1.controller.object.at[1].net is net

    assert not net1.controller.object.at[1].data_source is ds
    assert not net1.controller.object.at[1].data_source.df is ds.df

    assert not net1.line_geodata.coords.at[0] is net.line_geodata.coords.at[0]


def test_memory_leaks():
    net = pp.networks.example_simple()

    # first, test to check that there are no memory leaks
    types_dict1 = pp.toolbox.get_gc_objects_dict()

    for _ in range(100):
        net_copy = copy.deepcopy(net)
        # In each net copy it has only one controller
        pp.control.ContinuousTapControl(net_copy, tid=0, vm_set_pu=1)

    gc.collect()

    types_dict2 = pp.toolbox.get_gc_objects_dict()

    assert types_dict1[pandapower.auxiliary.pandapowerNet] == 1
    assert types_dict2[pandapower.auxiliary.pandapowerNet] == 2

    assert types_dict1.get(pandapower.control.ContinuousTapControl, 0) == 0
    assert types_dict2[pandapower.control.ContinuousTapControl] == 1

    # now, demonstrate how a memory leak occurs
    # emulates the earlier behavior before the fix with weakref
    for _ in range(10):
        net_copy = copy.deepcopy(net)
        MemoryLeakDemo(net_copy)

    # demonstrate how the garbage collector doesn't remove the objects even if called explicitly
    gc.collect()

    types_dict3 = pp.toolbox.get_gc_objects_dict()
    assert types_dict3[pandapower.auxiliary.pandapowerNet] == 11
    assert types_dict3[MemoryLeakDemo] == 10


if __name__ == '__main__':
    pytest.main([__file__, "-x"])
