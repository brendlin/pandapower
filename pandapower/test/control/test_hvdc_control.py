# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandapower as pp
import pandapower.networks
import pytest
from pandapower import control


def test_hvdc_control():
    net = pp.networks.case118()
    pp.runpp(net)

    # pp.plotting.simple_plotly(net)
    pp.control.HVDC_Controller(net, 102, 42, (-15, 0, 15), (-50, 0, 50))

    pp.runpp(net, run_control=True)

    angles = net.res_bus.va_degree.loc[[102, 42]].values
    delta = angles[1] - angles[0]
    c = net.characteristic.object.at[0]
    p = net.sgen.loc[net.sgen.bus == 102, 'p_mw'].values[0]
    assert np.isclose(p, c(delta))


if __name__ == '__main__':
    pytest.main(['-s', __file__])
