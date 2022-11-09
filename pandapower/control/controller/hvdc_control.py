# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
import numpy as np
from pandapower.control.basic_controller import Controller
from pandapower.control.controller.characteristic_control import CharacteristicControl
from pandapower.control.util.characteristic import SplineCharacteristic, Characteristic
from pandapower.toolbox import _detect_read_write_flag, read_from_net, write_to_net

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class HVDC_Controller(CharacteristicControl):
    def __init__(self, net, from_bus, to_bus, delta_va_degree_points, p_mw_points, use_spline=False, tol=1e-3):
        if use_spline:
            c = SplineCharacteristic(net, delta_va_degree_points, p_mw_points)
        else:
            c = Characteristic(net, delta_va_degree_points, p_mw_points)
        c_idx = c.index
        sgen_idx = pp.create_sgens(net, [from_bus, to_bus], 0, type="HVDC", controllable=False)
        super().__init__(net, "sgen", "p_mw", sgen_idx, "res_bus", "va_degree", [from_bus, to_bus], c_idx, tol=tol)
        self.values = np.array([0, 0])

    def initialize_control(self, net):
        super().initialize_control(net)
        self.values = np.array([0, 0])


    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # read input values
        input_values = read_from_net(net, self.input_element, self.input_element_index, self.input_variable,
                                     self.read_flag)
        delta_v = input_values[1] - input_values[0]
        # calculate set values
        set_value = net.characteristic.object.at[self.characteristic_index](delta_v)
        self.values = np.array([set_value, -set_value])
        # read previous set values
        output_values = read_from_net(net, self.output_element, self.output_element_index, self.output_variable,
                                      self.write_flag)
        # compare old and new set values
        diff = self.values - output_values
        # write new set values
        write_to_net(net, self.output_element, self.output_element_index, self.output_variable, self.values,
                     self.write_flag)
        return self.applied #and np.all(np.abs(diff) < self.tol)

    def __str__(self):
        return f"HVDC at buses {self.input_element_index} ({self.values.round(3) if self.values is not None else (0, 0)} MW)"
