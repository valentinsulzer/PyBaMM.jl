#
# convert an expression tree into a pack model
#

# TODO
# - x lumped thermal for pouch cells?
# - Jacobians
# - parallel
# - GPU offload
# - smarter initialization (use assume all cells have same current, solve prob)
import pybamm
from copy import deepcopy
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
import pybamm2julia


class offsetter(object):
    def __init__(self, offset):
        self._sv_done = []
        self.offset = offset

    def add_offset_to_state_vectors(self, symbol):
        # this function adds an offset to the state vectors
        new_y_slices = ()
        if isinstance(symbol, pybamm.StateVector):
            # need to make sure its in place
            if symbol.id not in self._sv_done:
                for this_slice in symbol.y_slices:
                    start = this_slice.start + self.offset
                    stop = this_slice.stop + self.offset
                    step = this_slice.step
                    new_slice = slice(start, stop, step)
                    new_y_slices += (new_slice,)
                self.replace_y_slices(symbol, *new_y_slices)
                symbol.set_id()
                self._sv_done += [symbol.id]
        elif isinstance(symbol, pybamm.StateVectorDot):
            # need to make sure its in place
            if symbol.id not in self._sv_done:
                for this_slice in symbol.y_slices:
                    start = this_slice.start + self.offset
                    stop = this_slice.stop + self.offset
                    step = this_slice.step
                    new_slice = slice(start, stop, step)
                    new_y_slices += (new_slice,)
                self.replace_y_slices(symbol, *new_y_slices)
                symbol.set_id()
                self._sv_done += [symbol.id]
        else:
            for child in symbol.children:
                self.add_offset_to_state_vectors(child)
                child.set_id()
            symbol.set_id()
            
    def replace_y_slices(self, symbol, *new_y_slices):
        for y_slice in new_y_slices:
            if not isinstance(y_slice, slice):
                raise TypeError("all y_slices must be slice objects")
        name_split = symbol.name.split("[")
        base_name = name_split[0]
        if new_y_slices[0].start is None:
            name = base_name + "[:{:d}".format(y_slice.stop)
        else:
            name = base_name + "[{:d}:{:d}".format(
                new_y_slices[0].start, new_y_slices[0].stop
            )
        if len(new_y_slices) > 1:
            name += ",{:d}:{:d}".format(new_y_slices[1].start, new_y_slices[1].stop)
            if len(new_y_slices) > 2:
                name += ",...,{:d}:{:d}]".format(
                    new_y_slices[-1].start, new_y_slices[-1].stop
                )
            else:
                name += "]"
        else:
            name += "]"
        symbol._y_slices = new_y_slices
        symbol._first_point = new_y_slices[0].start
        symbol._last_point = new_y_slices[-1].stop
        symbol.set_evaluation_array(new_y_slices, None)

class Pack(object):
    def __init__(
        self,
        model,
        netlist,
        parameter_values=None,
        functional=False,
        voltage_functional=False,
        thermal=False,
        build_jac=False,
        implicit=False,
        top_bc = "ambient",
        bottom_bc = "ambient",
        left_bc = "ambient",
        right_bc = "ambient",
        distribution_params = None,
        operating_mode = "CC"
    ):
        # this is going to be a work in progress for a while:
        # for now, will just do it at the julia level

        # Build the cell expression tree with necessary parameters.
        # think about moving this to a separate function.
        self.top_bc = top_bc
        self.bottom_bc = bottom_bc
        self.left_bc = left_bc
        self.right_bc = right_bc
        self._operating_mode = operating_mode

        self._implicit = implicit

        self.functional = functional
        self.voltage_functional = voltage_functional
        if self.voltage_functional:
            self.voltage_func = None
        self.build_jac = build_jac
        self._thermal = thermal

        if parameter_values is None:
            parameter_values = model.default_parameter_values
        
        self._distribution_params = {}
        if distribution_params is not None:
            if not self.functional:
                raise NotImplementedError(
                    "Distribution Parameters only works with functional packs!"
                )
            for param in distribution_params:
                dp = pybamm2julia.DistributionParameter(
                    distribution_params[param]["name"],
                    distribution_params[param]["mean"],
                    distribution_params[param]["stddev"]
                )
                parameter_values[param] = dp
                self._distribution_params[param] = dp
        else:
            self._distribution_params = None


        cell_current = pybamm2julia.PsuedoInputParameter("cell_current")
        self.cell_current = cell_current
        parameter_values.update({"Current function [A]": cell_current})

        if self._thermal:
            self.pack_ambient = pybamm.Scalar(
                parameter_values["Ambient temperature [K]"]
            )
            ambient_temperature = pybamm2julia.PsuedoInputParameter("ambient_temperature")
            self.ambient_temperature = ambient_temperature
            parameter_values.update({"Ambient temperature [K]": ambient_temperature})

        self.cell_parameter_values = parameter_values
        self.unbuilt_model = model
        #
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sim.build()
        self.len_cell_rhs = sim.built_model.len_rhs
        
        if self._implicit:
            self.cell_model = pybamm.numpy_concatenation(
                (pybamm.StateVectorDot(slice(0,self.len_cell_rhs)) - sim.built_model.concatenated_rhs), sim.built_model.concatenated_algebraic
            )
        else:
            self.cell_model = pybamm.numpy_concatenation(
                sim.built_model.concatenated_rhs, sim.built_model.concatenated_algebraic
            )

        self.timescale = sim.built_model.timescale

        self.len_cell_algebraic = sim.built_model.len_alg

        self.cell_size = self.cell_model.shape[0]
        self.built_model = sim.built_model


        if self.functional:
            sv = pybamm.StateVector(slice(0, self.cell_size))
            dsv = pybamm.StateVectorDot(slice(0,self.len_cell_rhs))
            if self._distribution_params is None:
                lsv = []
            else:
                lsv = list(self._distribution_params.values())

            if self._implicit:
                if self._thermal:
                    self.cell_model = pybamm2julia.PybammJuliaFunction(
                        [sv, cell_current, ambient_temperature, dsv] + lsv,
                        self.cell_model,
                        "cell!",
                        True,
                    )
                else:
                    self.cell_model = pybamm2julia.PybammJuliaFunction(
                        [sv, cell_current, dsv] + lsv, self.cell_model, "cell!", True
                    )
            else:
                if self._thermal:
                    self.cell_model = pybamm2julia.PybammJuliaFunction(
                        [sv, cell_current, ambient_temperature] + lsv,
                        self.cell_model,
                        "cell!",
                        True,
                    )
                else:
                    self.cell_model = pybamm2julia.PybammJuliaFunction(
                        [sv, cell_current] + lsv, self.cell_model, "cell!", True
                    )
        self._sv_done = []


        self.netlist = netlist
        self.process_netlist()

        # get x and y coords for nodes from graph.
        node_xs = [n for n in range(max(self.circuit_graph.nodes) + 1)]
        node_ys = [n for n in range(max(self.circuit_graph.nodes) + 1)]
        for row in netlist.itertuples():
            node_xs[row.node1] = row.node1_x
            node_ys[row.node1] = row.node1_y
        self.node_xs = node_xs
        self.node_ys = node_ys
        self.batt_string = None

    def lolz(self, Ns, Np):
        if self.batt_string is None:
            one_parallel = ""
            for np in range(Np):
                one_parallel += "🔋"
            batt = ""
            for ns in range(Ns):
                batt += one_parallel + "\n"
            self.batt_string = batt
        print(self.batt_string)

    def process_netlist(self):
        curr = [{} for i in range(len(self.netlist))]
        self.netlist.insert(0, "currents", curr)

        self.netlist = self.netlist.rename(
            columns={"node1": "source", "node2": "target"}
        )
        self.netlist["positive_node"] = self.netlist["source"]
        self.netlist["negative_node"] = self.netlist["target"]
        self.circuit_graph = nx.from_pandas_edgelist(self.netlist, edge_attr=True)

    # Function that adds new cells, and puts them in the appropriate places.
    def add_new_cell(self):
        # TODO: deal with variables dict here too.
        # This is the place to get clever.
        new_model = deepcopy(self.cell_model)
        # at some point need to figure out parameters
        my_offsetter = offsetter(self.offset)
        my_offsetter.add_offset_to_state_vectors(new_model)

        new_model.set_id()
        return new_model

    def get_new_terminal_voltage(self):
        symbol = deepcopy(self.built_model.variables["Terminal voltage [V]"])
        my_offsetter = offsetter(self.offset)
        if self.voltage_functional:
            sv = pybamm.StateVector(slice(0,self.len_cell_rhs+self.len_cell_algebraic))
            if self.voltage_func is None:
                if self._distribution_params is not None:
                    ldp = list(self._distribution_params)
                else:
                    ldp = []
                voltage_func = pybamm2julia.PybammJuliaFunction([sv, self.cell_current] + ldp, symbol, "voltage_func", True)
                self.voltage_func = voltage_func
            symbol = deepcopy(self.voltage_func)
        my_offsetter.add_offset_to_state_vectors(symbol)
        return symbol

    def get_new_cell_temperature(self):
        symbol = pybamm.Index(
            deepcopy(self.built_model.variables["Cell temperature [K]"]), slice(0, 1)
        )
        my_offsetter = offsetter(self.offset)
        my_offsetter.add_offset_to_state_vectors(symbol)
        return symbol

    def build_thermal_equations(self):
        for desc in self.batteries:
            batt = self.batteries[desc]
            batt_x = batt["x"]
            batt_y = batt["y"]
            neighbors = []
            x_diffs = []
            y_diffs = []
            for other_desc in self.batteries:
                if other_desc == desc:
                    # its the same battery
                    continue
                else:
                    other_x = self.batteries[other_desc]["x"]
                    other_y = self.batteries[other_desc]["y"]
                    y_diff = other_y - batt_y
                    x_diff = other_x - batt_x
                    x_diffs.append(x_diff)
                    y_diffs.append(y_diff)
                    is_vert = (abs(y_diff) == 3) and other_x == batt_x
                    is_horz = (abs(x_diff) == 1) and other_y == batt_y
                    if is_vert or is_horz:
                        neighbors.append(other_desc)
            num_neighbors = len(neighbors)
            if len(neighbors) > 0:
                expr = self.batteries[neighbors[0]]["temperature"]
            else:
                expr = self.pack_ambient
            for neighbor in neighbors[1:]:
                expr += self.batteries[neighbor]["temperature"]
            #Left Cell
            if all([x_diff <= 0.1 for x_diff in x_diffs]):
                if self.left_bc == "ambient":
                    neighbors.append(self.pack_ambient)
                    expr += self.pack_ambient
                    num_neighbors += 1
                elif self.left_bc == "symmetry":
                    pass
                else:
                    raise NotImplementedError("BC's must be ambient or symmetry")
            #Right Cell
            if all([x_diff >= 0 for x_diff in x_diffs]):
                if self.right_bc == "ambient":
                    neighbors.append(self.pack_ambient)
                    expr += self.pack_ambient
                    num_neighbors += 1
                elif self.top_bc == "symmetry":
                    pass
                else:
                    raise NotImplementedError("BC's must be ambient or symmetry")
            #Top Cell
            if all([y_diff <= 0 for y_diff in y_diffs]):
                if self.top_bc == "ambient":
                    neighbors.append(self.pack_ambient)
                    expr += self.pack_ambient
                    num_neighbors += 1
                elif self.top_bc == "symmetry":
                    pass
                else:
                    raise NotImplementedError("BC's must be ambient or symmetry")
            #Bottom Cell
            if all([y_diff >= 0 for y_diff in y_diffs]):
                if self.bottom_bc == "ambient":
                    neighbors.append(self.pack_ambient)
                    expr += self.pack_ambient
                    num_neighbors += 1
                elif self.bottom_bc == "symmetry":
                    pass
                else:
                    raise NotImplementedError("BC's must be ambient or symmetry")
            expr = expr / num_neighbors
            self.ambient_temperature.set_psuedo(self.batteries[desc]["cell"], expr)
            if self.build_jac:
                self.ambient_temperature.set_psuedo(
                    self.batteries[desc]["cell"].expr, expr
                )
            batt.update({"neighbors": neighbors})

    def build_pack(self):
        # this function builds expression trees to compute the current.

        # cycle basis is the list of loops over which we will do kirchoff mesh analysis
        mcb = nx.cycle_basis(self.circuit_graph)
        self.cyc_basis = deepcopy(mcb)

        # generate loop currents and current source voltages-- this is what we don't know.
        num_loops = len(mcb)

        curr_sources = [
            edge
            for edge in self.circuit_graph.edges
            if self.circuit_graph.edges[edge]["desc"][0] == "I"
        ]

        loop_currents = [
            pybamm.StateVector(slice(n, n + 1), name="current_{}".format(n))
            for n in range(num_loops)
        ]

        curr_sources = []
        n = num_loops
        for edge in self.circuit_graph.edges:
            if self.circuit_graph.edges[edge]["desc"][0] == "I":
                self.circuit_graph.edges[edge]["voltage"] = pybamm.StateVector(
                    slice(n, n + 1), name="current_source_{}".format(n)
                )
                n += 1
                curr_sources.append(edge)

        # now we know the offset, we should "build" the batteries here. will still need to replace the currents later.
        self.offset = num_loops + len(curr_sources)
        self.batteries = OrderedDict()
        for index, row in self.netlist.iterrows():
            desc = row["desc"]
            # I'd like a better way to do this.
            if desc[0] == "V":
                new_cell = self.add_new_cell()
                terminal_voltage = self.get_new_terminal_voltage()
                self.batteries[desc] = {
                    "cell": new_cell,
                    "voltage": terminal_voltage,
                    "current_replaced": False,
                    "ics": self.built_model.concatenated_initial_conditions
                }
                if self._thermal:
                    node1_x = row["node1_x"]
                    node2_x = row["node2_x"]
                    node1_y = row["node1_y"]
                    node2_y = row["node2_y"]
                    if node1_x != node2_x:
                        raise AssertionError("x's must be the same")
                    if abs(node1_y - node2_y) != 1:
                        raise AssertionError("batteries can only take up one y")
                    batt_y = min(node1_y, node2_y) + 0.5
                    temperature = self.get_new_cell_temperature()
                    self.batteries[desc].update(
                        {"x": node1_x, "y": batt_y, "temperature": temperature}
                    )
                params = {}
                if self._distribution_params is not None:
                    for param in self._distribution_params:
                        expr = self._distribution_params[param].sample()
                        self._distribution_params[param].set_psuedo(self.batteries[desc]["cell"], expr)
                        self._distribution_params[param].set_psuedo(self.batteries[desc]["ics"], expr)
                        if self.voltage_functional:
                            self._distribution_params[param].set_psuedo(self.batteries[desc]["cell"], expr)
                        params.update({self._distribution_params[param] : expr})
                        
                self.batteries[desc].update({"offset": self.offset})
                self.batteries[desc].update({"distribution parameters" : params})
                self.offset += self.cell_size

        if self._thermal:
            self.build_thermal_equations()

        self.num_cells = len(self.batteries)

        if len(curr_sources) != 1:
            raise NotImplementedError("can't do this yet")
        # copy the basis which we can use to place the loop currents
        basis_to_place = deepcopy(mcb)
        self.place_currents(loop_currents, basis_to_place)
        pack_eqs_vec = self.build_pack_equations(loop_currents, curr_sources)
        self.len_pack_eqs = len(pack_eqs_vec)
        pack_eqs = pybamm.numpy_concatenation(*pack_eqs_vec)

        cells = [d["cell"] for d in self.batteries.values()]
        cell_eqs = pybamm.numpy_concatenation(*cells)




        self.pack = pybamm.numpy_concatenation(pack_eqs, cell_eqs)
        if self._implicit:
            len_sv = len(cells)*self.cell_size + len(pack_eqs_vec)
            sv = pybamm.StateVector(slice(0,len_sv))
            dsv = pybamm.StateVectorDot(slice(0,len_sv))
            p = pybamm2julia.PsuedoInputParameter("lolol")
            t = pybamm.Time()
            self.pack = pybamm2julia.PybammJuliaFunction(
                [dsv, sv, p, t], self.pack, "pack", True
            )
        self.ics = self.initialize_pack(len(loop_currents),1)

    def initialize_pack(self, num_loops, num_curr_sources):
        curr_ics = pybamm.Vector([1.0 for curr_source in range(num_loops)])
        curr_source_v_ics = pybamm.Vector(
            [1.0 for curr_source in range(num_curr_sources)]
        )
        cell_ics = pybamm.numpy_concatenation(
            *[
                self.batteries[desc]["ics"]
                for desc in self.batteries
            ]
        )
        ics_function = pybamm.numpy_concatenation(*[curr_ics, curr_source_v_ics, cell_ics])
        ics_jl = pybamm2julia.PybammJuliaFunction([],ics_function,"u0!",inplace=False)
        return ics_jl

    def build_pack_equations(self, loop_currents, curr_sources):
        # start by looping through the loop currents. Sum Voltages
        pack_equations = []
        cells = []
        for i, loop_current in enumerate(loop_currents):
            # loop through the edges
            eq = []
            for edge in self.circuit_graph.edges:
                if loop_current in self.circuit_graph.edges[edge]["currents"]:
                    # get the name of the edge current.
                    edge_type = self.circuit_graph.edges[edge]["desc"][0]
                    desc = self.circuit_graph.edges[edge]["desc"]
                    direction = self.circuit_graph.edges[edge]["currents"][loop_current]
                    this_edge_current = loop_current
                    for current in self.circuit_graph.edges[edge]["currents"]:
                        if current == loop_current:
                            continue
                        elif (
                            self.circuit_graph.edges[edge]["currents"][current]
                            == direction
                        ):
                            this_edge_current = this_edge_current + current
                        else:
                            this_edge_current = this_edge_current - current
                    if edge_type == "R":
                        eq.append(
                            this_edge_current * self.circuit_graph.edges[edge]["value"]
                        )
                    elif edge_type == "I":
                        curr_source_num = self.circuit_graph.edges[edge]["desc"][1:]
                        if curr_source_num != "0":
                            raise NotImplementedError(
                                "multiple current sources is not yet supported"
                            )

                        if (
                            self.circuit_graph.edges[edge]["currents"][current]
                            == direction
                        ):
                            eq.append(self.circuit_graph.edges[edge]["voltage"])
                        else:
                            eq.append(-self.circuit_graph.edges[edge]["voltage"])
                    elif edge_type == "V":
                        # first, check and see if the battery has been done yet.
                        if not self.batteries[self.circuit_graph.edges[edge]["desc"]][
                            "current_replaced"
                        ]:
                            if self.circuit_graph.edges[edge]["currents"][loop_current][0] == self.circuit_graph.edges[edge]["positive_node"]:
                                this_edge_current = -this_edge_current
                            expr = this_edge_current
                            self.cell_current.set_psuedo(
                                self.batteries[self.circuit_graph.edges[edge]["desc"]][
                                    "cell"
                                ],
                                expr,
                            )
                            if self.build_jac:
                                self.cell_current.set_psuedo(
                                    self.batteries[
                                        self.circuit_graph.edges[edge]["desc"]
                                    ]["cell"].expr,
                                    expr,
                                )
                            self.batteries[self.circuit_graph.edges[edge]["desc"]][
                                "current_replaced"
                            ] = True
                        voltage = self.batteries[
                            self.circuit_graph.edges[edge]["desc"]
                        ]["voltage"]
                        if isinstance(self.unbuilt_model, pybamm.lithium_ion.SPM) or isinstance(self.unbuilt_model, pybamm.lithium_ion.SPMe) or self.voltage_functional:
                            self.cell_current.set_psuedo(
                                    self.batteries[
                                        self.circuit_graph.edges[edge]["desc"]
                                    ]["voltage"],
                                    expr,
                                )
                        if (
                            direction[0]
                            == self.circuit_graph.edges[edge]["positive_node"]
                        ):
                            eq.append(voltage)
                        else:
                            eq.append(-voltage)
                        # check to see if the battery input current has been replaced yet.
                        # If not, replace the current with the actual current.

            if len(eq) == 0:
                raise NotImplementedError(
                    "packs must include at least 1 circuit element"
                )
            elif len(eq) == 1:
                expr = eq[0]
            else:
                expr = eq[0] + eq[1]
                for e in range(2, len(eq)):
                    expr = expr + eq[e]
            # add equation to the pack.
            pack_equations.append(expr)

        # then loop through the current source voltages. Sum Currents.
        for i, curr_source in enumerate(curr_sources):
            currents = list(self.circuit_graph.edges[curr_source]["currents"])
            source_value = pybamm.Scalar(self.circuit_graph.edges[curr_source]["value"])
            pack_voltage = self.circuit_graph.edges[curr_source]["voltage"]
            #initialize the loop (0 will be optimized away)
            expr = pybamm.Scalar(0)
            for current in currents:
                if (
                    self.circuit_graph.edges[curr_source]["currents"][current][0]
                    == self.circuit_graph.edges[curr_source]["positive_node"]
                ):
                    expr = expr - current
                else:
                    expr = expr + current
            if self._operating_mode == "CC":
                pack_equations.append(expr + source_value)
            elif self._operating_mode == "CV":
                pack_equations.append(pack_voltage - source_value)
            elif self._operating_mode == "CP":
                pack_equations.append((pack_voltage * expr) + source_value)
            elif isinstance(self._operating_mode, pybamm.Experiment):
                experiment = self._operating_mode
                forcing_functions = []
                termination_functions = []
                for op_cond in experiment.operating_conditions:
                    local_termination_expressions = []
                    #Forcing Functions
                    if op_cond["type"] == "power":
                        source_value = pybamm.Scalar(op_cond["Power input [W]"])
                        forcing_expression = ((pack_voltage * expr) + source_value)
                    elif op_cond["type"] == "current":
                        source_value = pybamm.Scalar(op_cond["Current input [A]"])
                        forcing_expression = (expr + source_value)
                    elif op_cond["type"] == "voltage":
                        source_value = pybamm.Scalar(op_cond["Voltage input [V]"])
                        forcing_expression = (pack_voltage - source_value)
                    #Termination Conditions
                    t = pybamm.Time()
                    if op_cond["time"] is not None:
                        max_t = pybamm.Scalar(op_cond["time"]) 
                        time_event = max_t - t
                        local_termination_expressions.append(time_event)
                    if op_cond["events"] is not None:
                        if "Current input [A]" in op_cond["events"]:
                            curr_event = pybamm.AbsoluteValue(expr) - pybamm.Scalar(op_cond["events"]["Current input [A]"])
                            local_termination_expressions.append(curr_event)
                        if "Voltage input [V]" in op_cond["events"]:
                            if op_cond["type"] == "power":
                                inp = op_cond["Power input [W]"]
                            elif op_cond["type"] == "current":
                                inp = op_cond["Current input [A]"]
                            else:
                                raise NotImplementedError(
                                    "voltage cutoff only implemented for power and current"
                                )
                            sign_begin = pybamm.Scalar(np.sign(inp))
                            voltage_event = sign_begin * (pack_voltage - op_cond["events"]["Voltage input [V]"])
                            local_termination_expressions.append(voltage_event)
                    sv = pybamm.StateVector(slice(0,len(pack_equations) + 1))
                    total_termination_expression = pybamm.numpy_concatenation(*local_termination_expressions)
                    termination_function = pybamm2julia.PybammJuliaFunction([sv, t], total_termination_expression, "termination_function", False)
                    forcing_function = pybamm2julia.PybammJuliaFunction([sv, t], forcing_expression, "forcing_function", True)
                    forcing_functions.append(forcing_function)
                    termination_functions.append(termination_function)
                self.forcing_functions = forcing_functions
                self.termination_functions = termination_functions
                pack_equations.append(forcing_functions[0])


                    
                    

                    

        # concatenate all the pack equations and return it.
        return pack_equations

    # This function places the currents on the edges in a predefined order.
    # it begins at loop 0, and loops through each "loop" -- really a cycle
    # of the mcb (minimum cycle basis) of the graph which defines the circuit.
    # Once it finds a loop in which the current node is in, it places the
    # loop current on each edge. Once the loop is finished, it removes the
    # loop and then proceeds to the next node and does the same thing. It
    # loops until all the loop currents have been placed.
    def place_currents(self, loop_currents, mcb):
        bottom_loop = 0
        for this_loop, loop in enumerate(mcb):
            node = loop[0]
            done_nodes = set()
            this_node = node
            last_one = False
            while True:
                done_nodes.add(this_node)
                neighbors = self.circuit_graph.neighbors(this_node)
                

                my_neighbors = set(
                    neighbors
                ).intersection(set(loop))
                # if there are no neighbors in the group that have not been done, ur done!
                my_neighbors = my_neighbors - done_nodes

                if len(my_neighbors) == 0:
                    break
                elif len(my_neighbors) == 1:
                    next_node = min(my_neighbors)
                else:
                    next_node = min(my_neighbors)
                
                if len(loop) == len(done_nodes) + 1 and not last_one:
                    last_one = True
                    done_nodes.remove(node)
                
                # go find the edge.
                edge = self.circuit_graph.edges.get((this_node, next_node))
                if edge is None:
                    edge = self.circuit_graph.edges.get((next_node, this_node))
                if edge is None:
                    raise KeyError("uh oh")

                # add this current to the loop.
                direction = (this_node, next_node)

                edge["currents"].update({loop_currents[this_loop]: direction})
                this_node = next_node
