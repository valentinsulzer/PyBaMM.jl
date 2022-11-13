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
                symbol.replace_y_slices(*new_y_slices)
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
                symbol.replace_y_slices(*new_y_slices)
                symbol.set_id()
                self._sv_done += [symbol.id]
        else:
            for child in symbol.children:
                self.add_offset_to_state_vectors(child)
                child.set_id()
            symbol.set_id()


class Pack(object):
    def __init__(
        self,
        model,
        netlist,
        parameter_values=None,
        functional=False,
        thermal=False,
        build_jac=False,
        implicit=False,
    ):
        # this is going to be a work in progress for a while:
        # for now, will just do it at the julia level

        # Build the cell expression tree with necessary parameters.
        # think about moving this to a separate function.

        self._implicit = implicit

        self.functional = functional
        self.build_jac = build_jac
        self._thermal = thermal

        if parameter_values is None:
            parameter_values = model.default_parameter_values

        cell_current = pybamm.PsuedoInputParameter("cell_current")
        self.cell_current = cell_current
        parameter_values.update({"Current function [A]": cell_current})

        if self._thermal:
            self.pack_ambient = pybamm.Scalar(
                parameter_values["Ambient temperature [K]"]
            )
            ambient_temperature = pybamm.PsuedoInputParameter("ambient_temperature")
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

        if self.functional:
            sv = pybamm.StateVector(slice(0, self.cell_size))
            dsv = pybamm.StateVectorDot(slice(0,self.len_cell_rhs))
            if self._implicit:
                if self._thermal:
                    self.cell_model = pybamm.PybammJuliaFunction(
                        [sv, cell_current, ambient_temperature, dsv],
                        self.cell_model,
                        "cell!",
                        True,
                    )
                else:
                    self.cell_model = pybamm.PybammJuliaFunction(
                        [sv, cell_current, dsv], self.cell_model, "cell!", True
                    )
            else:
                if self._thermal:
                    self.cell_model = pybamm.PybammJuliaFunction(
                        [sv, cell_current, ambient_temperature],
                        self.cell_model,
                        "cell!",
                        True,
                    )
                else:
                    self.cell_model = pybamm.PybammJuliaFunction(
                        [sv, cell_current], self.cell_model, "cell!", True
                    )
        self._sv_done = []
        self.built_model = sim.built_model

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
            for other_desc in self.batteries:
                if other_desc == desc:
                    # its the same battery
                    continue
                else:
                    other_x = self.batteries[other_desc]["x"]
                    other_y = self.batteries[other_desc]["y"]
                    is_vert = (abs(other_y - batt_y) == 3) and other_x == batt_x
                    is_horz = (abs(other_x - batt_x) == 1) and other_y == batt_y
                    if is_vert or is_horz:
                        neighbors.append(other_desc)
            ambient_start = len(neighbors)
            if len(neighbors) > 0:
                expr = self.batteries[neighbors[0]]["temperature"]
            else:
                expr = self.pack_ambient
            for neighbor in neighbors[1:]:
                expr += self.batteries[neighbor]["temperature"]
            for ambient_aux in range(ambient_start, 4):
                neighbors.append(self.pack_ambient)
                expr += self.pack_ambient
            expr = expr / 4
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
                self.batteries[desc].update({"offset": self.offset})
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
            p = pybamm.PsuedoInputParameter("lolol")
            t = pybamm.Time()
            self.pack = pybamm.PybammJuliaFunction(
                [dsv, sv, p, t], self.pack, "pack", True
            )
        self.ics = self.initialize_pack(num_loops, len(curr_sources))

    def initialize_pack(self, num_loops, num_curr_sources):
        curr_ics = pybamm.Vector([1.0 for curr_source in range(num_loops)])
        curr_source_v_ics = pybamm.Vector(
            [1.0 for curr_source in range(num_curr_sources)]
        )
        cell_ics = pybamm.numpy_concatenation(
            *[
                self.built_model.concatenated_initial_conditions
                for n in range(len(self.batteries))
            ]
        )
        ics = pybamm.numpy_concatenation(*[curr_ics, curr_source_v_ics, cell_ics])
        return ics

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
                        if isinstance(self.unbuilt_model, pybamm.lithium_ion.SPM) or isinstance(self.unbuilt_model, pybamm.lithium_ion.SPMe):
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
            expr = pybamm.Scalar(self.circuit_graph.edges[curr_source]["value"])
            for current in currents:
                if (
                    self.circuit_graph.edges[curr_source]["currents"][current][0]
                    == self.circuit_graph.edges[curr_source]["positive_node"]
                ):
                    expr = expr - current
                else:
                    expr = expr + current
            pack_equations.append(expr)

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
