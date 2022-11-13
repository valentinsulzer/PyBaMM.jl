import pandas as pd
import numpy as np

def setup_circuit(
    Np=1,
    Ns=1,
    Ri=1e-2,
    Rc=1e-2,
    Rb=1e-4,
    Rt=1e-5,
    I=80.0,
    V=4.2,
    plot=False,
    terminals="left",
):
    """
    Define a netlist from a number of batteries in parallel and series

    Args:
        Np (int): Number of batteries in parallel.
        Ns (int): Number of batteries in series.
        Ri (float): Internal resistance ($\Omega$).
        Rc (float): Connection resistance ($\Omega$).
        Rb (float): Busbar resistance ($\Omega$).
        Rt (float): Terminal connection resistance ($\Omega$).
        I (float): Current (A).
        V (float): Initial battery voltage (V).
        plot (bool): Plot the circuit.
        terminals (string): The location of the terminals. Can be "left", "right",
            "left-right", "right-left" or a list or array of node integers.

    Returns:
        pandas.DataFrame:
            A netlist of circuit elements with format desc, node1, node2, value.

    """
    Nc = Np
    Nr = Ns * 3 + 1

    grid = np.arange(Nc * Nr).reshape([Nr, Nc])
    coords = np.indices(grid.shape)
    y = coords[0, :, :]
    x = coords[1, :, :]
    # make contiguous now instead of later when netlist is done as very slow
    mask = np.ones([Nr, Nc], dtype=bool)
    # This is no longer needed as terminals connect directly to battery
    # Guess could also add a terminal connection resistor though
    # mask[1:-1, 0] = False
    grid[mask] = np.arange(np.sum(mask)) + 1
    x = x[mask].flatten()
    y = y[mask].flatten()
    grid[~mask] = -2  # These should never be used

    # grid is a Nr x Nc matrix
    # 1st column is terminals only
    # 1st and last rows are busbars
    # Other rows alternate between series resistor and voltage source
    # For example if Np=1 and Nc=2,
    # grid = array([[ 0,  1], # busbar
    #                         # Rs
    #               [ 2,  3],
    #                         # V
    #               [ 4,  5],
    #                         # Ri
    #               [ 6,  7],
    #                         # Rs
    #               [ 8,  9],
    #                         # V
    #               [10, 11],
    #                         # Ri
    #               [12, 13]] # busbar)
    # Connections are across busbars in first and last rows, and down each column
    # See "01 Getting Started.ipynb"

    # Build data  with ['element type', node1, node2, value]
    netlist = []

    num_Rb = 0
    num_V = 0

    desc = []
    node1 = []
    node2 = []
    value = []

    # -ve busbars (bottom row of the grid)
    bus_nodes = [grid[0, :]]
    for nodes in bus_nodes:
        for i in range(len(nodes) - 1):
            # netline = []
            desc.append("Rbn" + str(num_Rb))
            num_Rb += 1
            node1.append(nodes[i])
            node2.append(nodes[i + 1])
            value.append(Rb)
    num_Rs = 0
    num_Ri = 0
    # Series resistors and voltage sources
    cols = np.arange(Nc)
    rows = np.arange(Nr)[:-1]
    rtype = ["Rc", "V", "Ri"] * Ns
    for col in cols:
        # Go down the column alternating Rs, V, Ri connections between nodes
        nodes = grid[:, col]
        for row in rows:
            if rtype[row] == "Rc":
                # Inter(c)onnection / weld
                desc.append(rtype[row] + str(num_Rs))
                num_Rs += 1
                val = Rc
            elif rtype[row] == "Ri":
                # Internal resistor
                desc.append(rtype[row] + str(num_Ri))
                num_Ri += 1
                val = Ri
            else:
                # Voltage source
                desc.append("V" + str(num_V))
                num_V += 1
                val = V
            node1.append(nodes[row + 1])
            node2.append(nodes[row])
            value.append(val)
            # netlist.append(netline)

    # +ve busbar (top row of the grid)
    bus_nodes = [grid[-1, :]]
    for nodes in bus_nodes:
        for i in range(len(nodes) - 1):
            # netline = []
            desc.append("Rbp" + str(num_Rb))
            num_Rb += 1
            node1.append(nodes[i])
            node2.append(nodes[i + 1])
            value.append(Rb)

    desc = np.asarray(desc)
    node1 = np.asarray(node1)
    node2 = np.asarray(node2)
    value = np.asarray(value)
    main_grid = {
        "desc": desc,
        "node1": node1,
        "node2": node2,
        "value": value,
        "node1_x": x[node1 - 1],
        "node1_y": y[node1 - 1],
        "node2_x": x[node2 - 1],
        "node2_y": y[node2 - 1],
    }

    # Current source - spans the entire pack
    if (terminals == "left") or (terminals is None):
        t_nodes = [0, 0]
    elif terminals == "right":
        t_nodes = [-1, -1]
    elif terminals == "left-right":
        t_nodes = [0, -1]
    elif terminals == "right-left":
        t_nodes = [-1, 0]
    elif isinstance(terminals, (list, np.ndarray)):
        t_nodes = terminals
    else:
        raise ValueError(
            'Please specify a valid terminals argument: "left", '
            + '"right", "left-right" or "right-left" or a list or '
            + "array of nodes"
        )
    # terminal nodes
    t1 = grid[-1, t_nodes[0]]
    t2 = grid[0, t_nodes[1]]
    # terminal coords
    x1 = x[t1 - 1]
    x2 = x[t2 - 1]
    y1 = y[t1 - 1]
    y2 = y[t2 - 1]
    nn = grid.max() + 1  # next node
    # coords of nodes forming current source loop
    if terminals == "left" or (
        isinstance(terminals, (list, np.ndarray)) and np.all(np.array(terminals) == 0)
    ):
        ix = x1 - 1
        dy = 0
    elif terminals == "right" or (
        isinstance(terminals, (list, np.ndarray)) and np.all(np.array(terminals) == -1)
    ):
        ix = x1 + 1
        dy = 0
    else:
        ix = -1
        dy = 1
    if dy == 0:
        desc = ["Rtp1", "I0", "Rtn1"]
        xs = np.array([x1, ix, ix, x2])
        ys = np.array([y1, y1, y2, y2])
        node1 = [t1, nn, 0]
        node2 = [nn, 0, t2]
        value = [Rt, I, Rt]
        num_elem = 3
    else:
        desc = ["Rtp0", "Rtp1", "I0", "Rtn1", "Rtn0"]
        xs = np.array([x1, x1, ix, ix, x2, x2])
        ys = np.array([y1, y1 + dy, y1 + dy, 0 - dy, 0 - dy, y2])
        node1 = [t1, nn, nn + 1, 0, nn + 2]
        node2 = [nn, nn + 1, 0, nn + 2, t2]
        hRt = Rt / 2
        value = [hRt, hRt, I, hRt, hRt]
        num_elem = 5

    desc = np.asarray(desc)
    node1 = np.asarray(node1)
    node2 = np.asarray(node2)
    value = np.asarray(value)
    current_loop = {
        "desc": desc,
        "node1": node1,
        "node2": node2,
        "value": value,
        "node1_x": xs[:num_elem],
        "node1_y": ys[:num_elem],
        "node2_x": xs[1:],
        "node2_y": ys[1:],
    }

    for key in main_grid.keys():
        main_grid[key] = np.concatenate((main_grid[key], current_loop[key]))
    netlist = pd.DataFrame(main_grid)
    return netlist

