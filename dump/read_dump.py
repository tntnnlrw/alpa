import pickle
import numpy as np
import dataclasses
from enum import Enum, auto
import logging
import multiprocessing
import os
import time
import traceback
from typing import Sequence, Optional, Union, Tuple
import warnings

import numpy as np
from jax._src.lib import xla_client as xc, xla_extension as xe
from jax.core import ShapedArray
from jax.interpreters import pxla
import argparse

INFINITY_COST = 1e13
def _call_solver_serialized_args(N,
                                 M,
                                 s_len_np,
                                 s_follow_np,
                                 E_np,
                                 A_np,
                                 L_np,
                                 c_np,
                                 d_np,
                                 m_np,
                                 r_np,
                                 v_np,
                                 args,
                                 s_init_np=None):
    """Call the solver with serialized arguments."""
    # pylint: disable=invalid-name
    global last_s_val, last_objective

    import pulp
    from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus
    tic = time.time()

    for x in [s_len_np, E_np, A_np, L_np, c_np, d_np, m_np, r_np, v_np]:
        assert isinstance(x, np.ndarray)
    assert len(s_len_np) == N, "s_len_np"

    # Dump arguments for re-solving
    # pickle.dump([N, M, s_len_np, s_follow_np, E_np, A_np, L_np,
    #              c_np, d_np, m_np, r_np, v_np, s_init_np],
    #              open("args.pkl", "wb"))
    # TODO(lmzheng): cache the ILP solution.

    def get_non_zero_index(binary_vector):
        """Get the index of non-zero item in a vector."""
        ct = 0
        ret = None
        for i, elem in enumerate(binary_vector):
            if pulp.value(elem):
                ret = i
                ct += 1

        assert ct == 1
        return ret

    # 0. Unpack flatten numpy arrays
    s_len = s_len_np
    s_follow = s_follow_np

    E = E_np.reshape((-1, 2))  # noqa
    r = []
    pt = 0
    edge_set = set()
    for (i, j) in E:
        prod_length = s_len[i] * s_len[j]

        if (i, j) in edge_set:
            raise ValueError(f"Duplicated edges: {(i, j)}")

        edge_set.add((i, j))
        r.append(r_np[pt:pt + prod_length])
        pt += prod_length
    assert pt == len(r_np)

    A = A_np.reshape((-1, 2))  # noqa
    v = []
    pt = 0
    for (i, j) in A:
        prod_length = s_len[i] * s_len[j]
        v.append(v_np[pt:pt + prod_length])
        pt += prod_length
    assert pt == len(v_np)

    L = []  # noqa
    pt = N
    for i in range(N):
        length = L_np[i]
        L.append(L_np[pt:pt + length])
        pt += length
    assert pt == len(L_np)
    
    # print(r_np.type())
    def fill_rand_nums(x_np):
        for i in range(len(x_np)):
            # if x_np[i] == 0:
            x_np[i] = np.random.rand()
        return x_np
    if args.random:
        c_np = fill_rand_nums(c_np)
        d_np = fill_rand_nums(d_np)
        r_np = fill_rand_nums(r_np)
    lenc=len(c_np)
    lenr=len(r_np)
    c = []
    d = []
    m = []
    pt = 0
    for i in range(N):
        length = s_len[i]
        c.append(c_np[pt:pt + length])
        d.append(d_np[pt:pt + length])
        m.append(m_np[pt:pt + length])
        pt += length
    assert pt == len(c_np), f"{pt} == {len(c_np)}"
    assert pt == len(d_np), f"{pt} == {len(d_np)}"
    assert pt == len(m_np), f"{pt} == {len(m_np)}"

    # 1. Create variables
    s = []   
    e = []

    num_nodes = 0
    reverse_follow_backpatch = []
    for i in range(N):
        if s_follow[i] < 0:
            if s_len[i] == 1:
                s.append([1])
            else:
                num_nodes += 1
                s.append(
                    LpVariable.matrix(f"s[{i}]", (range(s_len[i]),),
                                      cat="Binary"))
        else:
            if s_follow[i] < len(s):
                s.append(s[s_follow[i]])
            else:
                s.append(None)
                reverse_follow_backpatch.append(i)

    for i in reverse_follow_backpatch:
        s[i] = s[s_follow[i]]
  
    num_edges = 0
    for (idx, (i, j)) in enumerate(E):
        if len(s[i]) == 1:
            e.append(s[j])
        elif len(s[j]) == 1:
            e.append(s[i])
        else:
            num_edges += 1
            e.append(
                LpVariable.matrix(f"e[{i},{j}]",
                                  (range(len(s[i]) * len(s[j])),),
                                  cat="Binary"))
        assert len(e[idx]) == len(r[idx])

    # 2. Set initial value for warm start
    if s_init_np is not None:
        s_init = s_init_np.reshape((-1, 3))
        for (idx, value, fix) in s_init:
            for i in range(len(s[idx])):
                s[idx][i].setInitialValue(i == value)
                if fix:
                    s[idx][i].fixValue()

    # 3. Objective
    prob = LpProblem("myProblem", LpMinimize)
    # compute cost
    obj = 0
    for i in range(N):
        obj += lpDot(s[i], c[i]) + lpDot(s[i], d[i])

    # communication cost
    for i in range(len(E)):
        obj += lpDot(e[i], r[i])

    prob += obj

    # 4. Constraints
    # (a). specified by `cat="Binary"`

    # (b)
    for i in range(N):
        if s_follow[i] < 0:
            prob += lpSum(s[i]) == 1

    # (c)
    if args.memory:
        M = 30*1024*1024*1024
    if M > 0:
        for t in range(N):
            mem = 0
            for i in L[t]:
                mem += lpSum(s[i][j] * m[i][j] for j in range(len(s[i])))
            prob += mem <= M

    # (d). specified by `cat="Binary"`

    for (idx, (i, j)) in enumerate(E):
        if s_len[i] == 1 or s_len[j] == 1:
            continue

        # (e)
        prob += lpSum(e[idx]) == 1

        # (f)
        for row in range(len(s[i])):
            C = len(s[j])  # noqa
            prob += lpSum(
                e[idx][row * C + col] for col in range(0, C)) <= s[i][row]

        # (g)
        for col in range(len(s[j])):
            R = len(s[i])  # noqa
            C = len(s[j])  # noqa
            prob += lpSum(
                e[idx][row * C + col] for row in range(0, R)) <= s[j][col]

    # (h)
    alias_set = set()
    for (idx, (i, j)) in enumerate(A):
        R = len(s[i])  # noqa
        C = len(s[j])  # noqa
        if (i, j) in alias_set:
            raise ValueError(f"Duplicated edges: {(i, j)}")

        alias_set.add((i, j))
        alias_set.add((j, i))

        for row in range(len(s[i])):
            for col in range(len(s[j])):
                if v[idx][row * C + col] > 0.5:
                    prob += s[i][row] + s[j][col] <= 1

    verbose = False

    msg = verbose
    time_limit = 600
    assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
        "Please install ILP solvers by 'sudo apt install coinor-cbc'")

    solver = pulp.PULP_CBC_CMD(mip=True,
                               msg=msg,
                               timeLimit=time_limit,
                               threads=multiprocessing.cpu_count())
    prob.solve(solver)

    status = prob.status
    objective = pulp.value(prob.objective)
    objective = float(objective) if objective is not None else -1.0
    if verbose:
        print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}\t"
              f"Time: {time.time() - tic}")
        print(f"#nodes: {num_nodes},  #edges: {num_edges}")

    if prob.status in [pulp.LpStatusInfeasible]:
        raise RuntimeError(
            "Cannot run the function under the given memory budget. "
            "Please increase the memory budget.")

    # Get and check results
    s_val = np.full((N,), -1, dtype=np.int32)
    for i in range(N):
        s_val[i] = get_non_zero_index(s[i])

    e_val = np.full((len(E),), -1, dtype=np.int32)
    for (idx, (i, j)) in enumerate(E):
        e_val[idx] = get_non_zero_index(e[idx])
        i_spec_index = e_val[idx] // len(s[j])
        j_spec_index = e_val[idx] % len(s[j])
        assert i_spec_index == s_val[i], f"e_val[{i}][{j}]"
        assert j_spec_index == s_val[j], f"e_val[{i}][{j}]"
        if verbose and r[idx][e_val[idx]] > 0:
            print(f"Edge cost {(i, j)} : {r[idx][e_val[idx]]}")

    last_s_val = s_val
    last_objective = objective

    if objective > INFINITY_COST:
        warnings.warn("Detect unexpected behaviors in the auto-sharding pass.")

    return lenc, lenr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_file',
                        type=str,
                        help='set the loader file for experiment data')
    parser.add_argument('--random', action='store_true', help='randomize the c,d and r')
    parser.add_argument('--memory', action='store_true', help='set M=memory')

    args = parser.parse_args()

    with open(args.load_file,'rb') as f: 
        [N, M, s_len_np, s_follow_np, E_np, A_np, L_np,c_np, d_np, m_np, r_np, v_np, s_init_np,num_nodes,num_edges] = pickle.load(f)


    np.set_printoptions(threshold=100000)

    time_consume=[]   
    # for i in range(3):
    start_time=time.time()
    lenc, lenr= _call_solver_serialized_args(N,
                                    M,
                                    s_len_np,
                                    s_follow_np,
                                    E_np,
                                    A_np,
                                    L_np,
                                    c_np,
                                    d_np,
                                    m_np,
                                    r_np,
                                    v_np,
                                    args,
                                    s_init_np)
    end_time=time.time()
    time_consume.append(end_time-start_time)
    if args.random:
        savefile='data/mr_'+args.load_file if args.memory else 'data/r_'+args.load_file
    else:
        savefile='data/m_'+args.load_file if args.memory else 'data/'+args.load_file
    np.savetxt(savefile,[num_nodes,num_edges]+time_consume)
    print([num_nodes,num_edges]+time_consume)
