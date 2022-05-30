"""
A file in which I define all the functions necessary for the calculation of the
different gradients.

Here, I define the functions to calculate the gradient to increase the distance
between two equilibria (distance_gradient()) and to manipulate the eigenvalues
of the different equilibrium points.
"""

import numpy as np
import sympy as sym
from scipy.linalg import eig
from scipy.optimize import fsolve
from scipy.optimize import minimize


def hill(s:float, k:float, n:int):
    """
    Returns te occupancy theta of a protein calculated with a Hill equation
    using the given parameters.

    Input:
    - s: Substrate concentration.
    - k: Affinity constant.
    - n: Cooperativity coeficient.

    Output:
    - theta: The occupancy of the protein.
    """
    num = s**n
    denom = s**n + k**n
    theta = num / denom
    return theta


def dS(state, parameters):
    S, E, N, P = state
    n, k1, k2, k3, k4, a1, a2, bs = [parameters["n"],
                                     parameters["k1"],
                                     parameters["k2"],
                                     parameters["k3"],
                                     parameters["k4"],
                                     parameters["a1"],
                                     parameters["a2"],
                                     parameters["bs"]
                                    ]
    dif = (a1 * (1 - hill(E, k1, n)) + a2 * hill(S, k2, n) * hill(E, k3, n)
           * hill(N, k4, n) + bs - S)
    return dif


def dE(state, parameters):
    S, E, N, P = state
    n, k5, k6, k7, k8, a3, a4, be = [parameters["n"],
                                     parameters["k5"],
                                     parameters["k6"],
                                     parameters["k7"],
                                     parameters["k8"],
                                     parameters["a3"],
                                     parameters["a4"],
                                     parameters["be"]
                                    ]
    dif = (a3 * (1 - hill(S, k5, n)) + a4 * hill(E, k6, n) * hill(S, k7, n)
           * (1 - hill(N, k8, n)) + be - E)
    return dif


def dN(state, parameters):
    S, E, N, P = state
    n, k9, k10, k11, k12, a5, a6, a7, a8, bn = [parameters["n"],
                                                parameters["k9"],
                                                parameters["k10"],
                                                parameters["k11"],
                                                parameters["k12"],
                                                parameters["a5"],
                                                parameters["a6"],
                                                parameters["a7"],
                                                parameters["a8"],
                                                parameters["bn"]
                                               ]
    dif = (a5 * hill(S, k9, n) + a6 * hill(E, k10, n) + a7 * hill(N, k11, n)
           + a8 * hill(P, k12, n) + bn - N)
    return dif


def dP(state, parameters):
    S, E, N, P = state
    n, k13, k14, k15, k16, a9, a10, a11, bp = [parameters["n"],
                                               parameters["k13"],
                                               parameters["k14"],
                                               parameters["k15"],
                                               parameters["k16"],
                                               parameters["a9"],
                                               parameters["a10"],
                                               parameters["a11"],
                                               parameters["bp"]
                                              ]
    dif = (a9 * (1 - hill(S, k13, n))
           * ((a10 * hill(E, k14, n) * hill(N, k15, n))
           + a11 * hill(P, k16, n)) + bp - P)
    return dif


def F_n(state, parameters):
    ds = dS(state, parameters)
    de = dE(state, parameters)
    dn = dN(state, parameters)
    dp = dP(state, parameters)
    return np.array([ds, de, dn, dp])





###############################################################################
###############################################################################
###############################################################################
###############################################################################
#%% Definition of symbolic functions

sys_vars = "s e nf p"
sys_pars = ("n k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 a1 a2 a3"
            " a4 a5 a6 a7 a8 a9 a10 a11 bs be bn bp")
# Parameter names must match keys from parameter dictionary

s, e, nf, p = sym.symbols(sys_vars, real = True)
n, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, bs, be, bn, bp = sym.symbols(sys_pars, real = True)

f1 = (a1 * (1 - hill(e, k1, n)) + a2 * hill(s, k2, n) * hill(e, k3, n)
      * hill(nf, k4, n) + bs - s)
f2 = (a3 * (1 - hill(s, k5, n)) + a4 * hill(e, k6, n) * hill(s, k7, n)
      * (1 - hill(nf, k8, n)) + be - e)
f3 = (a5 * hill(s, k9, n) + a6 * hill(e, k10, n) + a7 * hill(nf, k11, n)
      + a8 * hill(p, k12, n) + bn - nf)
f4 = (a9 * (1 - hill(s, k13, n))
      * ((a10 * hill(e, k14, n) * hill(nf, k15, n))
      + a11 * hill(p, k16, n)) + bp - p)
F_s = sym.Matrix([[f1], [f2], [f3], [f4]])

state_vars = sym.Matrix([[s], [e], [nf], [p]])
parameters = sym.Matrix([[k1],  # These are only the DIFFERENTIABLE
                         [k2],  # parameters. As I don't really care
                         [k3],  # for the derivative with respect to
                         [k4],  # n, I don't include it here.
                         [k5],
                         [k6],
                         [k7],
                         [k8],
                         [k9],
                         [k10],
                         [k11],
                         [k12],
                         [k13],
                         [k14],
                         [k15],
                         [k16],
                         [a1],
                         [a2],
                         [a3],
                         [a4],
                         [a5],
                         [a6],
                         [a7],
                         [a8],
                         [a9],
                         [a10],
                         [a11],
                         [bs],
                         [be],
                         [bn],
                         [bp]
                        ]
                       )





###############################################################################
###############################################################################
###############################################################################
###############################################################################
#%% Basic interaction funcions

def subs_pars(symb_fun, values):
    """
    A function that substitutes the parameters in the system.

    Input:
    - symb_fun: the symbolic function into which the values will be substituted.
    - values: a dictionary containing the parameter values that will be
      substituted.

    Output:
    - A new symbolic object with the desired substitutions.
    """
    subs_dict = {k:values[k.name] for k in parameters}
    subs_dict[n] = values["n"]
    symb_fun = symb_fun.subs(subs_dict)

    return symb_fun


def subs_state(symb_fun, state):
    """
    A function that does the substitution of the state variables in a symbolic
    function.
    """
    #keys = sys_vars.split()
    subs_dict = {state_vars[ii]:state[ii] for ii in range(len(state))}
    symb_fun = symb_fun.subs(subs_dict)

    return symb_fun


def get_eigenvalues(F, state, pars, jac = None):
    """
    A function that calculates the eigenvalues of the system at a given
    equilibrium point.

    Input:
    - F: symbolic version of the function.
    - state: np.array where the eigenvalues are going to be calculated.
    - pars: the system parameters.
    - jac: the symbolic version of the jacobian. Defaults to None.

    Output:
    - eig: np.array of the eigenvalues.
    """
    if jac is None:
        jac = F.jacobian(state_vars)
    jac = subs_state(jac, state)
    jac = subs_pars(jac, pars)
    jac = np.asmatrix(jac, dtype = float)

    eigenvalues = eig(jac)

    return eigenvalues





###############################################################################
###############################################################################
###############################################################################
###############################################################################
#%% Basic sensitivity funcions

def state_sens(F, state: np.array, pars: dict, jac = None, dfdp = None):
    """
    A function that calculates the sensitvity of an equilibrium point to the
    change in parameters.

    Input:
    - F: the symbolic representation of the dynamical system.
    - state: a numpy array with the state of the system.
    - pars: a dictionary of the parameters.
    - jac: a symbolic version of the jacobian. Defaults to None.
    - dfdp: a symbolic version of the derivative of the system with respect to
      the parameters. Defaults to None.

    Output:
    - sens: a matrix containing the sensitivities for every parameter, in the
      order they are in the 'parameters' variable in this same script.
      NOTE: both derivatives are taken using variables that are global in this
      script. I don't know if this is a bad idea, or a *very* bad idea.
    """
    if jac is None:
        jac = F.jacobian(state_vars)
    if dfdp is None:
        dfdp = F.jacobian(parameters)

    jac = subs_state(jac, state)
    jac = subs_pars(jac, pars)
    dfdp = subs_state(dfdp, state)
    dfdp = subs_pars(dfdp, pars)
    
    sens = -jac.inv() * dfdp
    sens = np.array(sens, dtype = float)

    return sens


def distance_gradient(F, eq1, eq2, pars: dict, **kwargs):
    """
    A function that calculates the direction in parameter space to optimally
    increase the distance between two equilibria eq1 and eq2, solutions to
    the symbolic system F with parameters pars.

    Input:
    - F: the symbolic representation of the dynamical system.
    - eqX: the equilibria of the system as np.arrays.
    - pars: a dictionary containing the named parameters of the system.

    Output:
    - d: an np.array representing the direction in parameter space for the
      optimal increase of (eq1 - eq2).
    """
    j1 = state_sens(F, eq1, pars, **kwargs)
    j2 = state_sens(F, eq2, pars, **kwargs)
    d = np.dot(2*(eq1 - eq2).T, (j1 - j2))
    d = np.array(d, dtype = float)

    return d


def eigenvar_gradient(F, eq, pars: dict):
    """
    A function that calculates the direction in parameter space to optimally
    DECREASE EACH EIGENVALUE of eq1, given a symbolic dynamical system F, and a
    dictionary of parameters pars.

    Input:
    - F: the symbolic representation of the dynamical system.
    - eq: the equilibrium point.
    - pars: a dictionary of parameters.

    Output:
    - out: list with the eigenvalues in [0] and the directions of biggest
      decrease in eigenvalue value in [1].
    """
    j = F.jacobian(state_vars)
    a = [j.diff(ii) for ii in parameters]
    h = [j.diff(ii) for ii in state_vars]

    j = subs_state(j, eq)
    j = subs_pars(j, pars)
    j = np.asarray(j, dtype = float)
    a = [subs_pars(jj, pars) for jj in a]
    a = [subs_state(jj, eq) for jj in a]
    a = [np.asmatrix(jj, dtype = float) for jj in a]
    a = np.array(a)
    h = [subs_pars(jj, pars) for jj in h]
    h = [subs_state(jj, eq) for jj in h]
    h = np.asarray(h, dtype = float)

    dp = state_sens(F, eq, pars)
    DJ = [np.tensordot(h, dp[:, ii], ([0], [0])) + a[ii] for ii in range(len(parameters))]

    e, w, v = eig(j, left = True, right = True)
    d = np.empty(len(e), dtype = object)
    for ii in range(len(e)):
        ei = e[ii]
        wi = w[:, ii]
        vi = v[:, ii]
        di = [(wi.T @ dd @ vi) / (wi.T @ vi) for dd in DJ]
        di = np.array(di, dtype = float)
        di = np.real(di)
        d[ii] = -di

    out = [e, d]
    return out


def reference_distance(F, eq, ref, pars, **kwargs):
    """
    A function that calculates the gradient for the distance between an
    equilibrium point eq and a reference point in state-space ref.

    Input:
    - F: the symbolic representation of the dynamical system.
    - eq: the equilibrium point to study.
    - ref: the reference point to approach.
    - pars: a dictionary of parameters.

    Output:
    - grad: a numpy array of dimension m that represents the direction in
      parameter space for the biggest DECREASE in the distance between eq and
      ref.
    """
    j = state_sens(F_s, eq, pars, **kwargs)
    grad = -np.dot(2*(eq - ref).T, j)
    grad = np.array(grad, dtype = float)
    return grad





###############################################################################
###############################################################################
###############################################################################
###############################################################################
#%% Definition of control functions

def move_parameters(direction, mag, nom_pars, par_names = parameters):
    """
    A function that changes the parameters in a given direction in parameter
    space.

    Input:
    - direction: a numpy.array of size (m, ) or (m, 1), where m is the
      dimension of ``differentiable'' parameter space. Each element of this
      array indicates in which direction each parameter should be moved. It is
      asumed that said parameters are in the same order as in the variable
      ``parameters'' defined in this same script.
    - nom_pars: a dictionary containing the nominal parameters of the system.
    - mag: int or float stating how big the step-size in the given direction
      will be.
    - par_names: A Symbolic matrix containing the ``differentiable'' parameters
      of the system. These serve as reference for the modification of the
      dictionary, as they contain the names of the parameters which correspond
      to the keys in nom_pars.
    """
    new_pars = nom_pars.copy()
    for ii in range(len(par_names)):
        par = par_names[ii]
        new_pars[par.name] += mag * direction[ii]
    return new_pars


def combination_norm(coeff, directions):
    """
    A function that calculates the norm of the linear combination of the
    vectors, with coefficients shown in coefficients.

    Input:
    - coeff: list of float coefficients for the lineal combination of the
      vectors. Has to have the same amount of elements as directions.
    - directions: list of numpy arrays that are going to be combined. Has to
      have the same length as coeff.
    """
    if len(coeff) != len(directions):
        raise
    w = sum([coeff[ii] * directions[ii] for ii in range(len(coeff))])
    return np.linalg.norm(w)


def optimum_coefficients(directions):
    """
    A function that determines the optimum direction from a set of directions.
    Based on "Multiple Gradient Descent Algorithm" by Jean-Antoine Desideri.

    Input:
    - directions: dictionary of directions for which the optimum direction
      must be calculated.
    
    Output:
    - d: np.array representing the optimum direction.
    """
    eq_const = {"type": "eq",
                "fun": lambda x: np.sum(x) - 1,
                "jac": lambda x: np.ones(len(x))
               }
    ineq_const = {"type": "ineq",
                  "fun": lambda x: np.array([x[ii] for ii in range(len(x))]),
                  "jac": lambda x: np.eye(len(x)),
                 }
    x0 = np.concatenate((np.ones(1), np.zeros(len(directions) - 1)))
    sol = minimize(combination_norm, x0, args = (directions, ),
                   method = "SLSQP", constraints = [eq_const, ineq_const],
                   tol = 1e-8)
    d = sum([sol.x[ii] * directions[ii] for ii in range(len(sol.x))])
    return d


def control(directions, magnitude, nom_pars, eqs):
    """
    A function that moves the system's parameters in a certain direction, and
    determines the new positions of the equilibria. Also checks if, poorly, if
    a bifurcation happened by comparing the amount of equilibria known
    previously with the amount of equilibria found.

    Input:
    - directions: list of np.arrays.
    - magnitude: float or int representing the stepsize.
    - nom_pars: the nominal parameters of the system.
    - eqs: the equilibria of the system with the nominal parameters.
    """
    opt_direction = optimum_coefficients(directions)
    #print(np.linalg.norm(opt_direction))
    if np.allclose(np.linalg.norm(opt_direction), 0, atol = 1e-7):
        # Quit if this is a Pareto-stationary point
        print("This point in parameter space is Pareto-stationary.")
        return {"out": (nom_pars, eqs), "quit": True, 'lost': []}

    # I do this to force the step-size to be what I need it to be
    magnitude = magnitude / np.linalg.norm(opt_direction)

    new_pars = move_parameters(opt_direction, magnitude, nom_pars)
    new_equilibria = {}
    lost = []
    for eq in eqs:
        sol = fsolve(F_n, eqs[eq], new_pars, full_output = True)
        if sol[2] == 1:
            new_equilibria[eq] = sol[0]
        else:
            print('Eq {} not found. Possible bifurcation.'.format(eq))
            lost.append(eq)
    if len(lost) != 0:
        query = input('Lost equilibria {}.\nDo you wish to continue?\t'.format(*lost))
        if query != 'y':
            return {'out': (new_pars, new_equilibria), 'quit': True, 'lost': lost}
    found = np.array(list(new_equilibria.values()))

    #if len(np.unique(found, axis = 0)) != len(eqs):
        ## Quit if a bifurcation happens
        #print("A bifurcation point was reached.")
        #return {"out": (nom_pars, eqs), "quit": True}

    return {"out": (new_pars, new_equilibria), "quit": False, 'lost': lost}

