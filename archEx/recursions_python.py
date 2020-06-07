import numpy as np
from arch.compat.numba import jit
from arch.typing import NDArray
from arch.univariate.recursions_python import bounds_check


def ngarch11_recursion_python(
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        nobs: int,
        var_bounds: NDArray) -> NDArray:
    """
    Compute variance recursion for NGARCH(1,1) and related models
    Parameters
    ----------
    parameters : ndarray
        Model parameters
    resids : ndarray
        Value of residuals.
    sigma2 : ndarray
        Conditional variances with same shape as resids
    nobs : int
        Length of resids
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        transformed variances for each time period
    """

    omega = parameters[0]
    alpha = parameters[1]
    beta = parameters[2]
    theta = parameters[3]

    sigma2[0] = omega

    for t in range(1, nobs):
        sigma2[t] = omega + \
                    alpha * (resids[t - 1] - theta * np.sqrt(sigma2[t - 1])) ** 2.0 + \
                    beta * sigma2[t - 1]
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


ngarch11_recursion = jit(ngarch11_recursion_python, nopython=True)


def fixedngarch11_recursion_python(
        parameters: NDArray,
        theta: float,
        resids: NDArray,
        sigma2: NDArray,
        nobs: int,
        var_bounds: NDArray) -> NDArray:
    """
    Compute variance recursion for FixedNGARCH(1,1) and related models
    Parameters
    ----------
    parameters : ndarray
        Model parameters
    theta : float
        Value of fixed theta in model.
    resids : ndarray
        Value of residuals.
    sigma2 : ndarray
        Conditional variances with same shape as resids
    nobs : int
        Length of resids
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        transformed variances for each time period
    """

    omega = parameters[0]
    alpha = parameters[1]
    beta = parameters[2]

    sigma2[0] = omega

    for t in range(1, nobs):
        sigma2[t] = omega + \
                    alpha * (resids[t - 1] - theta * np.sqrt(sigma2[t - 1])) ** 2.0 + \
                    beta * sigma2[t - 1]
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


fixedngarch11_recursion = jit(fixedngarch11_recursion_python, nopython=True)


def garchx_recursion_python(
        parameters: NDArray,
        fresids: NDArray,
        sresids: NDArray,
        sigma2: NDArray,
        exog: NDArray,
        p: int,
        o: int,
        q: int,
        nobs: int,
        backcast: float,
        var_bounds: NDArray) -> NDArray:
    """
    Compute variance recursion for GARCH and related models
    Parameters
    ----------
    parameters : ndarray
        Model parameters
    fresids : ndarray
        Absolute value of residuals raised to the power in the model.  For
        example, in a standard GARCH model, the power is 2.0.
    sresids : ndarray
        Variable containing the sign of the residuals (-1.0, 0.0, 1.0)
    sigma2 : ndarray
        Conditional variances with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        transformed variances for each time period
    """

    for t in range(nobs):
        loc = 0
        sigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * fresids[t - 1 - j]
            loc += 1
        for j in range(o):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * 0.5 * backcast
            else:
                sigma2[t] += (
                        parameters[loc] * fresids[t - 1 - j] * (sresids[t - 1 - j] < 0)
                )
            loc += 1
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * sigma2[t - 1 - j]
            loc += 1

        if t > 0:
            sigma2[t] += parameters[loc] * exog[t - 1]

        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


garchx_recursion = jit(garchx_recursion_python, nopython=True)
