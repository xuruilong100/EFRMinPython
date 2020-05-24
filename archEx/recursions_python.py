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
