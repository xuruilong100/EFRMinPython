import numpy as np
from arch.typing import FloatOrArray, Union, Tuple, ArrayLike1D, RNGType, Optional
from arch.univariate.volatility import VolatilityProcess, VarianceForecast
from arch.utility.array import AbstractDocStringInheritor, NDArray, List, Sequence, ensure1d
from arch.utility.exceptions import initial_value_warning, InitialValueWarning

from archEx.recursions_python import ngarch11_recursion, fixedngarch11_recursion


class NGARCH11(VolatilityProcess, metaclass=AbstractDocStringInheritor):

    def __init__(
            self,
            starting_values: NDArray) -> None:
        super().__init__()
        self.p = 1
        self.q = 1
        self.power = 2.0
        self.num_params = 4
        self._name = self._generate_name()
        self._starting_values = starting_values

    def __str__(self) -> str:
        return 'NGARCH(1, 1)'

    def variance_bounds(
            self,
            resids: NDArray,
            power: float = 2.0) -> NDArray:
        return super().variance_bounds(resids, self.power)

    def _generate_name(self) -> str:
        return "NGARCH"

    def bounds(
            self,
            resids: NDArray) -> List[Tuple[float, float]]:
        v = np.mean(abs(resids) ** self.power)

        bounds = [(0.0, 10.0 * v)]  # for omega
        bounds.extend([(0.0, 1.0)])  # for alpha
        bounds.extend([(0.0, 1.0)])  # for beta
        bounds.extend([(-2.0, 2.0)])  # for theta

        return bounds

    def constraints(self) -> Tuple[NDArray, NDArray]:
        # omega > 0
        # alpha > 0
        # beta > 0
        # alpha + beta < 1
        # theta > -2
        # theta < 2

        a = np.zeros((6, self.num_params))
        for i in range(3):
            a[i, i] = 1.0

        a[3, 1] = -1.0
        a[3, 2] = -1.0
        a[4, 3] = 1.0
        a[5, 3] = -1.0

        b = np.zeros(6)
        b[3] = -1.0
        b[4] = -1.0
        b[5] = -2.0

        return a, b

    def compute_variance(
            self,
            parameters: NDArray,
            resids: NDArray,
            sigma2: NDArray,
            backcast: Union[float, NDArray],
            var_bounds: NDArray) -> NDArray:
        nobs = resids.shape[0]

        ngarch11_recursion(
            parameters, resids, sigma2, nobs, backcast, var_bounds)

        return sigma2

    def backcast_transform(
            self,
            backcast: FloatOrArray) -> FloatOrArray:
        backcast = super().backcast_transform(backcast)

        return backcast

    def backcast(
            self,
            resids: NDArray) -> float:
        tau = min(75, resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        backcast = np.sum((resids[:tau] ** self.power) * w)

        return float(backcast)

    def simulate(
            self,
            parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
            nobs: int,
            rng: RNGType,
            burn: int = 500,
            initial_value: Optional[float] = None) -> Tuple[NDArray, NDArray]:
        parameters = ensure1d(
            parameters, "parameters", False)
        errors = rng(nobs + burn)

        if initial_value is None:
            scale = np.ones_like(parameters)

            persistence = np.sum(parameters[1:] * scale[1:])
            if (1.0 - persistence) > 0:
                initial_value = parameters[0] / (1.0 - persistence)
            else:
                from warnings import warn
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)

        max_lag = 1
        sigma2[:max_lag] = initial_value
        data[:max_lag] = np.sqrt(sigma2[:max_lag]) * errors[:max_lag]

        omega = parameters[0]
        alpha = parameters[1]
        beta = parameters[2]
        theta = parameters[3]

        sigma2[max_lag] = omega
        data[max_lag] = errors[max_lag] * np.sqrt(sigma2[max_lag])

        for t in range(max_lag + 1, nobs + burn):
            loc = t - 1
            sigma2[t] = \
                omega + \
                alpha * (data[loc] - theta * np.sqrt(sigma2[loc])) ** 2.0 + \
                beta * sigma2[loc]
            data[t] = errors[t] * np.sqrt(sigma2[t])

        return data[burn:], sigma2[burn:]

    def starting_values(
            self,
            resids: NDArray) -> NDArray:

        return self._starting_values

    def parameter_names(self) -> List[str]:
        names = ['omega', 'alpha', 'beta', 'theta']
        return names

    def _check_forecasting_method(
            self,
            method: str,
            horizon: int) -> None:
        if horizon == 1:
            return

        return

    def _analytic_forecast(
            self,
            parameters: NDArray,
            resids: NDArray,
            backcast: float,
            var_bounds: NDArray,
            start: int,
            horizon: int) -> VarianceForecast:
        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon)

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts)

    def _simulate_paths(
            self,
            m: int,
            parameters: NDArray,
            horizon: int,
            std_shocks: NDArray,
            scaled_forecast_paths: NDArray,
            scaled_shock: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        omega = parameters[0]
        alpha = parameters[1]
        beta = parameters[2]
        theta = parameters[3]

        shock = np.full_like(scaled_forecast_paths, np.nan)

        for h in range(horizon):
            loc = h + m - 1

            scaled_forecast_paths[:, h + m] = \
                omega + \
                alpha * (scaled_shock[:, loc] - theta * np.sqrt(scaled_forecast_paths[:, loc])) ** 2.0 + \
                beta * scaled_forecast_paths[:, loc]

            shock[:, h + m] = std_shocks[:, h] * np.sqrt(scaled_forecast_paths[:, h + m])
            scaled_shock[:, h + m] = shock[:, h + m]

        forecast_paths = scaled_forecast_paths[:, m:]

        return np.mean(forecast_paths, 0), forecast_paths, shock[:, m:]

    def _simulation_forecast(
            self,
            parameters: NDArray,
            resids: NDArray,
            backcast: Union[float, NDArray],
            var_bounds: NDArray,
            start: int,
            horizon: int,
            simulations: int,
            rng: RNGType) -> VarianceForecast:
        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon)

        t = resids.shape[0]
        paths = np.full((t, simulations, horizon), np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        m = 1  # np.max([self.p, self.q])
        scaled_forecast_paths = np.zeros((simulations, m + horizon))
        scaled_shock = np.zeros((simulations, m + horizon))

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            if i - m < 0:
                scaled_forecast_paths[:, :m] = backcast
                scaled_shock[:, :m] = backcast

                count = i + 1
                scaled_forecast_paths[:, m - count: m] = sigma2[:count]
                scaled_shock[:, m - count: m] = resids[:count]
            else:
                scaled_forecast_paths[:, :m] = sigma2[i - m + 1: i + 1]
                scaled_shock[:, :m] = resids[i - m + 1: i + 1]

            f, p, s = self._simulate_paths(
                m,
                parameters,
                horizon,
                std_shocks,
                scaled_forecast_paths,
                scaled_shock)
            forecasts[i, :], paths[i], shocks[i] = f, p, s

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts, paths, shocks)

    def is_valid(self, alpha, beta, theta):
        return alpha * (1 + theta ** 2) + beta < 1


class FixedNGARCH11(VolatilityProcess, metaclass=AbstractDocStringInheritor):

    def __init__(
            self,
            theta: float,
            starting_values: NDArray) -> None:
        super().__init__()
        self.p = 1
        self.q = 1
        self.power = 2.0
        self.num_params = 3
        self.theta = theta
        self._name = self._generate_name()
        self._starting_values = starting_values

    def __str__(self) -> str:
        return 'FixedNGARCH(1, 1)'

    def variance_bounds(
            self,
            resids: NDArray,
            power: float = 2.0) -> NDArray:
        return super().variance_bounds(resids, self.power)

    def _generate_name(self) -> str:
        return "FixedNGARCH"

    def bounds(
            self,
            resids: NDArray) -> List[Tuple[float, float]]:
        v = np.mean(abs(resids) ** self.power)

        bounds = [(0.0, 10.0 * v)]  # for omega
        bounds.extend([(0.0, 1.0)])  # for alpha
        bounds.extend([(0.0, 1.0)])  # for beta

        return bounds

    def constraints(self) -> Tuple[NDArray, NDArray]:
        # omega > 0
        # alpha > 0
        # beta > 0
        # alpha * (1 + theta**2) + beta < 1
        a = np.zeros((4, self.num_params))
        for i in range(3):
            a[i, i] = 1.0

        a[3, 1] = -(1.0 + self.theta ** 2)
        a[3, 2] = -1.0

        b = np.zeros(4)
        b[3] = -1.0

        return a, b

    def compute_variance(
            self,
            parameters: NDArray,
            resids: NDArray,
            sigma2: NDArray,
            backcast: Union[float, NDArray],
            var_bounds: NDArray) -> NDArray:
        nobs = resids.shape[0]

        fixedngarch11_recursion(
            parameters, self.theta, resids, sigma2, nobs, backcast, var_bounds)

        return sigma2

    def backcast_transform(
            self,
            backcast: FloatOrArray) -> FloatOrArray:
        backcast = super().backcast_transform(backcast)

        return backcast

    def backcast(
            self,
            resids: NDArray) -> float:
        tau = min(75, resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** self.power) * w)

        return float(backcast)

    def simulate(
            self,
            parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
            nobs: int,
            rng: RNGType,
            burn: int = 500,
            initial_value: Optional[float] = None) -> Tuple[NDArray, NDArray]:
        parameters = ensure1d(
            parameters, "parameters", False)
        errors = rng(nobs + burn)

        if initial_value is None:
            scale = np.ones_like(parameters)

            persistence = np.sum(parameters[1:] * scale[1:])
            if (1.0 - persistence) > 0:
                initial_value = parameters[0] / (1.0 - persistence)
            else:
                from warnings import warn
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)

        max_lag = 1
        sigma2[:max_lag] = initial_value
        data[:max_lag] = np.sqrt(sigma2[:max_lag]) * errors[:max_lag]

        omega = parameters[0]
        alpha = parameters[1]
        beta = parameters[2]

        sigma2[max_lag] = omega
        data[max_lag] = errors[max_lag] * np.sqrt(sigma2[max_lag])

        for t in range(max_lag + 1, nobs + burn):
            loc = t - 1
            sigma2[t] = \
                omega + \
                alpha * (data[loc] - self.theta * np.sqrt(sigma2[loc])) ** 2.0 + \
                beta * sigma2[loc]
            data[t] = errors[t] * np.sqrt(sigma2[t])

        return data[burn:], sigma2[burn:]

    def starting_values(
            self,
            resids: NDArray) -> NDArray:

        return self._starting_values

    def parameter_names(self) -> List[str]:
        names = ['omega', 'alpha', 'beta']
        return names

    def _check_forecasting_method(
            self,
            method: str,
            horizon: int) -> None:
        if horizon == 1:
            return

        return

    def _analytic_forecast(
            self,
            parameters: NDArray,
            resids: NDArray,
            backcast: float,
            var_bounds: NDArray,
            start: int,
            horizon: int) -> VarianceForecast:

        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon)

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts)

    def _simulate_paths(
            self,
            m: int,
            parameters: NDArray,
            horizon: int,
            std_shocks: NDArray,
            scaled_forecast_paths: NDArray,
            scaled_shock: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        omega = parameters[0]
        alpha = parameters[1]
        beta = parameters[2]

        shock = np.full_like(scaled_forecast_paths, np.nan)

        for h in range(horizon):
            loc = h + m - 1

            scaled_forecast_paths[:, h + m] = \
                omega + \
                alpha * (scaled_shock[:, loc] - self.theta * np.sqrt(scaled_forecast_paths[:, loc])) ** 2.0 + \
                beta * scaled_forecast_paths[:, loc]

            shock[:, h + m] = std_shocks[:, h] * np.sqrt(scaled_forecast_paths[:, h + m])
            scaled_shock[:, h + m] = shock[:, h + m]

        forecast_paths = scaled_forecast_paths[:, m:]

        return np.mean(forecast_paths, 0), forecast_paths, shock[:, m:]

    def _simulation_forecast(
            self,
            parameters: NDArray,
            resids: NDArray,
            backcast: Union[float, NDArray],
            var_bounds: NDArray,
            start: int,
            horizon: int,
            simulations: int,
            rng: RNGType) -> VarianceForecast:

        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon)

        t = resids.shape[0]
        paths = np.full((t, simulations, horizon), np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        m = 1  # np.max([self.p, self.q])
        scaled_forecast_paths = np.zeros((simulations, m + horizon))
        scaled_shock = np.zeros((simulations, m + horizon))

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            if i - m < 0:
                scaled_forecast_paths[:, :m] = backcast
                scaled_shock[:, :m] = backcast

                count = i + 1
                scaled_forecast_paths[:, m - count: m] = sigma2[:count]
                scaled_shock[:, m - count: m] = resids[:count]
            else:
                scaled_forecast_paths[:, :m] = sigma2[i - m + 1: i + 1]
                scaled_shock[:, :m] = resids[i - m + 1: i + 1]

            f, p, s = self._simulate_paths(
                m,
                parameters,
                horizon,
                std_shocks,
                scaled_forecast_paths,
                scaled_shock)
            forecasts[i, :], paths[i], shocks[i] = f, p, s

        forecasts[:start] = np.nan

        return VarianceForecast(forecasts, paths, shocks)
