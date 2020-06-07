import numpy as np
from arch.typing import Union, Tuple, ArrayLike1D, RNGType, Optional
from arch.univariate.volatility import VarianceForecast, GARCH
from arch.utility.array import NDArray, List, Sequence

from archEx.recursions_python import garchx_recursion


class GARCHX(GARCH):

    def __init__(
            self,
            exog: NDArray,
            exog_bound: float = 100.0,
            p: int = 1,
            o: int = 0,
            q: int = 1,
            power: float = 2.0) -> None:
        super().__init__(p, o, q, power)
        
        self.exog = exog  # explanatory variable
        self.exog_bound = exog_bound  # absolute bound for explanatory variable
        self.num_params += 1  # add 1 for exog(kappa)
        self._name = self._generate_name()

    def _generate_name(self) -> str:
        name = super()._generate_name()
        new_name = name.replace('ARCH', 'ARCHX')

        return new_name

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        bounds = super().bounds(resids)
        bounds.extend([(-self.exog_bound, self.exog_bound)])  # for exog

        return bounds

    def constraints(self) -> Tuple[NDArray, NDArray]:
        p, o, q = self.p, self.o, self.q
        k_arch = p + o + q
        # omage > 0
        # alpha[i] >0
        # alpha[i] + gamma[i] > 0 for i<=p, otherwise gamma[i]>0
        # beta[i] >0
        # sum(alpha) + 0.5 sum(gamma) + sum(beta) < 1
        # kappa > -exog_bound
        # kappa < exog_bound
        a = np.zeros((k_arch + 4, k_arch + 2))
        for i in range(k_arch + 1):
            a[i, i] = 1.0
        for i in range(o):
            if i < p:
                a[i + p + 1, i + 1] = 1.0

        a[k_arch + 1, 1:] = -1.0
        a[k_arch + 1, p + 1: p + o + 1] = -0.5
        a[k_arch + 2, k_arch + 1] = 1.0
        a[k_arch + 3, k_arch + 1] = -1.0
        b = np.zeros(k_arch + 4)
        b[k_arch + 1] = -1.0
        b[k_arch + 2] = -self.exog_bound
        b[k_arch + 3] = -self.exog_bound
        return a, b

    def compute_variance(
            self,
            parameters: NDArray,
            resids: NDArray,
            sigma2: NDArray,
            backcast: Union[float, NDArray],
            var_bounds: NDArray) -> NDArray:
        # fresids is abs(resids) ** power
        # sresids is I(resids<0)
        power = self.power
        fresids = np.abs(resids) ** power
        sresids = np.sign(resids)

        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]

        garchx_recursion(
            parameters, fresids, sresids, sigma2, self.exog,
            p, o, q, nobs, backcast, var_bounds)
        inv_power = 2.0 / power
        sigma2 **= inv_power

        return sigma2

    def simulate(
            self,
            parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
            nobs: int,
            rng: RNGType,
            burn: int = 500,
            initial_value: Optional[float] = None) -> Tuple[NDArray, NDArray]:

        raise ValueError(
            "Simulation is not available for GARCHX")

    def starting_values(self, resids: NDArray) -> NDArray:

        svs = super().starting_values(resids)
        svs = np.hstack((svs, np.array([0.0])))
        return svs

    def parameter_names(self) -> List[str]:

        pn = super().parameter_names()
        pn.append('kappa')
        return pn

    def _check_forecasting_method(
            self, method: str, horizon: int) -> None:

        if horizon == 1:
            return
        else:
            raise ValueError(
                "Forecasts not available for horizon > 1")

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

        if horizon == 1:
            forecasts[:start] = np.nan
            return VarianceForecast(forecasts)
        else:
            raise ValueError(
                "Analytic forecasts not available for horizon > 1")

    def _simulate_paths(
            self,
            m: int,
            parameters: NDArray,
            horizon: int,
            std_shocks: NDArray,
            scaled_forecast_paths: NDArray,
            scaled_shock: NDArray,
            asym_scaled_shock: NDArray) -> Tuple[NDArray, NDArray, NDArray]:

        raise ValueError("Simulation is not available")

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

        raise ValueError("Simulation forecasts not available")
