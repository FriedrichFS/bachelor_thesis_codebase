import numpy as np
from scipy.stats import norm
from scipy.optimize import newton, brentq
from enum import Enum
import time
from typing import Dict, Optional, Tuple, Union
from src.config.enums import OptionType


class AmericanOptionPricerBase:
    """
    Base class for AMERICAN option pricing models.
    Holds common parameters and defines the interface for calculating
    price, Greeks, and timing the calculation.
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: OptionType):
        """ Initialize the option pricer. """
        # Used float64 for better precision in calculations
        self.S = np.float64(S)
        self.K = np.float64(K)
        self.T = max(np.float64(T), 1e-9)
        self.r = np.float64(r)
        self.sigma = max(np.float64(sigma), 1e-9)
        self.q = np.float64(q)
        self.option_type = option_type

        if self.option_type not in [OptionType.CALL, OptionType.PUT]:
            raise ValueError(
                "option_type must be OptionType.CALL or OptionType.PUT")

        # Parameters for finite difference Greek calculations
        self._dS_ratio = 0.001
        self._dT_years = min(0.002, self.T * 0.5)
        self._dSigma = 0.0005
        self._dr = 0.00005

    def _clone_with_params(self, **kwargs) -> 'AmericanOptionPricerBase':
        """Creates a new instance with potentially modified parameters."""
        params = {
            'S': self.S, 'K': self.K, 'T': self.T, 'r': self.r,
            'sigma': self.sigma, 'q': self.q, 'option_type': self.option_type
        }
        if hasattr(self, 'N'):
            params['N'] = self.N  # Preserve N for tree models
        params.update(kwargs)
        return type(self)(**params)

    def _price_impl(self) -> float:
        """ Internal abstract method to calculate the option price. """
        raise NotImplementedError(
            "Subclasses must implement the _price_impl() method.")

    def _greeks_impl(self) -> Dict[str, float]:
        """ Internal method to calculate option Greeks using finite differences. """
        price_mid = self._price_impl()
        dS = self.S * self._dS_ratio
        price_up_S = self._clone_with_params(S=self.S + dS)._price_impl()
        price_down_S = self._clone_with_params(S=self.S - dS)._price_impl()
        delta = (price_up_S - price_down_S) / (2 * dS)
        gamma = (price_up_S - 2 * price_mid + price_down_S) / (dS ** 2)
        dSigma = self._dSigma
        sigma_down = max(self.sigma - dSigma, 1e-9)
        price_up_sigma = self._clone_with_params(
            sigma=self.sigma + dSigma)._price_impl()
        price_down_sigma = self._clone_with_params(
            sigma=sigma_down)._price_impl()
        actual_dSigma = (self.sigma + dSigma) - sigma_down
        vega = (price_up_sigma - price_down_sigma) / \
            actual_dSigma if actual_dSigma > 1e-9 else 0.0
        vega /= 100
        dT = self._dT_years
        T_down = max(self.T - dT, 1e-9)
        price_down_T = self._clone_with_params(T=T_down)._price_impl()
        theta = -(price_mid - price_down_T) / dT / 365.25
        dr = self._dr
        r_down = self.r - dr
        price_up_r = self._clone_with_params(r=self.r + dr)._price_impl()
        price_down_r = self._clone_with_params(r=r_down)._price_impl()
        rho = (price_up_r - price_down_r) / (2 * dr)
        rho /= 100
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    def calculate_all(self) -> Dict[str, float]:
        """ Calculates price and Greeks, returning results and calculation time. """
        start_time = time.perf_counter()  # Ensures, timesteps are only set for calculation
        price = self._price_impl()
        greeks = self._greeks_impl()
        end_time = time.perf_counter()
        calc_time = end_time - start_time  # Timedelta only includes pricing and Greeks
        results = {"price": price}
        results.update(greeks)
        results["calc_time_sec"] = calc_time
        return results


class CRRBinomialAmericanPricer(AmericanOptionPricerBase):
    """ Prices American options using the CRR binomial tree model. """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: OptionType, N: int):
        """Initialize CRR Pricer. N is number of steps."""
        super().__init__(S, K, T, r, sigma, q, option_type)
        self.N = int(N)
        if self.N < 1:
            raise ValueError("N must be >= 1.")

    def _price_impl(self) -> float:
        """Calculates the American option price using the CRR binomial tree."""
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        exp_rqdt = np.exp((self.r - self.q) * dt)
        p_num = exp_rqdt - d
        p_den = u - d
        if abs(p_den) < 1e-15:
            p = 0.5
        else:
            p = p_num / p_den
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1.0 - epsilon)
        one_minus_p = 1.0 - p
        option_values = np.zeros(self.N + 1, dtype=np.float64)
        prices_at_exp = self.S * \
            (d ** np.arange(self.N, -1, -1)) * \
            (u ** np.arange(0, self.N + 1, 1))
        if self.option_type == OptionType.CALL:
            option_values = np.maximum(0.0, prices_at_exp - self.K)
        else:
            option_values = np.maximum(0.0, self.K - prices_at_exp)
        discount_factor = np.exp(-self.r * dt)
        for i in range(self.N - 1, -1, -1):
            expected_value = discount_factor * \
                (p * option_values[1:i+2] + one_minus_p * option_values[:i+1])
            current_prices = self.S * \
                (d ** np.arange(i, -1, -1)) * (u ** np.arange(0, i + 1, 1))
            if self.option_type == OptionType.CALL:
                intrinsic_value = np.maximum(0.0, current_prices - self.K)
            else:
                intrinsic_value = np.maximum(0.0, self.K - current_prices)
            option_values[:i+1] = np.maximum(expected_value, intrinsic_value)
        return option_values[0]


class LeisenReimerAmericanPricer(AmericanOptionPricerBase):
    """ Prices American options using the Leisen-Reimer binomial tree model. """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: OptionType, N: int):
        """ Initialize LR Pricer. N steps (will ensure N is odd)."""
        super().__init__(S, K, T, r, sigma, q, option_type)
        self.N = int(N) if N % 2 != 0 else int(N) + 1
        if self.N < 1:
            raise ValueError("N must be >= 1.")

    def _peizer_pratt_inversion(self, z, n):
        """ Peizer-Pratt inversion method 2 with stability checks """
        n_float = np.float64(n)
        epsilon = 1e-12
        if n_float < 1:
            return 0.5
        den_term_safe = n_float + 1/3 + 0.1 / (n_float + 1) + epsilon
        if abs(den_term_safe) < epsilon:
            return 0.5
        term1_sq = (z / den_term_safe)**2
        term2 = n_float + 1/6
        exp_arg = np.clip(-term1_sq * term2, -700, 700)
        inner_sqrt = np.maximum(1.0 - np.exp(exp_arg), 0.0)
        return 0.5 + np.sign(z) * 0.5 * np.sqrt(inner_sqrt)

    def _price_impl(self) -> float:
        """Calculates the American option price using the LR binomial tree."""
        dt = self.T / self.N
        discount_factor = np.exp(-self.r * dt)
        vol_sqrt_T = self.sigma * np.sqrt(self.T)
        if vol_sqrt_T < 1e-9:
            vol_sqrt_T = 1e-9
        d1 = (np.log(self.S / self.K) + (self.r - self.q +
              0.5 * self.sigma**2) * self.T) / vol_sqrt_T
        d2 = d1 - vol_sqrt_T
        p_d1_inverted = self._peizer_pratt_inversion(d1, self.N)
        p_d2_inverted = self._peizer_pratt_inversion(d2, self.N)
        epsilon = 1e-9
        p_d1_inverted = np.clip(p_d1_inverted, epsilon, 1.0 - epsilon)
        p_d2_inverted = np.clip(p_d2_inverted, epsilon, 1.0 - epsilon)
        drift_adj = np.exp((self.r - self.q) * dt)
        u = drift_adj * (p_d1_inverted / p_d2_inverted)
        p_d2_inv_minus_1 = 1.0 - p_d2_inverted
        if abs(p_d2_inv_minus_1) < epsilon:
            d = drift_adj / u if abs(u) > epsilon else drift_adj
        else:
            d = (drift_adj - p_d2_inverted * u) / p_d2_inv_minus_1
        p_exp = p_d2_inverted
        one_minus_p_exp = 1.0 - p_exp
        option_values = np.zeros(self.N + 1, dtype=np.float64)
        prices_at_exp = self.S * \
            (d ** np.arange(self.N, -1, -1)) * \
            (u ** np.arange(0, self.N + 1, 1))
        if self.option_type == OptionType.CALL:
            option_values = np.maximum(0.0, prices_at_exp - self.K)
        else:
            option_values = np.maximum(0.0, self.K - prices_at_exp)
        for i in range(self.N - 1, -1, -1):
            expected_value = discount_factor * \
                (p_exp * option_values[1:i+2] +
                 one_minus_p_exp * option_values[:i+1])
            current_prices = self.S * \
                (d ** np.arange(i, -1, -1)) * (u ** np.arange(0, i + 1, 1))
            if self.option_type == OptionType.CALL:
                intrinsic_value = np.maximum(0.0, current_prices - self.K)
            else:
                intrinsic_value = np.maximum(0.0, self.K - current_prices)
            option_values[:i+1] = np.maximum(expected_value, intrinsic_value)
        return option_values[0]


class BjerksundStensland2002Pricer(AmericanOptionPricerBase):
    """
    Calculates American option prices using the Bjerksund-Stensland (2002) approximation.
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: OptionType):
        """ Initialize the Bjerksund-Stensland 2002 Pricer. """
        super().__init__(S, K, T, r, sigma, q, option_type)

    def _N(self, x):
        """Standard normal cumulative distribution function."""
        return norm.cdf(x)

    def _bs_price(self, S, K, T, r, q, sigma, opt_type):
        """Standard Black-Scholes price for European options."""
        tol = 1e-9
        if T < tol:
            return max(0.0, S - K) if opt_type == OptionType.CALL else max(0.0, K - S)
        sigma = max(sigma, tol)
        vol_sqrt_T = sigma * np.sqrt(T)
        if vol_sqrt_T < tol:
            return max(0.0, S - K) if opt_type == OptionType.CALL else max(0.0, K - S)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_T
        d2 = d1 - vol_sqrt_T
        if opt_type == OptionType.CALL:
            price = S * np.exp(-q * T) * self._N(d1) - K * \
                np.exp(-r * T) * self._N(d2)
        else:  # PUT
            price = K * np.exp(-r * T) * self._N(-d2) - S * \
                np.exp(-q * T) * self._N(-d1)
        return price

    def _phi(self, S, T, gamma, H, I, r, q, sigma):
        """ Phi function as used in BS 2002 context"""
        T = max(T, 1e-9)
        sigma = max(sigma, 1e-9)
        vol_sqrt_T = sigma * np.sqrt(T)
        lambda_ = (-r + gamma * q + 0.5 * gamma * (gamma - 1) * sigma**2) * T
        d1_num = np.log(S / H) + (q + (gamma - 0.5) * sigma**2) * T
        d1 = d1_num / vol_sqrt_T
        d1_shifted = d1 - (2 * np.log(I/S)) / vol_sqrt_T
        kappa_pow = 2 * (r - q) / sigma**2 + 2*gamma - 1
        lambda_ = np.clip(lambda_, -700, 700)
        term1 = np.exp(lambda_) * (S**gamma)
        term2 = self._N(d1)
        I_S_ratio = I/S if S > 1e-12 else np.inf
        if I_S_ratio <= 0:
            term3_mult = 0.0
        else:
            base_clamp = max(I_S_ratio, 1e-10)
            try:
                term3_mult = base_clamp ** kappa_pow
            except OverflowError:
                term3_mult = np.inf
        term3 = term3_mult * self._N(d1_shifted)
        if np.isinf(term3_mult) and abs(self._N(d1_shifted)) < 1e-12:
            term3 = 0.0
        elif np.isinf(term3):
            term3 = 0.0
        return term1 * (term2 - term3)

    def _price_impl(self) -> float:
        """ Calculates the American option price using BS 2002 approximation. """
        tol = 1e-9
        if self.T < tol:
            return max(0.0, self.S - self.K) if self.option_type == OptionType.CALL else max(0.0, self.K - self.S)

        S, K, T, r, q, sigma = self.S, self.K, self.T, self.r, self.q, self.sigma
        sigma_sq = sigma**2

        if self.option_type == OptionType.CALL:
            # Calculation for CALL Option
            beta_num_sqrt = np.sqrt(((r - q)/sigma_sq - 0.5)**2 + 2*r/sigma_sq)
            beta = (0.5 - (r - q)/sigma_sq) + beta_num_sqrt
            if q == 0:
                B0 = K
            elif q > 0:
                B0 = max(K, (r / q) * K)
            else:
                B0 = K
            Binf = (beta / (beta - 1)) * K if abs(beta - 1) > tol else K
            h_T_num_sqrt = np.sqrt(T)
            h_T_den = Binf - B0
            if abs(h_T_den) < tol:
                return self._bs_price(S, K, T, r, q, sigma, OptionType.CALL)
            h_T_exponent = np.clip(-((r - q) * T + 2 *
                                   sigma * h_T_num_sqrt * K / h_T_den), -700, 700)

            h_T_exponent_check = - \
                ((r - q) * T + 2 * sigma * h_T_num_sqrt * K / h_T_den)
            h_T_exponent_check = np.clip(h_T_exponent_check, -700, 700)
            I = B0 + (Binf - B0) * (1 - np.exp(h_T_exponent_check))

            if S >= I:
                return max(0.0, S - K)
            else:
                eur_price = self._bs_price(
                    S, K, T, r, q, sigma, OptionType.CALL)
                phi_val = self._phi(S, T, beta, I, I, r, q, sigma)
                return max(eur_price + phi_val, eur_price, S - K)

        else:
            # Calculation for PUT Option
            gamma_put_num_sqrt = np.sqrt(
                ((r - q)/sigma_sq - 0.5)**2 + 2*r/sigma_sq)
            gamma_put = (0.5 - (r - q)/sigma_sq) - gamma_put_num_sqrt
            if q == 0:
                B0_put = K
            elif q > 0:
                B0_put = min(K, (r / q) * K)
            else:
                B0_put = K
            Binf_put = 0.0
            h_T_num_sqrt = np.sqrt(T)
            h_T_den = B0_put - Binf_put
            if abs(h_T_den) < tol:
                return self._bs_price(S, K, T, r, q, sigma, OptionType.PUT)
            h_T_exponent_put = np.clip(
                ((r - q) * T - 2 * sigma * h_T_num_sqrt * K / h_T_den), -700, 700)
            I_put = Binf_put + (B0_put - Binf_put) * \
                (1 - np.exp(h_T_exponent_put))

            if S <= I_put:
                return max(0.0, K - S)
            else:
                eur_price = self._bs_price(
                    S, K, T, r, q, sigma, OptionType.PUT)
                phi_val_put = self._phi(
                    S, T, gamma_put, I_put, I_put, r, q, sigma)
                return max(eur_price + phi_val_put, eur_price, K - S)
