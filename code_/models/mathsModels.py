import numpy as np
from scipy.stats import norm

def GARCH_VM(returns, omega, alpha, beta):
    """
    GARCH(1, 1) Model for Volatility Modeling.

    Parameters:
    - returns: np.array, Array of asset returns.
    - omega: float, Long-run variance coefficient.
    - alpha: float, ARCH coefficient (impact of lagged squared returns).
    - beta: float, GARCH coefficient (impact of lagged variance).

    Returns:
    - variances: np.array, Estimated conditional variances.
    """
    n = len(returns)
    variances = np.zeros(n)
    variances[0] = np.var(returns)

    for t in range(1, n):
        variances[t] = omega + alpha * returns[t - 1]**2 + beta * variances[t - 1]

    return variances





def blackScholes_VM(S, K, T, r, sigma, option_type):
    """
    Black-Scholes Model for Option Pricing.

    Parameters:
    - S: float, Current stock price.
    - K: float, Strike price.
    - T: float, Time to maturity (in years).
    - r: float, Risk-free interest rate.
    - sigma: float, Volatility of the stock.
    - option_type: str, "call" or "put".

    Returns:
    - option_price: float, Price of the option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "P":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price




def mertonJD_VM(S, K, T, r, sigma, lamb, muJ, sigmaJ, option_type="call", n_terms=50):
    """
    Merton Jump-Diffusion Model for Option Pricing.

    Parameters:
    - S: float, Current stock price.
    - K: float, Strike price.
    - T: float, Time to maturity (in years).
    - r: float, Risk-free interest rate.
    - sigma: float, Volatility of the stock.
    - lamb: float, Jump intensity (average number of jumps per year).
    - muJ: float, Mean jump size.
    - sigmaJ: float, Volatility of jumps.
    - option_type: str, "call" or "put".
    - n_terms: int, Number of terms in the summation.

    Returns:
    - option_price: float, Price of the option.
    """
    option_price = 0
    for k in range(n_terms):
        r_k = r - lamb * (np.exp(muJ + 0.5 * sigmaJ**2) - 1) + k * np.log(1 + muJ)
        sigma_k = np.sqrt(sigma**2 + (k * sigmaJ**2) / T)
        P_k = (np.exp(-lamb * T) * (lamb * T)**k) / np.math.factorial(k)
        
        option_price += P_k * blackScholes_VM(S, K, T, r_k, sigma_k, option_type)
    
    return option_price




def SMA_VM(prices, window):
    """
    Simple Moving Average (SMA).

    Parameters:
    - prices: np.array, Array of prices.
    - window: int, Window size for the moving average.

    Returns:
    - sma: np.array, Array of SMA values.
    """
    sma = np.convolve(prices, np.ones(window) / window, mode='valid')
    return sma



def EWMA_VM(returns, lambda_):
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.

    Parameters:
    - returns: np.array, Array of asset returns.
    - lambda_: float, Smoothing factor (0 < lambda_ < 1).

    Returns:
    - ewma: np.array, EWMA volatility.
    """
    n = len(returns)
    ewma = np.zeros(n)
    ewma[0] = np.var(returns)

    for t in range(1, n):
        ewma[t] = lambda_ * ewma[t - 1] + (1 - lambda_) * returns[t - 1]**2

    return np.sqrt(ewma)



def EGARCH_VM(returns, omega, alpha, beta, gamma):
    """
    Calculate EGARCH (Exponential GARCH) volatility.

    Parameters:
    - returns: np.array, Array of asset returns.
    - omega: float, Constant term.
    - alpha: float, Impact of past shocks (ARCH effect).
    - beta: float, Persistence of past volatility (GARCH effect).
    - gamma: float, Asymmetry coefficient.

    Returns:
    - egarch: np.array, EGARCH volatility.
    """
    n = len(returns)
    log_vol = np.zeros(n)
    log_vol[0] = np.log(np.var(returns))

    for t in range(1, n):
        z_t = returns[t - 1] / np.sqrt(np.exp(log_vol[t - 1])) if log_vol[t - 1] > 0 else 0
        log_vol[t] = omega + beta * log_vol[t - 1] + alpha * (np.abs(z_t) - np.sqrt(2 / np.pi)) + gamma * z_t

    return np.sqrt(np.exp(log_vol))



def RogersSatchell_VM(high, low, open_, close):
    """
    Calculate Rogers-Satchell volatility.

    Parameters:
    - high: np.array, Array of high prices.
    - low: np.array, Array of low prices.
    - open_: np.array, Array of open prices.
    - close: np.array, Array of close prices.

    Returns:
    - rs_vol: np.array, Rogers-Satchell volatility.
    """
    rs_vol = np.sqrt(
        (np.log(high / close) * np.log(high / open_)) +
        (np.log(low / close) * np.log(low / open_))
    )
    return rs_vol



def GarmanKlass_VM(open_prices, close_prices, high_prices, low_prices):
    """
    Calculate Garman-Klass volatility.
    
    Parameters:
        open_prices (array): Opening prices.
        close_prices (array): Closing prices.
        high_prices (array): High prices.
        low_prices (array): Low prices.
        
    Returns:
        float: Garman-Klass volatility.
    """
    
    log_high_low = np.log(high_prices / low_prices)
    log_close_open = np.log(close_prices / open_prices)
    
    return np.sqrt((0.5 * log_high_low ** 2) - ((2 * np.log(2) - 1) * log_close_open ** 2))
    
    
    
    
def parkinson_VM(high_prices, low_prices):
    """
    Calculate Parkinson volatility.
    
    Parameters:
        high_prices (array): High prices.
        low_prices (array): Low prices.
        
    Returns:
        float: Parkinson volatility.
    """
    log_high_low = np.log(high_prices / low_prices)
    
    return np.sqrt((1 / (4 * np.log(2))) * log_high_low ** 2)



def ARCH_VM(returns, lags=5):
    """
    Calculate ARCH volatility using past squared returns.
    
    Parameters:
        returns (array): Returns series.
        lags (int): Number of lagged returns to include.
        
    Returns:
        array: Estimated volatility series, padded with NaN for missing values.
    """
    squared_returns = returns ** 2
    volatilities = []

    for i in range(lags, len(returns)):
        past_squared = squared_returns[i - lags:i]
        volatilities.append(np.sqrt(np.mean(past_squared)))

    # Pad with NaN for the first `lags` rows
    return np.array([np.nan] * lags + volatilities)




def Yang_Zhang_VM(open_prices, close_prices, high_prices, low_prices):
    """
    Calculate Yang-Zhang volatility row by row, with padding to align output length.
    
    Parameters:
        open_prices (array): Opening prices.
        close_prices (array): Closing prices.
        high_prices (array): High prices.
        low_prices (array): Low prices.
        
    Returns:
        array: Yang-Zhang volatility values with NaN for the first row.
    """
    # Compute lagged log returns for open and close prices
    log_open_close = np.log(open_prices[1:] / close_prices[:-1])
    
    # Align other arrays to match the length of log_open_close
    trimmed_open_prices = open_prices[1:]
    trimmed_close_prices = close_prices[1:]
    trimmed_high_prices = high_prices[1:]
    trimmed_low_prices = low_prices[1:]
    
    # Calculate intermediate terms
    log_high_low = np.log(trimmed_high_prices / trimmed_low_prices)
    log_close_open = np.log(trimmed_close_prices / trimmed_open_prices)
    
    k = 0.34 / (1.34 + len(log_open_close))
    sigma_open_close = log_open_close ** 2
    sigma_close_open = log_close_open ** 2
    sigma_high_low = log_high_low ** 2

    # Compute Yang-Zhang volatility
    yang_zhang_volatility = np.sqrt(sigma_open_close + k * sigma_close_open + (1 - k) * sigma_high_low)

    # Pad the first row with NaN to match the original length
    return np.insert(yang_zhang_volatility, 0, np.nan)




#def VGamma_VM():
#def SABR_VM():
#def surface_VM():
#def smile_VM():



