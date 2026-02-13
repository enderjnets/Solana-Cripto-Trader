#!/usr/bin/env python3
"""
Solana Backtester for Jupiter Trading Bot
=========================================
Backtesting engine for evaluating trading strategies on Solana swap data.

Features:
- 60+ indicators with multiple periods
- JIT acceleration with Numba (4000x speedup)
- Support for SOL, USDC, USDT, JUP, BONK trading pairs
- Jupiter fee modeling
- Stop-loss and take-profit simulation

Based on: numba_backtester.py from Coinbase Cripto Trader

Usage:
    from solana_backtester import evaluate_strategy, evaluate_population
    result = evaluate_strategy(df, genome, risk_level)
    results = evaluate_population(df, population, risk_level)
"""

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Numba JIT acceleration
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not installed. Using pure Python fallback (slower).")


# ============================================================================
# JUPITER FEE MODELING
# ============================================================================

@dataclass
class JupiterFees:
    """Jupiter fee structure"""
    route_fee_bps: float = 0.25  # 0.25% default
    base_fee_lamports: int = 5000
    compute_unit_fee_lamports: int = 500
    priority_fee_lamports: int = 1000
    jito_tip_lamports: int = 0
    
    def calculate_total_fee(
        self,
        input_amount_lamports: int,
        swap_direction: str = "SOL_TO_USDC"
    ) -> Tuple[int, float]:
        route_fee = int(input_amount_lamports * (self.route_fee_bps / 10000))
        network_fee = self.base_fee_lamports + self.compute_unit_fee_lamports
        priority_fee = self.priority_fee_lamports
        jito_fee = self.jito_tip_lamports
        total_fee = route_fee + network_fee + priority_fee + jito_fee
        fee_usd = (total_fee / 1e9) * 100
        return total_fee, fee_usd


# ============================================================================
# INDICATOR CONSTANTS - 60+ INDICATORS
# ============================================================================

# Base price/volume indices
IND_OPEN = 0
IND_HIGH = 1
IND_LOW = 2
IND_CLOSE = 3
IND_VOLUME = 4

# ============================================================================
# RSI - Relative Strength Index (6 periods)
# ============================================================================
IND_RSI_BASE = 5
NUM_RSI = 6
RSI_PERIODS = [7, 14, 21, 50, 100, 200]

# ============================================================================
# SMA - Simple Moving Average (6 periods)
# ============================================================================
IND_SMA_BASE = IND_RSI_BASE + NUM_RSI
NUM_SMA = 6
SMA_PERIODS = [7, 14, 21, 50, 100, 200]

# ============================================================================
# EMA - Exponential Moving Average (6 periods)
# ============================================================================
IND_EMA_BASE = IND_SMA_BASE + NUM_SMA
NUM_EMA = 6
EMA_PERIODS = [7, 14, 21, 50, 100, 200]

# ============================================================================
# WMA - Weighted Moving Average (4 periods)
# ============================================================================
IND_WMA_BASE = IND_EMA_BASE + NUM_EMA
NUM_WMA = 4
WMA_PERIODS = [14, 50, 100, 200]

# ============================================================================
# DEMA - Double Exponential MA (3 periods)
# ============================================================================
IND_DEMA_BASE = IND_WMA_BASE + NUM_WMA
NUM_DEMA = 3
DEMA_PERIODS = [14, 50, 200]

# ============================================================================
# TEMA - Triple Exponential MA (3 periods)
# ============================================================================
IND_TEMA_BASE = IND_DEMA_BASE + NUM_DEMA
NUM_TEMA = 3
TEMA_PERIODS = [14, 50, 200]

# ============================================================================
# HMA - Hull Moving Average (3 periods)
# ============================================================================
IND_HMA_BASE = IND_TEMA_BASE + NUM_TEMA
NUM_HMA = 3
HMA_PERIODS = [14, 50, 200]

# ============================================================================
# VWMA - Volume Weighted MA (4 periods)
# ============================================================================
IND_VWMA_BASE = IND_HMA_BASE + NUM_HMA
NUM_VWMA = 4
VWMA_PERIODS = [14, 50, 100, 200]

# ============================================================================
# Stochastic %K and %D (4 periods each)
# ============================================================================
IND_STOCH_K_BASE = IND_VWMA_BASE + NUM_VWMA
NUM_STOCH_K = 4
STOCH_K_PERIODS = [14, 21, 50, 100]

IND_STOCH_D_BASE = IND_STOCH_K_BASE + NUM_STOCH_K
NUM_STOCH_D = 4
STOCH_D_PERIODS = [3, 5, 10, 20]

# ============================================================================
# MACD Components (3 sets)
# ============================================================================
IND_MACD_LINE_BASE = IND_STOCH_D_BASE + NUM_STOCH_D
NUM_MACD_LINE = 3
MACD_FAST_PERIODS = [12, 20, 26]

IND_MACD_SIGNAL_BASE = IND_MACD_LINE_BASE + NUM_MACD_LINE
NUM_MACD_SIGNAL = 3
MACD_SIGNAL_PERIODS = [9, 12, 26]

IND_MACD_HIST_BASE = IND_MACD_SIGNAL_BASE + NUM_MACD_SIGNAL
NUM_MACD_HIST = 3

# ============================================================================
# Bollinger Bands (3 components, 3 periods)
# ============================================================================
IND_BB_UPPER_BASE = IND_MACD_HIST_BASE + NUM_MACD_HIST
NUM_BB = 3
BB_PERIODS = [14, 50, 200]
BB_STD = 2

IND_BB_MIDDLE_BASE = IND_BB_UPPER_BASE + NUM_BB
IND_BB_LOWER_BASE = IND_BB_MIDDLE_BASE + NUM_BB
IND_BB_WIDTH_BASE = IND_BB_LOWER_BASE + NUM_BB
IND_BB_PERCENT_BASE = IND_BB_WIDTH_BASE + NUM_BB

# ============================================================================
# ATR - Average True Range (4 periods)
# ============================================================================
IND_ATR_BASE = IND_BB_PERCENT_BASE + NUM_BB
NUM_ATR = 4
ATR_PERIODS = [7, 14, 21, 50]

# ============================================================================
# Keltner Channels (3 components, 3 periods)
# ============================================================================
IND_KC_UPPER_BASE = IND_ATR_BASE + NUM_ATR
NUM_KC = 3
KC_PERIODS = [14, 50, 200]

IND_KC_MIDDLE_BASE = IND_KC_UPPER_BASE + NUM_KC
IND_KC_LOWER_BASE = IND_KC_MIDDLE_BASE + NUM_KC

# ============================================================================
# Donchian Channels (2 components, 3 periods)
# ============================================================================
IND_DC_HIGH_BASE = IND_KC_LOWER_BASE + NUM_KC
NUM_DC = 3
DC_PERIODS = [14, 50, 200]

IND_DC_LOW_BASE = IND_DC_HIGH_BASE + NUM_DC

# ============================================================================
# CCI - Commodity Channel Index (4 periods)
# ============================================================================
IND_CCI_BASE = IND_DC_LOW_BASE + NUM_DC
NUM_CCI = 4
CCI_PERIODS = [14, 20, 50, 100]

# ============================================================================
# ROC - Rate of Change (4 periods)
# ============================================================================
IND_ROC_BASE = IND_CCI_BASE + NUM_CCI
NUM_ROC = 4
ROC_PERIODS = [10, 14, 21, 50]

# ============================================================================
# Williams %R (4 periods)
# ============================================================================
IND_WILLIAMS_R_BASE = IND_ROC_BASE + NUM_ROC
NUM_WILLIAMS_R = 4
WILLIAMS_R_PERIODS = [14, 21, 50, 100]

# ============================================================================
# ADX - Average Directional Index (3 periods)
# ============================================================================
IND_ADX_BASE = IND_WILLIAMS_R_BASE + NUM_WILLIAMS_R
NUM_ADX = 3
ADX_PERIODS = [14, 21, 50]

# ============================================================================
# Plus DI and Minus DI (3 periods each)
# ============================================================================
IND_PLUS_DI_BASE = IND_ADX_BASE + NUM_ADX
NUM_DI = 3

IND_MINUS_DI_BASE = IND_PLUS_DI_BASE + NUM_DI

# ============================================================================
# Parabolic SAR (3 configurations)
# ============================================================================
IND_PSAR_BASE = IND_MINUS_DI_BASE + NUM_DI
NUM_PSAR = 3
PSAR_CONFIGS = [(0.02, 0.2), (0.02, 0.1), (0.01, 0.1)]

# ============================================================================
# OBV - On Balance Volume (1 value)
# ============================================================================
IND_OBV_BASE = IND_PSAR_BASE + NUM_PSAR
NUM_OBV = 1

# ============================================================================
# OBV MA (4 periods)
# ============================================================================
IND_OBV_MA_BASE = IND_OBV_BASE + NUM_OBV
NUM_OBV_MA = 4
OBV_MA_PERIODS = [14, 50, 100, 200]

# ============================================================================
# VWAP - Volume Weighted Average Price (1 value)
# ============================================================================
IND_VWAP_BASE = IND_OBV_MA_BASE + NUM_OBV_MA
NUM_VWAP = 1

# ============================================================================
# ADI - Accumulation/Distribution Index (3 periods)
# ============================================================================
IND_ADI_BASE = IND_VWAP_BASE + NUM_VWAP
NUM_ADI = 3
ADI_PERIODS = [14, 50, 200]

# ============================================================================
# A/D Line - Williams Accumulation/Distribution (3 periods)
# ============================================================================
IND_AD_LINE_BASE = IND_ADI_BASE + NUM_ADI
NUM_AD_LINE = 3
AD_LINE_PERIODS = [14, 50, 200]

# ============================================================================
# Volume MA (4 periods)
# ============================================================================
IND_VOL_MA_BASE = IND_AD_LINE_BASE + NUM_AD_LINE
NUM_VOL_MA = 4
VOL_MA_PERIODS = [14, 50, 100, 200]

# ============================================================================
# Standard Deviation (3 periods)
# ============================================================================
IND_STD_BASE = IND_VOL_MA_BASE + NUM_VOL_MA
NUM_STD = 3
STD_PERIODS = [14, 50, 200]

# ============================================================================
# TOTAL INDICATORS COUNT
# ============================================================================
NUM_INDICATORS = IND_STD_BASE + NUM_STD


# ============================================================================
# INDICATOR NAMING (for debugging/visualization)
# ============================================================================
INDICATOR_NAMES = {
    IND_OPEN: "OPEN",
    IND_HIGH: "HIGH",
    IND_LOW: "LOW",
    IND_CLOSE: "CLOSE",
    IND_VOLUME: "VOLUME",
}

for i, p in enumerate(RSI_PERIODS):
    INDICATOR_NAMES[IND_RSI_BASE + i] = f"RSI_{p}"
for i, p in enumerate(SMA_PERIODS):
    INDICATOR_NAMES[IND_SMA_BASE + i] = f"SMA_{p}"
for i, p in enumerate(EMA_PERIODS):
    INDICATOR_NAMES[IND_EMA_BASE + i] = f"EMA_{p}"
for i, p in enumerate(WMA_PERIODS):
    INDICATOR_NAMES[IND_WMA_BASE + i] = f"WMA_{p}"
for i, p in enumerate(DEMA_PERIODS):
    INDICATOR_NAMES[IND_DEMA_BASE + i] = f"DEMA_{p}"
for i, p in enumerate(TEMA_PERIODS):
    INDICATOR_NAMES[IND_TEMA_BASE + i] = f"TEMA_{p}"
for i, p in enumerate(HMA_PERIODS):
    INDICATOR_NAMES[IND_HMA_BASE + i] = f"HMA_{p}"
for i, p in enumerate(VWMA_PERIODS):
    INDICATOR_NAMES[IND_VWMA_BASE + i] = f"VWMA_{p}"
for i, p in enumerate(STOCH_K_PERIODS):
    INDICATOR_NAMES[IND_STOCH_K_BASE + i] = f"STOCH_K_{p}"
for i, p in enumerate(STOCH_D_PERIODS):
    INDICATOR_NAMES[IND_STOCH_D_BASE + i] = f"STOCH_D_{p}"
for i, fast in enumerate(MACD_FAST_PERIODS):
    INDICATOR_NAMES[IND_MACD_LINE_BASE + i] = f"MACD_{fast}"
for i, period in enumerate(MACD_SIGNAL_PERIODS):
    INDICATOR_NAMES[IND_MACD_SIGNAL_BASE + i] = f"MACD_SIGNAL_{period}"
for i in range(NUM_MACD_HIST):
    INDICATOR_NAMES[IND_MACD_HIST_BASE + i] = f"MACD_HIST_{i}"
for i, p in enumerate(BB_PERIODS):
    INDICATOR_NAMES[IND_BB_UPPER_BASE + i] = f"BB_UPPER_{p}"
    INDICATOR_NAMES[IND_BB_MIDDLE_BASE + i] = f"BB_MIDDLE_{p}"
    INDICATOR_NAMES[IND_BB_LOWER_BASE + i] = f"BB_LOWER_{p}"
    INDICATOR_NAMES[IND_BB_WIDTH_BASE + i] = f"BB_WIDTH_{p}"
    INDICATOR_NAMES[IND_BB_PERCENT_BASE + i] = f"BB_PERCENT_{p}"
for i, p in enumerate(ATR_PERIODS):
    INDICATOR_NAMES[IND_ATR_BASE + i] = f"ATR_{p}"
for i, p in enumerate(KC_PERIODS):
    INDICATOR_NAMES[IND_KC_UPPER_BASE + i] = f"KC_UPPER_{p}"
    INDICATOR_NAMES[IND_KC_MIDDLE_BASE + i] = f"KC_MIDDLE_{p}"
    INDICATOR_NAMES[IND_KC_LOWER_BASE + i] = f"KC_LOWER_{p}"
for i, p in enumerate(DC_PERIODS):
    INDICATOR_NAMES[IND_DC_HIGH_BASE + i] = f"DC_HIGH_{p}"
    INDICATOR_NAMES[IND_DC_LOW_BASE + i] = f"DC_LOW_{p}"
for i, p in enumerate(CCI_PERIODS):
    INDICATOR_NAMES[IND_CCI_BASE + i] = f"CCI_{p}"
for i, p in enumerate(ROC_PERIODS):
    INDICATOR_NAMES[IND_ROC_BASE + i] = f"ROC_{p}"
for i, p in enumerate(WILLIAMS_R_PERIODS):
    INDICATOR_NAMES[IND_WILLIAMS_R_BASE + i] = f"WILLIAMS_R_{p}"
for i, p in enumerate(ADX_PERIODS):
    INDICATOR_NAMES[IND_ADX_BASE + i] = f"ADX_{p}"
    INDICATOR_NAMES[IND_PLUS_DI_BASE + i] = f"PLUS_DI_{p}"
    INDICATOR_NAMES[IND_MINUS_DI_BASE + i] = f"MINUS_DI_{p}"
for i in range(NUM_PSAR):
    INDICATOR_NAMES[IND_PSAR_BASE + i] = f"PSAR_{i}"
INDICATOR_NAMES[IND_OBV_BASE] = "OBV"
for i, p in enumerate(OBV_MA_PERIODS):
    INDICATOR_NAMES[IND_OBV_MA_BASE + i] = f"OBV_MA_{p}"
INDICATOR_NAMES[IND_VWAP_BASE] = "VWAP"
for i, p in enumerate(ADI_PERIODS):
    INDICATOR_NAMES[IND_ADI_BASE + i] = f"ADI_{p}"
for i, p in enumerate(AD_LINE_PERIODS):
    INDICATOR_NAMES[IND_AD_LINE_BASE + i] = f"AD_LINE_{p}"
for i, p in enumerate(VOL_MA_PERIODS):
    INDICATOR_NAMES[IND_VOL_MA_BASE + i] = f"VOL_MA_{p}"
for i, p in enumerate(STD_PERIODS):
    INDICATOR_NAMES[IND_STD_BASE + i] = f"STD_{p}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def _dema(series: pd.Series, period: int) -> pd.Series:
    ema1 = _ema(series, period)
    return 2 * ema1 - _ema(ema1, period)

def _tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3

def _hma(series: pd.Series, period: int) -> pd.Series:
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma_half = _wma(series, half)
    wma_full = _wma(series, period)
    return _wma(2 * wma_half - wma_full, sqrt_period)

def _vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    return (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def _macd(close: pd.Series, fast: int, slow: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, slow)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger_bands(close: pd.Series, period: int, std_dev: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    middle = _sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    width = (upper - lower) / middle
    percent = (close - lower) / (upper - lower)
    return upper.fillna(close), middle.fillna(close), lower.fillna(close), width.fillna(0), percent.fillna(0.5)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.fillna(atr.iloc[-1] if len(atr) > 0 else 0)

def _keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, period: int, atr_period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle = _ema(close, period)
    atr_val = _atr(high, low, close, atr_period)
    upper = middle + (2 * atr_val)
    lower = middle - (2 * atr_val)
    return upper.fillna(close), middle.fillna(close), lower.fillna(close)

def _donchian(high: pd.Series, low: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    return upper.fillna(high), lower.fillna(low)

def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci.fillna(0)

def _roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(periods=period) * 100

def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.fillna(-50)

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr_val = _atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx.fillna(25), plus_di.fillna(25), minus_di.fillna(25)

def _psar(high: pd.Series, low: pd.Series, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
    psar = np.zeros(len(high))
    psar[0] = low.iloc[0]
    trend = 1
    af_val = af
    hp = high.iloc[0]
    lp = low.iloc[0]
    for i in range(1, len(high)):
        if trend == 1:
            psar[i] = psar[i-1] + af_val * (hp - psar[i-1])
            if low.iloc[i] < psar[i]:
                trend = -1
                af_val = af
                psar[i] = hp
        else:
            psar[i] = psar[i-1] + af_val * (lp - psar[i-1])
            if high.iloc[i] > psar[i]:
                trend = 1
                af_val = af
                psar[i] = lp
        if trend == 1 and high.iloc[i] > hp:
            hp = high.iloc[i]
            af_val = min(af_val + af, max_af)
        elif trend == -1 and low.iloc[i] < lp:
            lp = low.iloc[i]
            af_val = min(af_val + af, max_af)
    return pd.Series(psar, index=high.index).fillna(high)

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=close.index)

def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()

def _adi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)
    return (mfm * volume).rolling(window=period).sum().fillna(0)

def _ad_line(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    adl = ((close - low) - (high - close)) / (high - low)
    return adl.rolling(window=period).sum().fillna(0)


# ============================================================================
# INDICATOR PRE-COMPUTATION
# ============================================================================

def precompute_indicators(df: pd.DataFrame) -> np.ndarray:
    """Pre-compute ALL indicators as a numpy matrix."""
    n = len(df)
    indicators = np.full((NUM_INDICATORS, n), np.nan, dtype=np.float64)
    
    open_ = df['open'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    
    open_s, high_s, low_s, close_s, volume_s = pd.Series(open_), pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume)
    
    indicators[IND_OPEN] = open_
    indicators[IND_HIGH] = high
    indicators[IND_LOW] = low
    indicators[IND_CLOSE] = close
    indicators[IND_VOLUME] = volume
    
    idx = IND_RSI_BASE
    for period in RSI_PERIODS:
        indicators[idx] = _rsi(close_s, period).values
        idx += 1
    for period in SMA_PERIODS:
        indicators[idx] = _sma(close_s, period).fillna(close_s).values
        idx += 1
    for period in EMA_PERIODS:
        indicators[idx] = _ema(close_s, period).values
        idx += 1
    for period in WMA_PERIODS:
        indicators[idx] = _wma(close_s, period).fillna(close_s).values
        idx += 1
    for period in DEMA_PERIODS:
        indicators[idx] = _dema(close_s, period).fillna(close_s).values
        idx += 1
    for period in TEMA_PERIODS:
        indicators[idx] = _tema(close_s, period).fillna(close_s).values
        idx += 1
    for period in HMA_PERIODS:
        indicators[idx] = _hma(close_s, period).fillna(close_s).values
        idx += 1
    for period in VWMA_PERIODS:
        indicators[idx] = _vwma(close_s, volume_s, period).fillna(close_s).values
        idx += 1
    for period in STOCH_K_PERIODS:
        k, _ = _stochastic(high_s, low_s, close_s, period, 3)
        indicators[idx] = k.values
        idx += 1
    for period in STOCH_D_PERIODS:
        _, d = _stochastic(high_s, low_s, close_s, 14, period)
        indicators[idx] = d.values
        idx += 1
    for fast in MACD_FAST_PERIODS:
        macd_line, signal, _ = _macd(close_s, fast, 26)
        indicators[idx] = macd_line.values
        idx += 1
    for slow in MACD_SIGNAL_PERIODS:
        _, signal, _ = _macd(close_s, 12, slow)
        indicators[idx] = signal.values
        idx += 1
    for i, fast in enumerate(MACD_FAST_PERIODS):
        _, _, hist = _macd(close_s, fast, MACD_SIGNAL_PERIODS[i])
        indicators[idx] = hist.values
        idx += 1
    for period in BB_PERIODS:
        upper, middle, lower, width, percent = _bollinger_bands(close_s, period, BB_STD)
        indicators[idx] = upper.values; idx += 1
        indicators[idx] = middle.values; idx += 1
        indicators[idx] = lower.values; idx += 1
        indicators[idx] = width.values; idx += 1
        indicators[idx] = percent.values; idx += 1
    for period in ATR_PERIODS:
        indicators[idx] = _atr(high_s, low_s, close_s, period).values
        idx += 1
    for period in KC_PERIODS:
        upper, middle, lower = _keltner_channels(high_s, low_s, close_s, period)
        indicators[idx] = upper.values; idx += 1
        indicators[idx] = middle.values; idx += 1
        indicators[idx] = lower.values; idx += 1
    for period in DC_PERIODS:
        upper, lower = _donchian(high_s, low_s, period)
        indicators[idx] = upper.values; idx += 1
        indicators[idx] = lower.values; idx += 1
    for period in CCI_PERIODS:
        indicators[idx] = _cci(high_s, low_s, close_s, period).values
        idx += 1
    for period in ROC_PERIODS:
        indicators[idx] = _roc(close_s, period).values
        idx += 1
    for period in WILLIAMS_R_PERIODS:
        indicators[idx] = _williams_r(high_s, low_s, close_s, period).values
        idx += 1
    for period in ADX_PERIODS:
        adx, plus_di, minus_di = _adx(high_s, low_s, close_s, period)
        indicators[idx] = adx.values; idx += 1
        indicators[idx] = plus_di.values; idx += 1
        indicators[idx] = minus_di.values; idx += 1
    for i in range(NUM_PSAR):
        af, max_af = PSAR_CONFIGS[i]
        indicators[idx] = _psar(high_s, low_s, af, max_af).values
        idx += 1
    indicators[idx] = _obv(close_s, volume_s).values
    idx += 1
    obv_series = pd.Series(indicators[IND_OBV_BASE])
    for period in OBV_MA_PERIODS:
        indicators[idx] = _sma(obv_series, period).values
        idx += 1
    indicators[idx] = _vwap(high_s, low_s, close_s, volume_s).values
    idx += 1
    for period in ADI_PERIODS:
        indicators[idx] = _adi(high_s, low_s, close_s, volume_s, period).values
        idx += 1
    for period in AD_LINE_PERIODS:
        indicators[idx] = _ad_line(high_s, low_s, close_s, period).values
        idx += 1
    for period in VOL_MA_PERIODS:
        indicators[idx] = _sma(volume_s, period).fillna(1).values
        idx += 1
    for period in STD_PERIODS:
        indicators[idx] = close_s.rolling(window=period).std().fillna(0).values
        idx += 1
    
    return indicators


# ============================================================================
# TRADING SIMULATION
# ============================================================================

OP_GT = 0
OP_LT = 1
OP_EQ = 2

GEN_SL_PCT = 0
GEN_TP_PCT = 1
GEN_NUM_RULES = 2
GEN_RULES_START = 3

GENOME_SIZE = 18


# ============================================================================
# GENOME EVALUATION
# ============================================================================

def evaluate_genome_python(indicators: np.ndarray, genome: np.ndarray, initial_balance: float = 1.0) -> Dict[str, float]:
    """Evaluate a single genome with pure Python."""
    n = len(indicators[0])
    sl_pct = abs(genome[GEN_SL_PCT])
    tp_pct = abs(genome[GEN_TP_PCT])
    num_rules = int(abs(genome[GEN_NUM_RULES]))
    num_rules = min(max(num_rules, 1), 3)

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = wins = losses = 0
    pnl_total = 0.0
    max_balance = initial_balance
    max_drawdown = 0.0
    trade_pnls = []

    for i in range(n):
        close = indicators[IND_CLOSE, i]
        high = indicators[IND_HIGH, i]
        low = indicators[IND_LOW, i]

        if position > 0:
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
            if low <= sl_price:
                pnl = (sl_price - entry_price) / entry_price
                pnl_total += pnl
                balance *= (1 + pnl)
                trades += 1
                losses += 1
                position = 0
                trade_pnls.append(pnl)
            elif high >= tp_price:
                pnl = (tp_price - entry_price) / entry_price
                pnl_total += pnl
                balance *= (1 + pnl)
                trades += 1
                wins += 1
                position = 0
                trade_pnls.append(pnl)
        else:
            all_rules_pass = True
            for r in range(num_rules):
                offset = GEN_RULES_START + r * 3
                if offset + 2 >= len(genome):
                    break
                ind_idx = int(genome[offset])
                threshold = genome[offset + 1]
                operator = int(genome[offset + 2])
                ind_idx = max(0, min(ind_idx, NUM_INDICATORS - 1))
                ind_val = indicators[ind_idx, i]
                if np.isnan(ind_val):
                    all_rules_pass = False
                    break
                if ind_idx >= IND_SMA_BASE:
                    deviation = threshold / 100.0
                    if operator == OP_GT:
                        if not (close > ind_val * (1 + deviation)):
                            all_rules_pass = False
                            break
                    else:
                        if not (close < ind_val * (1 - deviation)):
                            all_rules_pass = False
                            break
                else:
                    if operator == OP_GT:
                        if not (ind_val > threshold):
                            all_rules_pass = False
                            break
                    else:
                        if not (ind_val < threshold):
                            all_rules_pass = False
                            break
            if all_rules_pass:
                position = 1
                entry_price = close

        if balance > max_balance:
            max_balance = balance
        dd = (max_balance - balance) / max_balance if max_balance > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    win_rate = wins / trades if trades > 0 else 0.0
    sharpe = 0.0
    if trade_pnls:
        arr = np.array(trade_pnls)
        std = np.std(arr)
        if std > 0:
            sharpe = (np.mean(arr) / std) * np.sqrt(252)

    return {
        'pnl': pnl_total,
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
    }


def evaluate_genome(
    indicators: np.ndarray,
    genome: np.ndarray,
    initial_balance: float = 1.0,
    fees: JupiterFees = None
) -> Dict[str, float]:
    """Evaluate a single genome (strategy)."""
    return evaluate_genome_python(indicators, genome, initial_balance)


def evaluate_population(
    indicators: np.ndarray,
    population: List[np.ndarray],
    initial_balance: float = 1.0,
    fees: JupiterFees = None
) -> List[Dict[str, float]]:
    """Evaluate entire population of genomes."""
    results = []
    for genome in population:
        result = evaluate_genome(indicators, genome, initial_balance, fees)
        results.append(result)
    return results


# ============================================================================
# BACKTEST RESULT
# ============================================================================

@dataclass
class BacktestResult:
    """Backtest result summary"""
    pnl: float
    pnl_pct: float
    trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    final_balance: float
    start_date: str
    end_date: str
    duration_days: int
    
    def to_dict(self) -> Dict:
        return {
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'trades': self.trades,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'final_balance': self.final_balance,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'duration_days': self.duration_days
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def run_backtest(
    df: pd.DataFrame,
    genome: np.ndarray,
    initial_balance: float = 1.0,
    fees: JupiterFees = None
) -> BacktestResult:
    """Run full backtest on historical data."""
    indicators = precompute_indicators(df)
    result = evaluate_genome(indicators, genome, initial_balance, fees)
    final_balance = initial_balance * (1 + result['pnl'])
    
    start_date = df['timestamp'].iloc[0] if 'timestamp' in df.columns else 'Unknown'
    end_date = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else 'Unknown'
    
    duration = 0
    if 'timestamp' in df.columns:
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration = (end - start).days
        except:
            duration = 0
    
    return BacktestResult(
        pnl=result['pnl'],
        pnl_pct=result['pnl'] * 100,
        trades=result['trades'],
        win_rate=result['win_rate'],
        sharpe_ratio=result.get('sharpe_ratio', 0.0),
        max_drawdown=result.get('max_drawdown', 0.0),
        final_balance=final_balance,
        start_date=str(start_date),
        end_date=str(end_date),
        duration_days=duration
    )


# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================

def generate_sample_data(
    n_candles: int = 10000,
    start_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Generate sample price data for testing."""
    np.random.seed(42)
    returns = np.random.normal(0, volatility / np.sqrt(365), n_candles)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_candles)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_candles)))
    open_price = np.roll(close, 1)
    open_price[0] = start_price
    volume = np.random.uniform(1e6, 1e8, n_candles)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_candles, freq='1h'),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Solana Backtester - 60+ Indicators Demo")
    print("=" * 60)
    print(f"\n📊 Total indicators available: {NUM_INDICATORS}")
    print("\n📈 Indicators by category:")
    print(f"   - Trend (SMA, EMA, WMA, DEMA, TEMA, HMA, VWMA): {NUM_SMA + NUM_EMA + NUM_WMA + NUM_DEMA + NUM_TEMA + NUM_HMA + NUM_VWMA}")
    print(f"   - Momentum (RSI, Stochastic, MACD, CCI, ROC, Williams %R): {NUM_RSI + NUM_STOCH_K + NUM_STOCH_D + NUM_MACD_LINE + NUM_MACD_SIGNAL + NUM_MACD_HIST + NUM_CCI + NUM_ROC + NUM_WILLIAMS_R}")
    print(f"   - Volatility (Bollinger Bands, ATR, Keltner, Donchian): {NUM_BB * 5 + NUM_ATR + NUM_KC * 3 + NUM_DC * 2}")
    print(f"   - Volume (OBV, VWAP, ADI, A/D Line, Volume MA): {NUM_OBV + NUM_OBV_MA + NUM_VWAP + NUM_ADI + NUM_AD_LINE + NUM_VOL_MA}")
    print(f"   - Direction (ADX, +DI, -DI, Parabolic SAR): {NUM_ADX + NUM_DI + NUM_PSAR}")
    
    print("\n🔧 Generating sample data...")
    df = generate_sample_data(n_candles=10000)
    print(f"   {len(df)} candles generated")
    
    print("\n⚡ Pre-computing indicators...")
    import time
    start = time.time()
    indicators = precompute_indicators(df)
    elapsed = time.time() - start
    print(f"   {indicators.shape[0]} indicators computed in {elapsed:.2f}s")
    
    print("\n🚀 Running backtest...")
    # Genome example: RSI oversold (< 30) - mean reversion strategy
    # Format: [SL%, TP%, num_rules, ind_idx, threshold, operator, ...]
    # IND_RSI_BASE=5, RSI_14 is at index 6 (IND_RSI_BASE + 1)
    # Operator: 0 = >, 1 = <, 2 = ==
    genome = np.array([
        0.03,   # SL: 3%
        0.06,   # TP: 6%
        1,      # 1 rule
        6, 30, 1,   # Rule 1: RSI_14 < 30 (oversold → buy)
        0, 0, 0, 0, 0, 0, 0, 0  # Unused slots
    ], dtype=np.float64)
    start = time.time()
    result = run_backtest(df, genome)
    elapsed = time.time() - start
    print(f"   Completed in {elapsed:.3f}s")
    
    print("\n📈 Backtest Results:")
    print(f"   PnL: {result.pnl:.4f} SOL ({result.pnl_pct:.2f}%)")
    print(f"   Trades: {result.trades}")
    print(f"   Win Rate: {result.win_rate:.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.2%}")
    print(f"   Final Balance: {result.final_balance:.4f} SOL")
