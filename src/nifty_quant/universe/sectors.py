"""Sector classification for equities in the Nifty universe.

WARNING: This is a CURRENT-DAY sector map, NOT point-in-time. A symbol that
changed sectors historically is mislabelled here. Use this only for
contemporaneous cross-sectional analysis. For backtests covering extended
periods, point-in-time sector assignments are necessary.
"""

from __future__ import annotations

import numpy as np

SECTOR_MAP = {
    # Financials (34 symbols: banks, insurance, NBFC, brokers, asset mgmt, holding)
    "AXISBANK": "Financials",
    "HDFCBANK": "Financials",
    "ICICIBANK": "Financials",
    "INDUSINDBK": "Financials",
    "KOTAKBANK": "Financials",
    "CANBK": "Financials",
    "BANKBARODA": "Financials",
    "SBIN": "Financials",
    "UNIONBANK": "Financials",
    "YESBANK": "Financials",
    "BANDHANBNK": "Financials",
    "PNB": "Financials",
    "ICICIGI": "Financials",
    "ICICIPRULI": "Financials",
    "SBILIFE": "Financials",
    "SBICARD": "Financials",
    "HDFCLIFE": "Financials",
    "LICI": "Financials",
    "NIACL": "Financials",
    "BAJAJFINSV": "Financials",
    "BAJFINANCE": "Financials",
    "SHRIRAMFIN": "Financials",
    "MUTHOOTFIN": "Financials",
    "ABCAPITAL": "Financials",
    "LICHSGFIN": "Financials",
    "OFSS": "Financials",
    "JIOFIN": "Financials",
    "HDFCAMC": "Financials",
    "PIIND": "Financials",
    "CHOLAFIN": "Financials",
    "GICRE": "Financials",
    "TATACAP": "Financials",
    "CGPOWER": "Financials",
    "BAJAJHLDNG": "Financials",

    # Information Technology (5 symbols)
    "INFY": "Information Technology",
    "HCLTECH": "Information Technology",
    "WIPRO": "Information Technology",
    "TECHM": "Information Technology",
    "TCS": "Information Technology",

    # Pharma & Healthcare (12 symbols)
    "CIPLA": "Pharma & Healthcare",
    "AUROPHARMA": "Pharma & Healthcare",
    "DRREDDY": "Pharma & Healthcare",
    "LUPIN": "Pharma & Healthcare",
    "SUNPHARMA": "Pharma & Healthcare",
    "BIOCON": "Pharma & Healthcare",
    "GLAXO": "Pharma & Healthcare",
    "GLENMARK": "Pharma & Healthcare",
    "TORNTPHARM": "Pharma & Healthcare",
    "MAXHEALTH": "Pharma & Healthcare",
    "DIVISLAB": "Pharma & Healthcare",
    "APOLLOHOSP": "Pharma & Healthcare",
    "ZYDUSLIFE": "Pharma & Healthcare",

    # FMCG & Consumer (13 symbols)
    "ITC": "FMCG & Consumer",
    "COLPAL": "FMCG & Consumer",
    "HINDUNILVR": "FMCG & Consumer",
    "BRITANNIA": "FMCG & Consumer",
    "MARICO": "FMCG & Consumer",
    "DABUR": "FMCG & Consumer",
    "GODREJCP": "FMCG & Consumer",
    "NAUKRI": "FMCG & Consumer",
    "TRENT": "FMCG & Consumer",
    "PGHH": "FMCG & Consumer",
    "TATACONSUM": "FMCG & Consumer",
    "DMART": "FMCG & Consumer",
    "NESTLEIND": "FMCG & Consumer",

    # Automobiles (11 symbols)
    "MARUTI": "Automobiles",
    "BAJAJ-AUTO": "Automobiles",
    "EICHERMOT": "Automobiles",
    "HEROMOTOCO": "Automobiles",
    "M&M": "Automobiles",
    "TVSMOTOR": "Automobiles",
    "ASHOKLEY": "Automobiles",
    "MOTHERSON": "Automobiles",
    "HYUNDAI": "Automobiles",
    "TMCV": "Automobiles",
    "ETERNAL": "Automobiles",

    # Metals & Mining (9 symbols)
    "HINDALCO": "Metals & Mining",
    "TATASTEEL": "Metals & Mining",
    "JSWSTEEL": "Metals & Mining",
    "VEDL": "Metals & Mining",
    "SAIL": "Metals & Mining",
    "JINDALSTEL": "Metals & Mining",
    "NMDC": "Metals & Mining",
    "COALINDIA": "Metals & Mining",
    "HINDZINC": "Metals & Mining",

    # Energy & Oil/Gas (8 symbols)
    "ONGC": "Energy & Oil/Gas",
    "RELIANCE": "Energy & Oil/Gas",
    "IOC": "Energy & Oil/Gas",
    "BPCL": "Energy & Oil/Gas",
    "GAIL": "Energy & Oil/Gas",
    "OIL": "Energy & Oil/Gas",
    "PETRONET": "Energy & Oil/Gas",
    "HINDPETRO": "Energy & Oil/Gas",

    # Power & Utilities (8 symbols)
    "POWERGRID": "Power & Utilities",
    "NTPC": "Power & Utilities",
    "BHEL": "Power & Utilities",
    "TATAPOWER": "Power & Utilities",
    "ADANIPOWER": "Power & Utilities",
    "PFC": "Power & Utilities",
    "NHPC": "Power & Utilities",
    "RECLTD": "Power & Utilities",

    # Infrastructure & Capital Goods (11 symbols)
    "LT": "Infrastructure & Capital Goods",
    "CONCOR": "Infrastructure & Capital Goods",
    "IRCTC": "Infrastructure & Capital Goods",
    "MAZDOCK": "Infrastructure & Capital Goods",
    "BEL": "Infrastructure & Capital Goods",
    "CUMMINSIND": "Infrastructure & Capital Goods",
    "HAL": "Infrastructure & Capital Goods",
    "BOSCHLTD": "Infrastructure & Capital Goods",
    "LTM": "Infrastructure & Capital Goods",
    "SIEMENS": "Infrastructure & Capital Goods",
    "GRASIM": "Infrastructure & Capital Goods",

    # Cement & Construction (4 symbols)
    "SHREECEM": "Cement & Construction",
    "ULTRACEMCO": "Cement & Construction",
    "AMBUJACEM": "Cement & Construction",
    "ACC": "Cement & Construction",

    # Chemicals (5 symbols)
    "PIDILITIND": "Chemicals",
    "UPL": "Chemicals",
    "ATGL": "Chemicals",
    "BERGEPAINT": "Chemicals",
    "SRF": "Chemicals",

    # Real Estate (3 symbols) - consolidated with Infrastructure
    "DLF": "Infrastructure & Capital Goods",
    "LODHA": "Infrastructure & Capital Goods",
    "UNITDSPR": "Infrastructure & Capital Goods",

    # Telecom & Media (5 symbols)
    "BHARTIARTL": "Telecom & Media",
    "IDEA": "Telecom & Media",
    "SUNTV": "Telecom & Media",
    "ZEEL": "Telecom & Media",
    "NYKAA": "Telecom & Media",

    # Renewables & Infrastructure (6 symbols)
    "ADANIENSOL": "Renewables & Infrastructure",
    "ADANIGREEN": "Renewables & Infrastructure",
    "SOLARINDS": "Renewables & Infrastructure",
    "ADANIPORTS": "Renewables & Infrastructure",
    "INDUSTOWER": "Renewables & Infrastructure",
    "ADANIENT": "Renewables & Infrastructure",

    # Consumer Durables (6 symbols)
    "HAVELLS": "Consumer Durables",
    "ASIANPAINT": "Consumer Durables",
    "TITAN": "Consumer Durables",
    "MRF": "Consumer Durables",
    "AWL": "Consumer Durables",
    "PAGEIND": "Consumer Durables",

    # Aviation & Hospitality (3 symbols) - consolidated with FMCG & Consumer
    "INDIGO": "FMCG & Consumer",
    "INDHOTEL": "FMCG & Consumer",
    "IRFC": "FMCG & Consumer",

    # Conglomerates & Trading - consolidated into primary sectors
    "EMAMILTD": "FMCG & Consumer",
    "ENRIN": "Energy & Oil/Gas",
    "TMPV": "Metals & Mining",

    # Utilities & Other (1 symbol)
    "VBL": "Power & Utilities",

    # ABB (electrical equipment) - Infrastructure
    "ABB": "Infrastructure & Capital Goods",
}


def sector_ids(symbols: list[str] | np.ndarray) -> np.ndarray:
    """Map symbol names to sector integer IDs.

    Args:
        symbols: List or 1D array of symbol strings.

    Returns:
        int32 array where each element is the sector ID for the corresponding symbol.
        Unknown symbols get ID -1.
    """
    symbols_list = list(symbols)
    sector_id_map = {}
    unique_sectors = sorted(set(SECTOR_MAP.values()))
    for idx, sector in enumerate(unique_sectors):
        sector_id_map[sector] = idx

    ids = np.full(len(symbols_list), -1, dtype=np.int32)
    for i, sym in enumerate(symbols_list):
        if sym in SECTOR_MAP:
            sector = SECTOR_MAP[sym]
            ids[i] = sector_id_map[sector]

    return ids
