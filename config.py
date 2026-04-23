from dataclasses import dataclass, field
from typing import List, Optional


SECTOR_ETFS: List[str] = [
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLV", "XLY",
    "IYR", "IYT", "OIH", "SMH", "RTH", "RKH", "UTH",
    "XLB",
    "XLU",
    "GDX",
    "IBB",
    "KRE",
    "XOP",
    "XHB",
    "XRT",
    "IYZ",
    "XME",
]

MARKET_ETF: str = "SPY"

DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "BAC", "GS", "WFC", "C",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "JNJ", "PFE", "UNH", "ABT", "MRK",
    "HD", "PG", "KO", "PEP", "WMT",
    "NEE", "DUK", "SO", "D", "AEP",
    "UNP", "UPS", "FDX", "CSX", "NSC",
    "NVDA", "INTC", "AVGO", "TXN", "QCOM",
]

# ─────────────────────────────────────────────────────────────────────────────
# Paper-style universe (~1,000 names)
#
# Avellaneda & Lee (2010) use the point-in-time CRSP universe of roughly
# 1,417 US stocks with market cap > $1B on each trading date, 1997-2007.
# We cannot reconstruct that exactly without per-date constituent lists, but
# the list below is a close static proxy: S&P 500 survivors + S&P 400/600
# large-caps + major delisted names (Lehman, Bear Stearns, Fannie, Freddie,
# Compaq, Sun, WaMu, Wyeth, Schering-Plough, Burlington Northern, XTO,
# pre-merger airlines, pre-2005 Kraft, etc.) that were actively traded
# during the 1997-2007 sample.
#
# To use it, swap this block in for DEFAULT_TICKERS above:
#     DEFAULT_TICKERS = PAPER_TICKERS
#
# Data-quality notes (already handled in the pipeline):
#   - Delisted names: the CRSP data source maps them via historical PERMNO
#     lookups. With yfinance many of the delisted tickers return nothing;
#     those names are silently dropped by the `available` filter in
#     app/Home.py. Prefer CRSP for this universe.
#   - IPOs mid-sample: returns are NaN before the first trading day; the
#     OU fitter ignores those rows via the np.isfinite mask.
#   - PCA window with lots of IPO/delisting: the helper fills isolated NaN
#     with 0 (unbiased for log-returns) so a single missing print doesn't
#     collapse the 252-day window (fix in statarb/factors/pca.py).
# ─────────────────────────────────────────────────────────────────────────────
PAPER_TICKERS: List[str] = [
    # --- Technology / Hardware / Semis (XLK, SMH) ---
    "AAPL", "MSFT", "ORCL", "IBM", "HPQ", "DELL", "CSCO", "INTC", "AMD",
    "NVDA", "TXN", "QCOM", "AVGO", "MU", "AMAT", "KLAC", "LRCX", "ADI",
    "ADSK", "CTXS", "CA", "SYMC", "INTU", "CRM", "ADBE", "BMC", "CHKP",
    "VRSN", "AKAM", "EBAY", "YHOO", "GOOG", "GOOGL", "AMZN", "PCLN", "EXPE",
    "NTAP", "EMC", "STX", "WDC", "SNDK", "BRCD", "JNPR", "FFIV", "RHT",
    "TLAB", "CIEN", "FNSR", "JDSU", "NVLS", "LLTC", "MCHP", "XLNX", "ALTR",
    "BRCM", "MRVL", "NVLSQ", "ATML", "FSL", "ONNN", "TER", "CY", "IRF",
    "POWI", "SWKS", "RFMD", "TQNT", "AVCT", "VRSK", "PAYX", "ADP", "FIS",
    "FISV", "JKHY", "WU", "MA", "V", "DFS", "COF", "AXP", "GLW", "CPQ",
    "SUNW", "NCR", "PALM", "RIMM", "MOT", "NOK", "ERIC", "LU", "NT",
    "SCMR", "LSI", "AGR",

    # --- Financials: Banks / Broker-Dealers / Asset Mgmt / Insurance (XLF, KRE) ---
    "JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "BK", "STT",
    "NTRS", "SCHW", "TROW", "BEN", "IVZ", "AMG", "JNS", "AFL", "AIG",
    "ALL", "HIG", "MET", "PRU", "TRV", "CB", "CINF", "PGR", "XL", "LNC",
    "MMC", "AJG", "AON", "WRB", "AFG", "RNR", "MKL", "RE", "AXS", "BRO",
    "CNA", "HCC", "ORI", "L", "UNM", "GNW", "STAN", "WM",  # WaMu delisted
    "LEH",  # Lehman (delisted 2008)
    "BSC",  # Bear Stearns (acquired 2008)
    "MER",  # Merrill (acquired 2009)
    "ABK", "MBI", "AFC", "AMTD", "ETFC", "LM", "EV", "FITB", "KEY",
    "HBAN", "MTB", "RF", "CMA", "ZION", "BBT", "STI", "SNV", "FHN", "PBCT",
    "NYB", "IFC", "WL", "CBH", "MI",  # Marshall & Ilsley (delisted)
    "CFC",  # Countrywide (delisted 2008)
    "NCC",  # National City (delisted 2008)
    "FNM",  # Fannie Mae (delisted 2008 → OTC)
    "FRE",  # Freddie Mac (delisted 2008 → OTC)

    # --- Energy: Majors / E&P / Services (XLE, XOP, OIH) ---
    "XOM", "CVX", "COP", "OXY", "MRO", "HES", "APA", "DVN", "APC", "EOG",
    "CHK", "XTO",  # XTO delisted 2010
    "PXD", "CAM", "NBL", "RRC", "STR", "WLL", "CLR", "COG", "QEP", "SWN",
    "EQT", "BEXP", "CRZO", "MUR", "DNR", "CXO", "OAS",
    "SLB", "HAL", "BHI", "NOV", "FTI", "PDE",  # Pride Intl
    "RIG", "ESV", "NE", "ATW", "RDC", "DO", "PTEN", "NBR", "BJS",  # BJ Services delisted
    "SII",  # Smith Intl delisted 2010
    "WFT", "HERO", "CFW", "PKD", "HP", "UNT", "VLO", "TSO", "MPC", "PSX",
    "SUN",  # Sunoco-legacy
    "HOC", "FTO",  # Frontier Oil delisted
    "WNR", "TGP", "WMB", "KMI", "EPD", "ENB", "EP",  # El Paso delisted 2012

    # --- Healthcare / Pharma / Biotech / Med Devices (XLV, IBB) ---
    "JNJ", "PFE", "MRK", "BMY", "ABT", "LLY", "WYE",  # Wyeth delisted 2009 (→PFE)
    "SGP",  # Schering-Plough delisted 2009 (→MRK)
    "AZN", "GSK", "NVS", "SNY", "FRX",  # Forest Labs delisted 2014
    "MYL", "TEVA", "WPI", "PRGO", "ENDP", "AGN", "ALXN", "BIIB", "CELG",
    "AMGN", "GILD", "VRTX", "REGN", "ALKS", "INCY", "JAZZ", "MDCO",
    "ILMN", "LIFE", "AFFX", "SIAL", "PKI", "MTD", "WAT", "BIO", "TECH",
    "BMET",  # Biomet delisted 2007
    "MDT", "SYK", "BSX", "ZMH", "STJ", "BDX", "BAX", "CAH", "MCK", "ABC",
    "HSIC", "PDCO", "CERN", "MDRX", "QSII", "CYH", "HCA", "THC", "UHS",
    "LPNT", "HMA", "HLS",  # HealthSouth
    "UNH", "CI", "AET", "HUM", "WLP",  # WellPoint (became ANTM)
    "HNT",  # Health Net
    "MOH", "CNC", "GTS", "DHR", "COV", "ISRG", "VAR", "HOLX",
    "RMD", "ABMD", "XRAY", "RGEN", "EW", "NVO", "CFN",

    # --- Consumer Discretionary: Retail / Autos / Media (XLY, XRT, RTH, XHB) ---
    "HD", "LOW", "TGT", "COST", "KR", "SWY",  # Safeway delisted 2015
    "BBY", "SPLS", "ODP", "OMX",  # OfficeMax delisted 2013
    "TJX", "ROST", "KSS", "JCP", "M", "JWN", "DDS", "BIG", "DLTR", "FDO",
    "BJ",  # BJ's Wholesale delisted 2011
    "RSH", "ANF", "LTD", "GPS", "URBN", "AEO", "TLB",  # Talbots delisted 2012
    "CHS", "CATO", "ZUMZ", "PSUN",  # Pacific Sunwear
    "FL", "PLCE", "WTSLA", "DEST", "CWTR", "BEBE", "DLIA",
    "SBUX", "MCD", "YUM", "CMG", "DRI", "EAT", "BJRI", "DIN", "PNRA",
    "TXRH", "BWLD", "DPZ", "SONC", "RT",  # Ruby Tuesday
    "F", "GM", "HOG", "JCI", "LEA", "BWA", "DLPH", "TEN", "ALV", "TRW",
    "VC", "TKR", "SNA", "WHR", "LEG", "MHK", "NVR", "LEN", "DHI", "PHM",
    "TOL", "KBH", "MDC", "RYL", "MTH", "BZH", "HOV", "WCI", "SPF",
    "DIS", "CMCSA", "CMCSK", "VIA", "VIAB", "TWX",  # Time Warner (became WBD)
    "CBS", "NWS", "NWSA", "DISH", "DTV",  # DirecTV delisted 2015
    "SIRI", "XMSR", "TRIP", "LVS", "WYNN", "MGM", "HET",
    "ISLE", "CZR", "PENN", "BYI", "WMS", "GTK", "SGMS", "IGT",

    # --- Consumer Staples (XLP) ---
    "PG", "KO", "PEP", "KMB", "CL", "CHD", "CLX", "EL", "AVP",
    "MO", "PM", "LO", "RAI",  # Reynolds
    "STZ", "DEO", "TAP", "BFB", "BUD",
    "GIS", "K", "CPB", "CAG", "MKC", "HRL", "SJM", "TSN", "ADM", "BG",
    "KFT",  # Kraft pre-split 2012
    "SLE",  # Sara Lee delisted 2012
    "WAG", "CVS", "RAD", "WFM", "SVU", "DLM", "HNZ",  # Heinz delisted 2013
    "DF", "POST", "FLO", "MJN", "ENR", "NWL", "RBI",

    # --- Industrials / Defense / Airlines / Rails / Machinery (XLI, IYT) ---
    "BA", "GE", "HON", "MMM", "UTX", "LMT", "RTN", "GD", "NOC", "COL",
    "LLL", "TXT", "PCP", "HEI", "TDG", "BWXT", "GY",
    "CAT", "DE", "CMI", "PCAR", "NAV", "PH", "ETN", "EMR", "ROK", "ROP",
    "DOV", "ITT", "HUBB", "AME", "APH", "ITW", "TYC",
    "LUK", "FLS", "PNR", "WTS", "LII", "AOS", "MLM", "VMC", "EXP", "SUM",
    "UNP", "CSX", "NSC", "BNI",  # BNSF (delisted 2010, acquired by BRK)
    "KSU", "CP", "CNI", "GWR", "JBHT", "LSTR", "ODFL", "CHRW", "EXPD",
    "FDX", "UPS",
    "AMR",  # pre-merger American Airlines (delisted 2013)
    "UAUA",  # pre-merger UAL (delisted 2010)
    "CAL",  # pre-merger Continental (delisted 2010)
    "DAL", "NWAC",  # Northwest Airlines
    "LCC",  # US Airways (delisted 2013)
    "LUV", "JBLU", "ALK", "SKYW", "R", "DSX", "EXM", "OSG", "TNK",
    "GNK", "DRYS", "EGLE", "BALT", "NM", "FRO",

    # --- Utilities (XLU, UTH) ---
    "NEE", "DUK", "SO", "D", "EXC", "AEP", "XEL", "PPL", "ETR", "PCG",
    "ED", "SRE", "PEG", "EIX", "FE", "AEE", "CMS", "CNP", "DTE", "NI",
    "NRG", "WEC", "SCG", "PNW", "POM", "VVC", "ALE", "IDA", "MGEE", "OGE",
    "PNM", "WR", "BKH", "HE", "ITC", "UIL", "UNS", "AVA", "WGL", "LG",
    "SJI", "NJR", "NWN", "PNY", "SWX", "NSTR", "CPN",

    # --- Materials / Metals / Chemicals (XLB, XME, GDX) ---
    "DD", "DOW", "MON", "POT", "MOS", "CF", "AGU", "IPI", "SYT", "FMC",
    "EMN", "IFF", "PPG", "SHW", "RPM", "CBT", "ALB", "FUL", "LYB", "WLK",
    "HUN", "CC", "OLN", "ARG", "APD", "PX", "AA",  # Alcoa pre-split
    "CENX", "KALU", "X", "NUE", "STLD", "MT", "ATI", "CRS", "CMC", "HAYN",
    "ROCK", "HSC", "AKS", "WOR", "BLL", "IP", "WY", "PKG", "BZ",
    "SON", "GEF", "LPX", "BCC", "DEL", "CDE", "NEM", "ABX", "GG", "AUY",
    "KGC", "IAG", "EGO", "HMY", "FCX", "TC", "PAAS", "SSRI", "AG", "HL",
    "SLW", "RGLD",

    # --- Real Estate REITs (IYR) ---
    "SPG", "PSA", "AVB", "EQR", "BXP", "VNO", "HCN", "HCP", "VTR", "O",
    "NNN", "REG", "SLG", "DRE", "FRT", "KIM", "DDR", "GGP", "PEI",
    "MAC", "TCO", "ESS", "CPT", "UDR", "AIV", "EQY", "RYN", "PLD", "AMB",
    "DLR", "MAA", "EXR", "PSB", "CUZ", "HIW", "BRE", "AHT", "LHO", "HST",
    "DRH", "SHO", "BEE", "FCH", "IHR", "FR",

    # --- Telecom / Communications (IYZ) ---
    "T", "VZ", "CTL", "S",  # Sprint
    "BLS",  # BellSouth (delisted 2006, acquired by AT&T)
    "SBC",  # SBC Communications (became T in 2005)
    "Q",  # Qwest (delisted 2011, acquired by CTL)
    "FTR", "WIN", "TDS", "USM", "LVLT", "CNSL", "GNCMA", "ALSK",
    "CTB", "CBB", "PAET", "IPG", "OMC", "MDP", "WPO", "NYT", "GCI",
    "LEE", "JRN", "MNI",

    # --- Misc large-caps, conglomerates ---
    "GWW", "FAST", "MSM",
    "HRS", "HAR", "GRMN", "TRMB", "CGNX", "ROG", "IEX",
]
# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT_TICKERS = PAPER_TICKERS

SECTOR_TO_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Real Estate": "IYR",
    "Utilities": "UTH",
    "Basic Materials": "XLI",
    "Communication Services": "XLK",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Materials": "XLB",
    "Semiconductors": "SMH",
    "Telecom": "IYZ",
    "Telecommunications": "IYZ",
}

DATA_SOURCES = ["yfinance", "crsp"]


@dataclass
class FactorConfig:
    model_type: str = "pca"
    pca_lookback: int = 252
    pca_n_components: Optional[int] = 15
    explained_variance_threshold: float = 0.55
    use_ledoit_wolf: bool = True
    beta_rolling_window: int = 252


@dataclass
class OUConfig:
    estimation_window: int = 60
    kappa_min: float = 8.4
    mean_center: bool = True


@dataclass
class SignalConfig:
    s_bo: float = 1.25
    s_so: float = 1.25
    s_sc: float = 0.50
    s_bc: float = 0.75
    s_limit: float = 4.0


@dataclass
class VolumeConfig:
    enabled: bool = False
    trailing_window: int = 10


@dataclass
class PairsConfig:
    pvalue_threshold: float = 0.10
    max_pairs: int = 20
    min_half_life: float = 1.0
    max_half_life: float = 126.0
    lookback_window: int = 252
    auto_select: bool = True


@dataclass
class BacktestConfig:
    initial_equity: float = 1_000_000.0
    leverage_long: float = 2.0
    leverage_short: float = 2.0
    tc_bps: float = 5.0
    hedge_instrument: str = "SPY"
    risk_free_rate: float = 0.02
    dt: float = 1.0 / 252.0


@dataclass
class Config:
    factor: FactorConfig = field(default_factory=FactorConfig)
    ou: OUConfig = field(default_factory=OUConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    trading_mode: str = "statarb"
    data_source: str = "yfinance"
    start_date: str = "1997-01-01"
    end_date: str = "2007-12-31"
    tickers: List[str] = field(default_factory=lambda: DEFAULT_TICKERS.copy())
