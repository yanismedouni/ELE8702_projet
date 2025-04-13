 # Path Loss Calculation based on 3GPP Models

import math

# Common parameters
hE = 1  # Effective environment height (meters)
W = 20  # Street width (meters)
h = 5   # Building height (meters)
c = 3e8  # Speed of light (m/s)


# -------------------------------------------------
#     RMa-NLOS
# -------------------------------------------------

def rma_nlos(frequency: float, distance: float, h_BS: float, h_UT: float) -> float:
    d_3D = math.sqrt(distance**2 + (h_BS - h_UT)**2)
    pl_los = rma_los(frequency, distance, h_BS, h_UT)  # Using RMa-LOS as part of the calculation

    if 10 <= distance <= 5000:
        pl_prime = (
            161.04 - 7.1 * math.log10(W) + 7.5 * math.log10(h) 
            - (24.37 - 3.7 * (h / h_BS)**2) * math.log10(h_BS)
            + (43.42 - 3.1 * math.log10(h_BS)) * (math.log10(d_3D) - 3) 
            + 20 * math.log10(frequency) - (3.2 * (math.log10(11.75 * h_UT))**2 - 4.97)
        )
        return max(pl_los, pl_prime)
    elif distance < 10: return 0
    elif distance > 5000: return math.inf


# -------------------------------------------------
#     RMa-LOS
# -------------------------------------------------

def rma_los(frequency: float, distance: float, h_BS: float, h_UT: float) -> float:
    d_BP = 2 * math.pi * h_BS * h_UT * frequency * 1e9 / c
    d_3D = math.sqrt(distance**2 + (h_BS - h_UT)**2)

    if 10 <= distance <= d_BP:
        pl1 = (
            20 * math.log10(40 * math.pi * d_3D * frequency / 3) + min(0.03 * h**1.72, 10) * math.log10(d_3D)
            - min(0.044 * h**1.72, 14.77) + 0.002 * math.log10(h) * d_3D
        )
        return pl1
    elif d_BP <= distance <= 10000:
        pl1 = (
            20 * math.log10(40 * math.pi * d_BP * frequency / 3) + min(0.03 * h**1.72, 10) * math.log10(d_BP)
            - min(0.044 * h**1.72, 14.77) + 0.002 * math.log10(h) * d_BP
        )
        pl2 = pl1 + 40 * math.log10(d_3D / d_BP)
        return pl2
    elif distance < 10: return 0
    elif distance > 10000: return math.inf


# -------------------------------------------------
#     UMa-NLOS
# -------------------------------------------------

def uma_nlos(frequency: float, distance: float, h_BS: float, h_UT: float) -> float:
    d_3D = math.sqrt(distance**2 + (h_BS - h_UT)**2)
    if 10 <= distance <= 5000:
        pl_los = uma_los(frequency, distance, h_BS, h_UT)
        pl_prime = 13.54 + 39.08 * math.log10(d_3D) + 20 * math.log10(frequency) - 0.6 * (h_UT - 1.5)
        return max(pl_los, pl_prime)
    elif distance < 10: return 0
    elif distance > 5000: return math.inf


# -------------------------------------------------
#     UMa-LOS
# -------------------------------------------------

def uma_los(frequency: float, distance: float, h_BS: float, h_UT: float) -> float:
    h_BS_prime = h_BS - hE
    h_UT_prime = h_UT - hE
    d_BP_prime = 4 * h_BS_prime * h_UT_prime * frequency * 1e9 / c
    d_3D = math.sqrt(distance**2 + (h_BS - h_UT)**2)

    if 10 <= distance <= d_BP_prime:
        pl1 = 28.0 + 22 * math.log10(d_3D) + 20 * math.log10(frequency)
        return pl1
    elif d_BP_prime <= distance <= 5000:
        pl2 = 28.0 + 40 * math.log10(d_3D) + 20 * math.log10(frequency) - 9 * math.log10((d_BP_prime)**2 + (h_BS - h_UT)**2)
        return pl2
    elif distance < 10: return 0
    elif distance > 5000: return math.inf


# -------------------------------------------------
#     UMi-NLOS
# -------------------------------------------------

def umi_nlos(frequency: float, distance: float, h_BS: float, h_UT: float) -> float:
    if 10 <= distance <= 5000:
        d_3D = math.sqrt(distance**2 + (h_BS - h_UT)**2)
        pl_los = umi_los(frequency, distance, h_BS, h_UT)
        pl_prime = 35.3 * math.log10(d_3D) + 22.4 + 21.3 * math.log10(frequency) - 0.3 * (h_UT - 1.5)
        return max(pl_los, pl_prime)
    elif distance < 10: return 0
    elif distance > 5000: return math.inf


# -------------------------------------------------
#     UMi-LOS
# -------------------------------------------------

def umi_los(frequency: float, distance: float, h_BS: float, h_UT: float) -> float:
    h_BS_prime = h_BS - hE
    h_UT_prime = h_UT - hE
    d_BP_prime = 4 * h_BS_prime * h_UT_prime * frequency * 1e9 / c
    d_3D = math.sqrt(distance**2 + (h_BS - h_UT)**2)

    if 10 <= distance <= d_BP_prime:
        pl1 = 32.4 + 21 * math.log10(d_3D) + 20 * math.log10(frequency)
        return pl1
    elif d_BP_prime <= distance <= 5000:
        pl2 = 32.4 + 40 * math.log10(d_3D) + 20 * math.log10(frequency) - 9.5 * math.log10((d_BP_prime)**2 + (h_BS - h_UT)**2)
        return pl2
    elif distance < 10: return 0
    elif distance > 5000: return math.inf