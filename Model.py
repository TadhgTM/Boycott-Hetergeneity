import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

mu = 0.0
sigma = 0.7
M = 1.0

def F_lognormal(v):
    v = np.maximum(v, 1e-12)
    z = (np.log(v) - mu) / (sigma * np.sqrt(2.0))
    erf_vec = np.vectorize(erf)
    return 0.5 * (1.0 + erf_vec(z))

def f_lognormal(v):
    v = np.maximum(v, 1e-12)
    z = (np.log(v) - mu) / sigma
    return (1.0 / (v * sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * z**2)

def Q_baseline(p):
    return M * (1.0 - F_lognormal(p))

def s_left(v, beta, k=3.0, v0=1.0):
    v = np.maximum(v, 1e-12)
    g = 1.0 / (1.0 + np.exp(-k*(np.log(v) - np.log(v0))))
    return 1.0 - beta*(1.0 - g)

def s_right(v, beta, k=3.0, v0=1.0):
    v = np.maximum(v, 1e-12)
    g = 1.0 / (1.0 + np.exp(-k*(np.log(v) - np.log(v0))))
    return 1.0 - beta*(g)

def s_middle(v, beta, k=3.0, vL=0.8, vH=1.6):
    v = np.maximum(v, 1e-12)
    a = 1.0 / (1.0 + np.exp(-k*(np.log(v) - np.log(vL))))
    b = 1.0 / (1.0 + np.exp(-k*(np.log(v) - np.log(vH))))
    window = a * (1.0 - b) 
    return 1.0 - beta * window

def Q_boycott(p, beta, sfun, vmax=40.0, ngrid=6000):
    grid = np.linspace(p, vmax, ngrid)
    integrand = sfun(grid, beta) * f_lognormal(grid)
    return M * np.trapezoid(integrand, grid)

def Q_boycott_vec(p_grid, beta, sfun):
    return np.array([Q_boycott(p, beta, sfun) for p in p_grid])

def profit(p, c, beta=None, sfun=None):
    if beta is None:
        q = Q_baseline(p)
    else:
        q = Q_boycott(p, beta, sfun)
    return (p - c) * q

def argmax_price(c, beta=None, sfun=None, pmin=1e-4, pmax=8.0, n=4000):
    p_grid = np.linspace(pmin, pmax, n)
    if beta is None:
        q = Q_baseline(p_grid)
    else:
        q = Q_boycott_vec(p_grid, beta, sfun)
    pi = (p_grid - c) * q
    idx = int(np.argmax(pi))
    return float(p_grid[idx]), float(pi[idx])

def scenario_row(name, c, p0, beta=None, sfun=None):
    if beta is None:
        p_star, pi_star = argmax_price(c, beta=None)
        q_star = Q_baseline(p_star)
        d_at_p0 = Q_baseline(p0)
    else:
        p_star, pi_star = argmax_price(c, beta=beta, sfun=sfun)
        q_star = Q_boycott(p_star, beta, sfun)
        d_at_p0 = Q_boycott(p0, beta, sfun)

    return {
        "Scenario": name,
        "Boycott type": "None" if beta is None else name.split(":")[0],
        "beta": 0.0 if beta is None else beta,
        "D(p0)": d_at_p0,
        "P*": p_star,
        "Q*": q_star,
        "Profit*": pi_star
    }

def print_table(rows):
    cols = ["Scenario","Boycott type","beta","D(p0)","P*","Q*","Profit*","ΔP*","ΔQ*","ΔProfit*"]
    base = rows[0]
    for r in rows:
        r["ΔP*"] = r["P*"] - base["P*"]
        r["ΔQ*"] = r["Q*"] - base["Q*"]
        r["ΔProfit*"] = r["Profit*"] - base["Profit*"]

    # format
    fmt = {
        "beta": "{:.2f}".format,
        "D(p0)": "{:.4f}".format,
        "P*": "{:.4f}".format,
        "Q*": "{:.4f}".format,
        "Profit*": "{:.6f}".format,
        "ΔP*": "{:+.4f}".format,
        "ΔQ*": "{:+.4f}".format,
        "ΔProfit*": "{:+.6f}".format,
    }

    def cell(col, r):
        v = r.get(col, "")
        if col in fmt:
            return fmt[col](v)
        return str(v)

    widths = {c: max(len(c), max(len(cell(c, r)) for r in rows)) for c in cols}

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-"*widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(cell(c, r).ljust(widths[c]) for c in cols))

c = 0.4
beta = 0.6
p0, _ = argmax_price(c, beta=None)

rows = []
rows.append(scenario_row("Baseline: no boycott", c, p0, beta=None))

rows.append(scenario_row("Left-tail: low WTP exit more", c, p0, beta=beta, sfun=s_left))
rows.append(scenario_row("Right-tail: high WTP exit more", c, p0, beta=beta, sfun=s_right))
rows.append(scenario_row("Middle: mid WTP exit more", c, p0, beta=beta, sfun=s_middle))

print("\nComparative table (baseline vs boycott)\n")
print_table(rows)
