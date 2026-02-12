"""Monopoly pricing with heterogeneous boycott selection.
1) solve optimal monopoly pricing across boycott scenarios
2) sweep boycott intensity beta and write clean CSV outputs
3) generate publication-quality figures

Notes:
- Hardware tested on: Macbook Pro (M5, 24GB RAM, 2025)
- Last updated: Dec 2025

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import lognorm


EPS = 1e-12


@dataclass(frozen=True)
class ModelParams:
    mu: float = 0.0
    sigma: float = 0.7
    market_size: float = 1.0
    marginal_cost: float = 0.4
    v_floor: float = 1e-4
    v_max: float = 10.0
    valuation_grid_size: int = 12000
    price_grid_size: int = 2600


@dataclass(frozen=True)
class SurvivalScenario:
    key: str
    label: str
    color: str
    function: Callable[[np.ndarray, float], np.ndarray]


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#D5D9E3",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D8DDE8",
            "grid.alpha": 0.65,
            "axes.facecolor": "#FBFCFF",
            "figure.facecolor": "white",
        }
    )


def lognormal_cdf(v: np.ndarray | float, params: ModelParams) -> np.ndarray:
    x = np.maximum(np.asarray(v, dtype=float), EPS)
    return lognorm.cdf(x, s=params.sigma, scale=np.exp(params.mu))


def lognormal_pdf(v: np.ndarray | float, params: ModelParams) -> np.ndarray:
    x = np.maximum(np.asarray(v, dtype=float), EPS)
    return lognorm.pdf(x, s=params.sigma, scale=np.exp(params.mu))


def baseline_demand(p: np.ndarray | float, params: ModelParams) -> np.ndarray:
    return params.market_size * (1.0 - lognormal_cdf(p, params))


def logistic_in_log_space(v: np.ndarray | float, slope: float, pivot: float) -> np.ndarray:
    x = np.maximum(np.asarray(v, dtype=float), EPS)
    return 1.0 / (1.0 + np.exp(-slope * (np.log(x) - np.log(pivot))))


def survival_left(v: np.ndarray | float, beta: float, slope: float = 3.0, pivot: float = 1.0) -> np.ndarray:
    g = logistic_in_log_space(v, slope=slope, pivot=pivot)
    return 1.0 - beta * (1.0 - g)


def survival_right(v: np.ndarray | float, beta: float, slope: float = 3.0, pivot: float = 1.0) -> np.ndarray:
    g = logistic_in_log_space(v, slope=slope, pivot=pivot)
    return 1.0 - beta * g


def survival_middle(
    v: np.ndarray | float, beta: float, slope: float = 3.0, low_pivot: float = 0.8, high_pivot: float = 1.6
) -> np.ndarray:
    low = logistic_in_log_space(v, slope=slope, pivot=low_pivot)
    high = logistic_in_log_space(v, slope=slope, pivot=high_pivot)
    window = low * (1.0 - high)
    return 1.0 - beta * window


def build_boycott_demand_lookup(
    params: ModelParams, beta: float, survival_function: Callable[[np.ndarray, float], np.ndarray]
) -> Callable[[np.ndarray | float], np.ndarray]:
    v_grid = np.linspace(params.v_floor, params.v_max, params.valuation_grid_size)
    density = lognormal_pdf(v_grid, params)
    survival = np.clip(survival_function(v_grid, beta), 0.0, 1.0)
    integrand = survival * density

    dv = np.diff(v_grid)
    trapezoids = 0.5 * (integrand[:-1] + integrand[1:]) * dv
    tail_integral = np.zeros_like(v_grid)
    tail_integral[:-1] = np.cumsum(trapezoids[::-1])[::-1]
    tail_quantity = params.market_size * tail_integral

    def demand(price: np.ndarray | float) -> np.ndarray:
        p = np.asarray(price, dtype=float)
        clipped = np.clip(p, v_grid[0], v_grid[-1])
        q = np.interp(clipped, v_grid, tail_quantity)
        q = np.where(p > v_grid[-1], 0.0, q)
        return q

    return demand


def elasticity_at_price(
    price: float,
    demand_function: Callable[[np.ndarray | float], np.ndarray],
    params: ModelParams,
    beta: float | None = None,
    survival_function: Callable[[np.ndarray, float], np.ndarray] | None = None,
) -> float:
    q = float(np.maximum(demand_function(price), EPS))
    if beta is None or survival_function is None:
        d_q_dp = -params.market_size * float(lognormal_pdf(price, params))
    else:
        local_survival = float(np.clip(survival_function(np.array([price]), beta)[0], 0.0, 1.0))
        d_q_dp = -params.market_size * local_survival * float(lognormal_pdf(price, params))
    return float((d_q_dp * price) / q)


def solve_optimal_price(
    params: ModelParams,
    demand_function: Callable[[np.ndarray | float], np.ndarray],
    beta: float | None = None,
    survival_function: Callable[[np.ndarray, float], np.ndarray] | None = None,
) -> dict[str, float]:
    lower = max(params.marginal_cost + 1e-6, params.v_floor)
    upper = max(lower + 1e-3, params.v_max * 0.98)
    p_grid = np.linspace(lower, upper, params.price_grid_size)

    q_grid = np.asarray(demand_function(p_grid), dtype=float)
    profit_grid = (p_grid - params.marginal_cost) * q_grid
    idx = int(np.argmax(profit_grid))
    p_guess = float(p_grid[idx])

    if 0 < idx < len(p_grid) - 1:
        left = float(p_grid[idx - 1])
        right = float(p_grid[idx + 1])

        def objective(p: float) -> float:
            q = float(np.maximum(demand_function(p), 0.0))
            return -((p - params.marginal_cost) * q)

        result = minimize_scalar(objective, bounds=(left, right), method="bounded")
        p_star = float(result.x) if result.success else p_guess
    else:
        p_star = p_guess

    q_star = float(np.maximum(demand_function(p_star), 0.0))
    profit_star = (p_star - params.marginal_cost) * q_star
    elasticity = elasticity_at_price(
        price=p_star,
        demand_function=demand_function,
        params=params,
        beta=beta,
        survival_function=survival_function,
    )
    return {"p_star": p_star, "q_star": q_star, "profit_star": profit_star, "elasticity": elasticity}


def default_scenarios() -> list[SurvivalScenario]:
    return [
        SurvivalScenario(
            key="left_tail",
            label="Left-tail selective (low WTP exits)",
            color="#0A9396",
            function=survival_left,
        ),
        SurvivalScenario(
            key="right_tail",
            label="Right-tail selective (high WTP exits)",
            color="#AE2012",
            function=survival_right,
        ),
        SurvivalScenario(
            key="middle_band",
            label="Middle-band selective",
            color="#5E548E",
            function=survival_middle,
        ),
    ]


def build_point_comparison(params: ModelParams, beta: float, scenarios: list[SurvivalScenario]) -> pd.DataFrame:
    baseline_fn = lambda p: baseline_demand(p, params)
    baseline_solution = solve_optimal_price(params, baseline_fn)
    p0 = baseline_solution["p_star"]

    rows: list[dict[str, float | str]] = [
        {
            "scenario": "baseline",
            "label": "Baseline (no boycott)",
            "beta": 0.0,
            "demand_at_baseline_price": float(baseline_fn(p0)),
            "p_star": baseline_solution["p_star"],
            "q_star": baseline_solution["q_star"],
            "profit_star": baseline_solution["profit_star"],
            "elasticity_at_p_star": baseline_solution["elasticity"],
        }
    ]

    for scenario in scenarios:
        demand_fn = build_boycott_demand_lookup(params, beta=beta, survival_function=scenario.function)
        solution = solve_optimal_price(params, demand_fn, beta=beta, survival_function=scenario.function)
        rows.append(
            {
                "scenario": scenario.key,
                "label": scenario.label,
                "beta": beta,
                "demand_at_baseline_price": float(demand_fn(p0)),
                "p_star": solution["p_star"],
                "q_star": solution["q_star"],
                "profit_star": solution["profit_star"],
                "elasticity_at_p_star": solution["elasticity"],
            }
        )

    out = pd.DataFrame(rows)
    base = out.iloc[0]
    out["delta_p_star"] = out["p_star"] - float(base["p_star"])
    out["delta_q_star"] = out["q_star"] - float(base["q_star"])
    out["delta_profit_star"] = out["profit_star"] - float(base["profit_star"])
    return out


def build_beta_paths(params: ModelParams, beta_grid: np.ndarray, scenarios: list[SurvivalScenario]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    baseline_fn = lambda p: baseline_demand(p, params)
    baseline_solution = solve_optimal_price(params, baseline_fn)

    for beta in beta_grid:
        rows.append(
            {
                "scenario": "baseline",
                "label": "Baseline (no boycott)",
                "beta": float(beta),
                "p_star": baseline_solution["p_star"],
                "q_star": baseline_solution["q_star"],
                "profit_star": baseline_solution["profit_star"],
                "elasticity_at_p_star": baseline_solution["elasticity"],
            }
        )

        for scenario in scenarios:
            demand_fn = build_boycott_demand_lookup(params, beta=float(beta), survival_function=scenario.function)
            solution = solve_optimal_price(
                params,
                demand_fn,
                beta=float(beta),
                survival_function=scenario.function,
            )
            rows.append(
                {
                    "scenario": scenario.key,
                    "label": scenario.label,
                    "beta": float(beta),
                    "p_star": solution["p_star"],
                    "q_star": solution["q_star"],
                    "profit_star": solution["profit_star"],
                    "elasticity_at_p_star": solution["elasticity"],
                }
            )

    return pd.DataFrame(rows)


def plot_survival_profiles(params: ModelParams, scenarios: list[SurvivalScenario], figure_dir: Path) -> None:
    v = np.linspace(0.25, 3.5, 650)
    beta_values = [0.2, 0.5, 0.8]
    beta_colors = ["#3A86FF", "#FF7F11", "#8338EC"]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 4.4), sharey=True)
    for ax, scenario in zip(axes, scenarios):
        for beta, color in zip(beta_values, beta_colors):
            survival = np.clip(scenario.function(v, beta), 0.0, 1.0)
            ax.plot(v, survival, lw=2.3, color=color, label=f"beta = {beta:.1f}")

        ax.set_title(scenario.label)
        ax.set_xlabel("Valuation v")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(v.min(), v.max())
    axes[0].set_ylabel("Survival probability s(v; beta)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Boycott Selection Shapes by Valuation", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "survival_profiles.png", bbox_inches="tight")
    plt.close(fig)


def plot_demand_and_profit(
    params: ModelParams,
    beta: float,
    scenarios: list[SurvivalScenario],
    figure_dir: Path,
) -> None:
    p = np.linspace(max(params.marginal_cost + 0.01, 0.2), 3.2, 600)
    baseline_fn = lambda x: baseline_demand(x, params)
    baseline_solution = solve_optimal_price(params, baseline_fn)

    scenario_map: list[tuple[str, str, str, np.ndarray, np.ndarray, float, float]] = []
    q_base = np.asarray(baseline_fn(p), dtype=float)
    pi_base = (p - params.marginal_cost) * q_base
    scenario_map.append(
        (
            "Baseline (no boycott)",
            "baseline",
            "#222222",
            q_base,
            pi_base,
            baseline_solution["p_star"],
            baseline_solution["profit_star"],
        )
    )

    for scenario in scenarios:
        demand_fn = build_boycott_demand_lookup(params, beta=beta, survival_function=scenario.function)
        solution = solve_optimal_price(params, demand_fn, beta=beta, survival_function=scenario.function)
        q = np.asarray(demand_fn(p), dtype=float)
        pi = (p - params.marginal_cost) * q
        scenario_map.append(
            (
                scenario.label,
                scenario.key,
                scenario.color,
                q,
                pi,
                solution["p_star"],
                solution["profit_star"],
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    for label, key, color, q, _, p_star, _ in scenario_map:
        style = "--" if key == "baseline" else "-"
        axes[0].plot(p, q, lw=2.4 if key != "baseline" else 2.0, color=color, linestyle=style, label=label)
        q_star = np.interp(p_star, p, q)
        axes[0].scatter([p_star], [q_star], color=color, s=36, zorder=5)

    for label, key, color, _, pi, p_star, pi_star in scenario_map:
        style = "--" if key == "baseline" else "-"
        axes[1].plot(p, pi, lw=2.4 if key != "baseline" else 2.0, color=color, linestyle=style, label=label)
        axes[1].scatter([p_star], [pi_star], color=color, s=36, zorder=5)

    axes[0].set_title("Residual Demand Curves")
    axes[0].set_xlabel("Price p")
    axes[0].set_ylabel("Quantity Q(p)")

    axes[1].set_title("Profit Curves")
    axes[1].set_xlabel("Price p")
    axes[1].set_ylabel("Profit pi(p)")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Demand and Profit Under Boycott Intensity beta = {beta:.2f}", y=1.08, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "demand_profit_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_optimal_paths(paths: pd.DataFrame, scenarios: list[SurvivalScenario], figure_dir: Path) -> None:
    color_map = {"baseline": "#222222"}
    for scenario in scenarios:
        color_map[scenario.key] = scenario.color

    metric_specs = [
        ("p_star", "Optimal price p*"),
        ("q_star", "Optimal quantity Q*"),
        ("profit_star", "Optimal profit pi*"),
        ("elasticity_at_p_star", "Elasticity at optimum"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2), sharex=True)
    axes_flat = axes.ravel()

    for ax, (metric, title) in zip(axes_flat, metric_specs):
        for key, group in paths.groupby("scenario"):
            group = group.sort_values("beta")
            label = group["label"].iloc[0]
            linestyle = "--" if key == "baseline" else "-"
            ax.plot(
                group["beta"],
                group[metric],
                lw=2.2 if key != "baseline" else 2.0,
                linestyle=linestyle,
                color=color_map[key],
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("Boycott intensity beta")
        ax.grid(alpha=0.55)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Comparative Statics Paths Across Boycott Intensity", y=1.06, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "optimal_paths.png", bbox_inches="tight")
    plt.close(fig)


def plot_price_quantity_path(paths: pd.DataFrame, scenarios: list[SurvivalScenario], figure_dir: Path) -> None:
    color_map = {"baseline": "#222222"}
    for scenario in scenarios:
        color_map[scenario.key] = scenario.color

    fig, ax = plt.subplots(figsize=(8.4, 6.0))

    for key, group in paths.groupby("scenario"):
        group = group.sort_values("beta")
        label = group["label"].iloc[0]
        color = color_map[key]

        if key == "baseline":
            p = float(group["p_star"].iloc[0])
            q = float(group["q_star"].iloc[0])
            ax.scatter([q], [p], color=color, s=75, marker="D", label=label, zorder=6)
            ax.annotate("baseline", (q, p), textcoords="offset points", xytext=(8, 7), fontsize=10, color=color)
            continue

        ax.plot(group["q_star"], group["p_star"], color=color, lw=2.5, label=label)
        ax.scatter(group["q_star"].iloc[0], group["p_star"].iloc[0], color=color, s=34, zorder=5)
        ax.scatter(group["q_star"].iloc[-1], group["p_star"].iloc[-1], color=color, s=58, marker="X", zorder=6)

        n = len(group)
        mid = max(2, n // 2)
        x0, y0 = float(group["q_star"].iloc[mid - 1]), float(group["p_star"].iloc[mid - 1])
        x1, y1 = float(group["q_star"].iloc[mid]), float(group["p_star"].iloc[mid])
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8})

    ax.set_xlabel("Optimal quantity Q*")
    ax.set_ylabel("Optimal price p*")
    ax.set_title("Profit-Maximizing Quantity-Price Path by Boycott Intensity")
    ax.grid(alpha=0.58)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "price_quantity_path.png", bbox_inches="tight")
    plt.close(fig)


def print_compact_table(df: pd.DataFrame) -> None:
    show = df.copy()
    numeric_cols = [c for c in show.columns if c not in {"scenario", "label"}]
    for col in numeric_cols:
        show[col] = show[col].astype(float).round(5)
    print("\nComparative table at chosen beta\n")
    print(show.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended monopoly-boycott model with plots and beta sweeps.")
    parser.add_argument("--beta", type=float, default=0.60, help="Point boycott intensity used in comparison charts.")
    parser.add_argument("--beta-max", type=float, default=0.90, help="Maximum beta used in comparative statics sweep.")
    parser.add_argument("--beta-grid-size", type=int, default=31, help="Number of beta points in sweep (includes 0).")
    parser.add_argument("--mu", type=float, default=0.0, help="Mean of log valuation.")
    parser.add_argument("--sigma", type=float, default=0.7, help="Std dev of log valuation.")
    parser.add_argument("--market-size", type=float, default=1.0, help="Total market size M.")
    parser.add_argument("--cost", type=float, default=0.4, help="Constant marginal cost c.")
    parser.add_argument("--figure-dir", type=str, default="figures", help="Output folder for generated figures.")
    parser.add_argument("--results-dir", type=str, default="results", help="Output folder for CSV summaries.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plot_style()

    params = ModelParams(
        mu=args.mu,
        sigma=args.sigma,
        market_size=args.market_size,
        marginal_cost=args.cost,
    )
    scenarios = default_scenarios()

    figure_dir = Path(args.figure_dir)
    results_dir = Path(args.results_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_point_comparison(params=params, beta=args.beta, scenarios=scenarios)
    comparison_path = results_dir / f"comparison_beta_{args.beta:.2f}.csv"
    comparison.to_csv(comparison_path, index=False)
    print_compact_table(comparison)

    beta_grid = np.linspace(0.0, args.beta_max, args.beta_grid_size)
    paths = build_beta_paths(params=params, beta_grid=beta_grid, scenarios=scenarios)
    path_out = results_dir / "optimal_paths.csv"
    paths.to_csv(path_out, index=False)

    if not args.skip_plots:
        plot_survival_profiles(params=params, scenarios=scenarios, figure_dir=figure_dir)
        plot_demand_and_profit(params=params, beta=args.beta, scenarios=scenarios, figure_dir=figure_dir)
        plot_optimal_paths(paths=paths, scenarios=scenarios, figure_dir=figure_dir)
        plot_price_quantity_path(paths=paths, scenarios=scenarios, figure_dir=figure_dir)

    print("\nWrote files:")
    print(f"- {comparison_path}")
    print(f"- {path_out}")
    if not args.skip_plots:
        print(f"- {figure_dir / 'survival_profiles.png'}")
        print(f"- {figure_dir / 'demand_profit_comparison.png'}")
        print(f"- {figure_dir / 'optimal_paths.png'}")
        print(f"- {figure_dir / 'price_quantity_path.png'}")


if __name__ == "__main__":
    main()
