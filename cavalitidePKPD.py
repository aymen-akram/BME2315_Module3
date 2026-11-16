import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import curve_fit, brentq

# Receptor-binding parameters
Kd1 = 0.17       # nM
GLP1 = 0.015     # nM

# Pharmacokinetic parameters
Favailable = 0.01
ka = 0.006      # 1/h
ke = 0.0039     # 1/h
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01

# Dosing and simulation settings
baseline_dose = 21.0   # mg
interval = 24.0       # h
n_doses = 100
dt = 0.1              # h between output samples

folder = Path(__file__).resolve().parent


def Fact(concentration, Kd2):
    """Fraction of active receptors at a free drug concentration in nM."""
    return (Kd2 * GLP1 + Kd1 * concentration) / (
        Kd1 * Kd2 + Kd2 * GLP1 + Kd1 * concentration
    )


def fit_Kd():
    """Fit Cavalitide's dissociation constant."""
    data = np.loadtxt(
        folder / "cavalitide_data.csv",
        delimiter=",",
        skiprows=1
    )

    parameters, _ = curve_fit(
        Fact,
        data[:, 0],
        data[:, 1],
        p0=[0.1],
        bounds=(0, np.inf)
    )

    return float(parameters[0])


def ODEfunc(t, y):
    depot, blood = y

    d_depot = -ka * depot
    d_blood = ka * depot / V - ke * blood

    return [d_depot, d_blood]


def simulate(dose):
    """Return time in hours and free blood concentration in nM."""
    times = []
    concentrations = []
    state = np.array([0.0, 0.0])

    for i in range(n_doses):
        state[0] += Favailable * dose

        start = i * interval
        end = start + interval
        time_points = np.linspace(
            start, end, int(round(interval / dt)) + 1
        )

        solution = solve_ivp(
            ODEfunc,
            (start, end),
            state,
            t_eval=time_points,
            rtol=1e-9,
            atol=1e-11
        )

        if not solution.success:
            raise RuntimeError(solution.message)

        # Avoid repeating timestamps at dosing boundaries.
        keep = slice(None) if i == 0 else slice(1, None)

        times.append(solution.t[keep])
        concentrations.append(solution.y[1, keep])
        state = solution.y[:, -1].copy()

    time = np.concatenate(times)
    blood_mg_L = np.concatenate(concentrations)

    total_nM = blood_mg_L * 1e6 / MW
    free_nM = Ffree * total_nM

    return time, free_nM


def final_cycle_stats(time, values):
    """Mean, minimum, and maximum during the final dosing interval."""
    mask = time >= time[-1] - interval
    cycle_time = time[mask]
    cycle_values = values[mask]

    mean = trapezoid(cycle_values, cycle_time) / (
        cycle_time[-1] - cycle_time[0]
    )

    return mean, cycle_values.min(), cycle_values.max()


def main():
    Kd2 = fit_Kd()

    # Simulate the initial 21 mg daily regimen.
    time, free_baseline = simulate(baseline_dose)
    activation_baseline = Fact(free_baseline, Kd2)

    # Optimize dose for 90% average activation during days 99–100.
    # Concentration scales with dose in this linear PK model.
    def target_error(dose):
        free_concentration = free_baseline * dose / baseline_dose
        activation = Fact(free_concentration, Kd2)
        mean_activation = final_cycle_stats(time, activation)[0]
        return mean_activation - 0.90

    optimized_dose = brentq(target_error, 0.0, 100.0)

    # Simulate the optimized regimen.
    time_opt, free_optimized = simulate(optimized_dose)
    activation_optimized = Fact(free_optimized, Kd2)

    print(f"Cavalitide Kd: {Kd2:.8f} nM")
    print(f"Optimized dose: {optimized_dose:.4f} mg every 24 hours")
    print(
        f"Dose increase: "
        f"{100 * (optimized_dose / baseline_dose - 1):.2f}%"
    )

    for label, t, concentration, activation in [
        ("Initial regimen", time, free_baseline, activation_baseline),
        (
            "Optimized regimen",
            time_opt,
            free_optimized,
            activation_optimized
        ),
    ]:
        c_mean, c_min, c_max = final_cycle_stats(t, concentration)
        a_mean, a_min, a_max = final_cycle_stats(t, activation)

        print(f"\n{label}:")
        print(f"  Mean free concentration: {c_mean:.6f} nM")
        print(f"  Concentration range: {c_min:.6f}–{c_max:.6f} nM")
        print(f"  Mean receptor activation: {a_mean:.4%}")
        print(f"  Activation range: {a_min:.4%}–{a_max:.4%}")

    # Plot concentration and activation over 100 days.
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(
        time / 24,
        free_baseline,
        label="21 mg daily"
    )
    axes[0].plot(
        time_opt / 24,
        free_optimized,
        label=f"{optimized_dose:.2f} mg daily"
    )
    axes[0].set_ylabel("Free blood concentration (nM)")
    axes[0].set_title("Cavalitide Pharmacokinetic–Pharmacodynamic Model")

    axes[1].plot(
        time / 24,
        activation_baseline * 100,
        label="21 mg daily"
    )
    axes[1].plot(
        time_opt / 24,
        activation_optimized * 100,
        label=f"{optimized_dose:.2f} mg daily"
    )
    axes[1].axhline(
        90,
        color="gray",
        linestyle="--",
        label="90% activation target"
    )
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Active GLP1 receptors (%)")
    axes[1].set_ylim(0, 100)

    for ax in axes:
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        folder / "Task3_PKPD.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()