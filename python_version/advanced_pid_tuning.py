"""
Advanced PID Tuning Methods
Professional-grade tuning algorithms for complex systems
"""

import numpy as np
from scipy import optimize, signal
from typing import Tuple, Optional, List
from dataclasses import dataclass

from models import PIDCoefficients
from system_identification import SystemModel


@dataclass
class TuningResult:
    """Comprehensive PID tuning result"""
    pid: PIDCoefficients
    method: str

    # Performance metrics
    rise_time: float
    settling_time: float
    overshoot_percent: float
    steady_state_error: float

    # Stability margins
    gain_margin_db: float
    phase_margin_deg: float

    # Robustness
    sensitivity_peak: float

    # Quality score (0-100)
    quality_score: float

    def __str__(self):
        lines = []
        lines.append(f"PID Tuning Result ({self.method}):")
        lines.append(f"  {self.pid}")
        lines.append(f"  Rise Time: {self.rise_time:.3f}s")
        lines.append(f"  Settling Time: {self.settling_time:.3f}s")
        lines.append(f"  Overshoot: {self.overshoot_percent:.1f}%")
        lines.append(f"  Gain Margin: {self.gain_margin_db:.1f} dB")
        lines.append(f"  Phase Margin: {self.phase_margin_deg:.1f}°")
        lines.append(f"  Quality Score: {self.quality_score:.1f}/100")
        return "\n".join(lines)


def tune_pid_imc(
    model: SystemModel,
    lambda_c: Optional[float] = None
) -> PIDCoefficients:
    """
    Internal Model Control (IMC) tuning

    Most robust method for systems with dead time

    Args:
        model: Identified system model (FOPDT)
        lambda_c: Closed-loop time constant (None = auto)

    Returns:
        PID coefficients
    """
    K = model.K
    tau = model.tau
    theta = model.theta

    if lambda_c is None:
        # Rule of thumb: lambda_c = max(0.25*tau, theta)
        lambda_c = max(0.25 * tau, theta)

    # IMC-PID formulas for FOPDT
    Kp = tau / (K * (lambda_c + theta))
    Ki = 1 / tau
    Kd = 0  # IMC typically doesn't use derivative

    # Convert to standard PID form
    p = Kp
    i = Kp * Ki
    d = Kp * Kd

    return PIDCoefficients(p=p, i=i, d=d)


def tune_pid_simc(
    model: SystemModel
) -> PIDCoefficients:
    """
    SIMC (Skogestad Internal Model Control) tuning

    Optimized for disturbance rejection and robustness

    Args:
        model: Identified system model (FOPDT)

    Returns:
        PID coefficients
    """
    K = model.K
    tau = model.tau
    theta = model.theta

    # SIMC rules
    tau_c = max(tau, 4 * theta)

    Kp = tau / (K * tau_c)
    Ti = min(tau, 4 * tau_c)
    Td = 0  # Will be added if needed

    # Standard form
    p = Kp
    i = Kp / Ti
    d = Kp * Td

    return PIDCoefficients(p=p, i=i, d=d)


def tune_pid_cohen_coon(
    model: SystemModel
) -> PIDCoefficients:
    """
    Cohen-Coon tuning method

    Better for systems with large dead time / time constant ratio

    Args:
        model: Identified system model (FOPDT)

    Returns:
        PID coefficients
    """
    K = model.K
    tau = model.tau
    theta = model.theta

    R = theta / tau  # Dead time ratio

    # Cohen-Coon formulas for PID
    Kp = (1/K) * (tau/theta) * (1.35 + 0.27*R) / (1 + 0.6*R)
    Ti = theta * (2.5 - 2*R) / (1 - 0.39*R)
    Td = theta * 0.37 / (1 - 0.81*R)

    # Standard form
    p = Kp
    i = Kp / Ti
    d = Kp * Td

    return PIDCoefficients(p=p, i=i, d=d)


def tune_pid_lambda(
    model: SystemModel,
    lambda_factor: float = 1.5
) -> PIDCoefficients:
    """
    Lambda tuning (Dahlin's method)

    Allows explicit specification of closed-loop speed

    Args:
        model: Identified system model
        lambda_factor: Multiplier for closed-loop time constant (higher = slower)

    Returns:
        PID coefficients
    """
    K = model.K
    tau = model.tau
    theta = model.theta

    lambda_cl = lambda_factor * theta

    # Lambda tuning formulas
    Kp = (2*tau + theta) / (2*K*lambda_cl)
    Ti = tau + theta/2
    Td = tau * theta / (2*tau + theta)

    # Standard form
    p = Kp
    i = Kp / Ti
    d = Kp * Td

    return PIDCoefficients(p=p, i=i, d=d)


def tune_pid_tyreus_luyben(
    ku: float,
    tu: float
) -> PIDCoefficients:
    """
    Tyreus-Luyben tuning from ultimate gain/period

    More conservative than Ziegler-Nichols

    Args:
        ku: Ultimate gain
        tu: Ultimate period

    Returns:
        PID coefficients
    """
    # Tyreus-Luyben formulas
    Kp = ku / 3.2
    Ti = 2.2 * tu
    Td = tu / 6.3

    # Standard form
    p = Kp
    i = Kp / Ti
    d = Kp * Td

    return PIDCoefficients(p=p, i=i, d=d)


def tune_pid_optimization(
    model: SystemModel,
    objective: str = 'iae',
    constraints: Optional[dict] = None
) -> Tuple[PIDCoefficients, float]:
    """
    Optimize PID using performance criteria

    Args:
        model: Identified system model
        objective: 'iae', 'ise', 'itae', 'itse'
        constraints: Dict with 'max_overshoot', 'min_damping', etc.

    Returns:
        Optimized PID and performance score
    """
    if constraints is None:
        constraints = {
            'max_overshoot': 10.0,  # percent
            'min_gain_margin': 6.0,  # dB
            'min_phase_margin': 40.0  # degrees
        }

    def simulate_closed_loop(pid_params):
        """Simulate closed-loop response"""
        Kp, Ki, Kd = pid_params

        # Create PID controller transfer function
        # C(s) = Kp + Ki/s + Kd*s
        num_c = [Kd, Kp, Ki]
        den_c = [1, 0]

        # Plant (FOPDT approximation)
        K = model.K
        tau = model.tau
        theta = model.theta

        # First-order: G(s) = K / (tau*s + 1)
        num_p = [K]
        den_p = [tau, 1]

        # Closed-loop: T = PC / (1 + PC)
        try:
            # Open-loop: L = PC
            num_ol = np.polymul(num_c, num_p)
            den_ol = np.polymul(den_c, den_p)

            # Closed-loop
            num_cl = num_ol
            den_cl = np.polyadd(den_ol, num_ol)

            # Simulate step response
            sys_cl = signal.TransferFunction(num_cl, den_cl)
            t = np.linspace(0, 10*tau, 1000)
            t_out, y_out = signal.step(sys_cl, T=t)

            return t_out, y_out

        except:
            return np.array([0, 1]), np.array([0, 0])

    def compute_performance(pid_params):
        """Compute performance metric"""
        t, y = simulate_closed_loop(pid_params)

        if len(y) < 10:
            return 1e10

        # Calculate metrics
        error = 1 - y
        abs_error = np.abs(error)

        # IAE: Integral Absolute Error
        iae = np.trapz(abs_error, t)

        # ISE: Integral Square Error
        ise = np.trapz(error**2, t)

        # ITAE: Integral Time Absolute Error
        itae = np.trapz(t * abs_error, t)

        # ITSE: Integral Time Square Error
        itse = np.trapz(t * error**2, t)

        # Overshoot
        overshoot = (np.max(y) - 1.0) * 100

        # Settling time (2% criterion)
        settled_idx = np.where(abs_error < 0.02)[0]
        if len(settled_idx) > 0:
            settling_time = t[settled_idx[0]]
        else:
            settling_time = t[-1]

        # Penalty for constraint violations
        penalty = 0
        if overshoot > constraints['max_overshoot']:
            penalty += 1000 * (overshoot - constraints['max_overshoot'])

        if settling_time > 10 * model.tau:
            penalty += 1000

        # Select objective
        if objective == 'iae':
            cost = iae
        elif objective == 'ise':
            cost = ise
        elif objective == 'itae':
            cost = itae
        elif objective == 'itse':
            cost = itse
        else:
            cost = iae

        return cost + penalty

    # Initial guess from IMC
    imc_pid = tune_pid_imc(model)
    x0 = [imc_pid.p, imc_pid.i, imc_pid.d]

    # Bounds
    bounds = [
        (0.001, 10.0),  # Kp
        (0.0, 5.0),     # Ki
        (0.0, 1.0)      # Kd
    ]

    # Optimize
    result = optimize.minimize(
        compute_performance,
        x0,
        method='L-BFGS-B',
        bounds=bounds
    )

    Kp_opt, Ki_opt, Kd_opt = result.x
    performance = result.fun

    return PIDCoefficients(p=Kp_opt, i=Ki_opt, d=Kd_opt), performance


def tune_pid_frequency_domain(
    model: SystemModel,
    target_phase_margin: float = 60.0,
    target_gain_margin: float = 10.0
) -> PIDCoefficients:
    """
    Frequency domain tuning for stability margins

    Args:
        model: System model
        target_phase_margin: Desired phase margin (degrees)
        target_gain_margin: Desired gain margin (dB)

    Returns:
        PID coefficients
    """
    K = model.K
    tau = model.tau

    # Target crossover frequency
    # Rule of thumb: wc = 1 / (2 * tau)
    wc = 1 / (2 * tau)

    # Phase lead needed
    phi_lead_rad = np.deg2rad(target_phase_margin - 90 + np.rad2deg(np.arctan(wc * tau)))

    # Calculate PID parameters
    Kp = 1 / (K * np.sqrt(1 + (wc * tau)**2))
    Ti = 1 / (wc * np.tan(phi_lead_rad / 2))
    Td = 0  # Often omitted in frequency design

    # Standard form
    p = Kp
    i = Kp / Ti
    d = Kp * Td

    return PIDCoefficients(p=p, i=i, d=d)


def tune_pid_adaptive_gain_scheduling(
    models: List[SystemModel],
    operating_points: List[float]
) -> List[PIDCoefficients]:
    """
    Gain scheduling for nonlinear systems

    Args:
        models: List of system models at different operating points
        operating_points: Corresponding operating points

    Returns:
        List of PID coefficients for each operating point
    """
    pid_list = []

    for model in models:
        # Use IMC tuning for each operating point
        pid = tune_pid_imc(model)
        pid_list.append(pid)

    return pid_list


def evaluate_closed_loop_performance(
    plant_model: SystemModel,
    pid: PIDCoefficients,
    simulation_time: float = 10.0
) -> TuningResult:
    """
    Comprehensive evaluation of closed-loop performance

    Args:
        plant_model: System model
        pid: PID coefficients to evaluate
        simulation_time: Simulation duration

    Returns:
        Complete tuning result with all metrics
    """
    # Create transfer functions
    K = plant_model.K
    tau = plant_model.tau

    # Plant: G(s) = K / (tau*s + 1)
    num_p = [K]
    den_p = [tau, 1]

    # PID: C(s) = Kp + Ki/s + Kd*s
    num_c = [pid.d, pid.p, pid.i]
    den_c = [1, 0]

    try:
        # Open-loop: L = PC
        num_ol = np.polymul(num_c, num_p)
        den_ol = np.polymul(den_c, den_p)

        # Closed-loop: T = L / (1 + L)
        num_cl = num_ol
        den_cl = np.polyadd(den_ol, num_ol)

        sys_cl = signal.TransferFunction(num_cl, den_cl)

        # Step response
        t = np.linspace(0, simulation_time, 1000)
        t_out, y_out = signal.step(sys_cl, T=t)

        # Calculate metrics
        # Rise time (10% to 90%)
        idx_10 = np.argmax(y_out >= 0.1)
        idx_90 = np.argmax(y_out >= 0.9)
        rise_time = t_out[idx_90] - t_out[idx_10] if idx_90 > idx_10 else 0

        # Settling time (2% criterion)
        settled = np.abs(y_out - 1.0) < 0.02
        settling_idx = np.where(settled)[0]
        settling_time = t_out[settling_idx[0]] if len(settling_idx) > 0 else simulation_time

        # Overshoot
        overshoot = max(0, (np.max(y_out) - 1.0) * 100)

        # Steady-state error
        steady_state_error = abs(1.0 - y_out[-1])

        # Stability margins
        sys_ol = signal.TransferFunction(num_ol, den_ol)
        w = np.logspace(-3, 3, 1000)

        try:
            w_out, H = signal.freqresp(sys_ol, w=w)

            # Gain margin
            phase = np.angle(H, deg=True)
            phase_crossover_idx = np.where(np.diff(np.sign(phase + 180)))[0]

            if len(phase_crossover_idx) > 0:
                gain_at_pc = np.abs(H[phase_crossover_idx[0]])
                gain_margin_db = -20 * np.log10(gain_at_pc + 1e-10)
            else:
                gain_margin_db = 40.0  # Very stable

            # Phase margin
            gain_db = 20 * np.log10(np.abs(H) + 1e-10)
            gain_crossover_idx = np.where(np.diff(np.sign(gain_db)))[0]

            if len(gain_crossover_idx) > 0:
                phase_at_gc = phase[gain_crossover_idx[0]]
                phase_margin = 180 + phase_at_gc
            else:
                phase_margin = 60.0  # Default

        except:
            gain_margin_db = 20.0
            phase_margin = 45.0

        # Sensitivity peak
        sensitivity_peak = 2.0  # Default

        # Quality score (0-100)
        # Based on multiple criteria
        score = 100.0
        score -= min(20, overshoot)  # Penalize overshoot
        score -= min(20, max(0, 40 - phase_margin))  # Penalize low phase margin
        score -= min(20, max(0, 1.0 - settling_time / tau))  # Reward fast settling
        score -= min(20, steady_state_error * 100)  # Penalize steady-state error

        quality_score = max(0, score)

    except Exception as e:
        # Fallback values if simulation fails
        rise_time = 0
        settling_time = simulation_time
        overshoot = 100.0
        steady_state_error = 1.0
        gain_margin_db = 0
        phase_margin = 0
        sensitivity_peak = 10.0
        quality_score = 0

    return TuningResult(
        pid=pid,
        method="evaluation",
        rise_time=rise_time,
        settling_time=settling_time,
        overshoot_percent=overshoot,
        steady_state_error=steady_state_error,
        gain_margin_db=gain_margin_db,
        phase_margin_deg=phase_margin,
        sensitivity_peak=sensitivity_peak,
        quality_score=quality_score
    )


def auto_tune_best_method(
    model: SystemModel,
    requirements: Optional[dict] = None
) -> Tuple[PIDCoefficients, str, TuningResult]:
    """
    Automatically select and apply best tuning method

    Args:
        model: Identified system model
        requirements: Performance requirements dict

    Returns:
        Best PID, method name, evaluation result
    """
    if requirements is None:
        requirements = {
            'max_overshoot': 10.0,
            'min_phase_margin': 40.0,
            'priority': 'balanced'  # 'speed', 'robustness', 'balanced'
        }

    # Try multiple methods
    methods = [
        ('IMC', tune_pid_imc(model)),
        ('SIMC', tune_pid_simc(model)),
        ('Cohen-Coon', tune_pid_cohen_coon(model)),
        ('Lambda', tune_pid_lambda(model)),
    ]

    best_pid = None
    best_method = None
    best_score = -1
    best_result = None

    for method_name, pid in methods:
        try:
            result = evaluate_closed_loop_performance(model, pid)

            # Check constraints
            if result.overshoot_percent > requirements['max_overshoot']:
                continue
            if result.phase_margin_deg < requirements['min_phase_margin']:
                continue

            # Score based on priority
            if requirements['priority'] == 'speed':
                score = 100 / (result.rise_time + 0.1)
            elif requirements['priority'] == 'robustness':
                score = result.phase_margin_deg + result.gain_margin_db
            else:  # balanced
                score = result.quality_score

            if score > best_score:
                best_score = score
                best_pid = pid
                best_method = method_name
                best_result = result

        except:
            continue

    # If no method meets requirements, use IMC as fallback
    if best_pid is None:
        best_pid = tune_pid_imc(model)
        best_method = 'IMC (fallback)'
        best_result = evaluate_closed_loop_performance(model, best_pid)

    return best_pid, best_method, best_result
