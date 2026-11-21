"""중심차분과 등고선 위 gradient."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def central_difference(function, x, h=1e-5):
    """[f(x+h) - f(x-h)] / (2h)로 1변수 함수의 도함수를 근사한다."""
    if h <= 0:
        raise ValueError("h는 양수여야 합니다.")
    return (function(x + h) - function(x - h)) / (2 * h)


def plot_gradient():
    """원형 등고선에 gradient와 접선을 표시하고 최대 내적 오차를 반환한다."""
    x, y = np.meshgrid(np.linspace(-3, 3, 241), np.linspace(-3, 3, 241))
    radii = np.array([1.0, 2.0])
    angles = np.arange(8) * np.pi / 4
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    points = (radii[:, None, None] * directions).reshape(-1, 2)
    gradients = 2 * points
    tangents = np.column_stack((-points[:, 1], points[:, 0]))
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    dot = np.sum(gradients * tangents, axis=1)

    fig, ax = plt.subplots(figsize=(6.3, 6), layout="constrained")
    contours = ax.contour(x, y, x**2 + y**2, levels=[1, 4, 6.25], colors="#70879a")
    ax.clabel(contours, fmt="f = %g", fontsize=9)
    ax.quiver(*points.T, *gradients.T, angles="xy", scale_units="xy", scale=6,
              color="#d54c35", label="Gradient (arrows scaled)")
    for i, (point, tangent) in enumerate(zip(points, tangents)):
        segment = point + np.array([[-0.22], [0.22]]) * tangent
        ax.plot(*segment.T, color="#175c9e", label="Local tangent" if i == 0 else None)
    ax.set(title="Gradient is normal to the contour", xlabel="x", ylabel="y",
           xlim=(-3, 3), ylim=(-3, 3), aspect="equal")
    ax.legend(loc="upper right")
    return fig, float(np.max(np.abs(dot)))


def run_calculus(output_dir):
    """중심차분 및 등고선에 대한 gradient의 직교성을 확인한다."""
    derivative = central_difference(np.square, 3.0)
    error = abs(derivative - 6.0)
    assert error <= 1e-4
    figure, perpendicular_error = plot_gradient()
    assert perpendicular_error <= 1e-12
    figure.savefig(output_dir / "03_gradient.png", dpi=150)
    plt.close(figure)
    print(f"Central difference at x=3: {derivative:.10f}, error={error:.3e} [PASS]")
    print(f"Maximum |gradient dot tangent|={perpendicular_error:.3e} [PASS]")


if __name__ == "__main__":
    plt.switch_backend("Agg")
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)
    run_calculus(output_dir)
