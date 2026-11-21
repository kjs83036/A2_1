"""Vanilla GD와 Momentum을 구현하고 최적화 경로를 비교한다."""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class VanillaGD:
    """현재 기울기를 반대 방향으로 학습률만큼 반영한다."""

    def __init__(self, lr):
        """양의 학습률을 저장한다."""
        if lr <= 0:
            raise ValueError("학습률은 양수여야 합니다.")
        self.lr = lr

    def step(self, point, gradient):
        """p_{t+1} = p_t - lr * grad를 반환한다."""
        return point - self.lr * gradient


class Momentum(VanillaGD):
    """이전 속도를 보관하는 Heavy Ball 방식의 Momentum이다."""

    def __init__(self, lr, beta=0.9):
        """학습률과 관성 계수를 정하고 첫 속도를 0으로 둔다."""
        super().__init__(lr)
        if not 0 <= beta < 1:
            raise ValueError("beta는 0 이상 1 미만이어야 합니다.")
        self.beta = beta
        self.velocity = 0.0

    def step(self, point, gradient):
        """v_{t+1} = beta * v_t + grad, p_{t+1} = p_t - lr * v_{t+1}."""
        self.velocity = self.beta * self.velocity + gradient
        return super().step(point, self.velocity)


def trajectory(optimizer, gradient, start, steps):
    """초기점과 각 업데이트 위치를 순서대로 반환한다."""
    point = np.asarray(start, dtype=float)
    path = [point]
    for _ in range(steps):
        point = optimizer.step(point, gradient(point))
        path.append(point)
    return np.array(path)


def quadratic_loss(points, y_weight=1):
    """점들의 f(x,y) = x² + y_weight*y²를 계산한다."""
    points = np.asarray(points)
    return points[..., 0] ** 2 + y_weight * points[..., 1] ** 2


def sustained_step(values, threshold):
    """마지막 반복까지 임계값 이하를 유지하기 시작한 반복 번호를 구한다."""
    above = np.flatnonzero(np.asarray(values) > threshold)
    step = int(above[-1] + 1) if above.size else 0
    return step if step < len(values) else None


def plot_convergence(paths, y_weight, title, metric="loss", threshold=None):
    """등고선 이동 경로와 반복별 손실(또는 원점 거리)을 나란히 그린다."""
    fig, (space, progress) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    extent = max(np.max(np.abs(path)) for path in paths.values()) * 1.1
    axis = np.linspace(-extent, extent, 301)
    x, y = np.meshgrid(axis, axis)
    levels = extent**2 * np.array([0.005, 0.02, 0.07, 0.2, 0.5, 1, 2, 4])
    space.contour(x, y, quadratic_loss(np.stack((x, y), axis=-1), y_weight),
                  levels=levels, colors="#b9c5cf", linewidths=0.9)
    for (label, path), color in zip(paths.items(), ("#175c9e", "#d54c35", "#158363", "#80299d")):
        space.plot(*path.T, "--", marker="o", markevery=max(1, len(path) // 12),
                   markersize=3, linewidth=1.3, color=color, label=label)
        values = (quadratic_loss(path, y_weight) if metric == "loss"
                  else np.linalg.norm(path, axis=1))
        progress.plot(np.arange(len(path)), values, color=color, label=label)
    space.scatter(0, 0, marker="*", color="#152434", s=90, zorder=5)
    space.set(title="Path on contours", xlabel="x", ylabel="y", aspect="equal")
    progress.set(title="Progress by iteration", xlabel="Iteration",
                 ylabel="Objective f(x, y)" if metric == "loss" else "Distance to origin",
                 yscale="symlog", ylim=(0, None))
    if metric == "loss":
        progress.set_yscale("symlog", linthresh=1e-4)
    if threshold is not None:
        progress.axhline(threshold, color="#70879a", linestyle=":", label=f"Target {threshold:g}")
    progress.grid(alpha=0.25)
    progress.legend()
    fig.suptitle(title)
    return fig


def quadratic_gradient(point, y_weight=1):
    """f(x,y) = x² + y_weight*y²의 해석적 기울기를 계산한다."""
    return 2 * np.array([1, y_weight]) * point


def run_optimization(output_dir):
    """학습률 경계와 공통 초기조건에서의 Momentum 효과를 확인한다."""
    start = [5.0, 5.0]
    target = 1e-2
    for weight, rate, steps, file_name in [
        (1, 0.1, 100, "04_gd_momentum_circle.png"),
        (10, 0.01, 200, "06_gd_momentum_ellipse.png"),
    ]:
        gradient = partial(quadratic_gradient, y_weight=weight)
        paths = {
            "Vanilla GD": trajectory(VanillaGD(rate), gradient, start, steps),
            "Momentum (beta=0.9)": trajectory(Momentum(rate), gradient, start, steps),
        }
        figure = plot_convergence(paths, weight, f"x² + {weight}y² · lr={rate:g}",
                                  threshold=target)
        figure.savefig(output_dir / file_name, dpi=150)
        plt.close(figure)
        reached = {}
        for name, path in paths.items():
            losses = quadratic_loss(path, weight)
            reached[name] = sustained_step(losses, target)
            print(f"w={weight}, lr={rate:g}, {name}: step={steps}, point={path[-1]}, "
                  f"radius={np.linalg.norm(path[-1]):.6e}, "
                  f"loss={losses[-1]:.6e}, sustained step={reached[name]}")
        if weight == 1:
            assert np.linalg.norm(paths["Vanilla GD"][-1]) <= 0.1
            print("Vanilla GD after 100 steps is within radius 0.1 [PASS]")
        else:
            assert reached["Momentum (beta=0.9)"] < reached["Vanilla GD"]
            print("Momentum reaches and maintains the loss threshold sooner [PASS]")

    paths = {
        f"lr={rate:g}": trajectory(VanillaGD(rate), quadratic_gradient, start, 12)
        for rate in (0.5, 0.75, 1.0, 1.1)
    }
    radii = {label: np.linalg.norm(path, axis=1) for label, path in paths.items()}
    assert radii["lr=0.5"][1] == 0
    assert np.all(np.diff(radii["lr=0.75"]) < 0)
    assert np.allclose(paths["lr=1"][1:], -paths["lr=1"][:-1])
    assert np.all(np.diff(radii["lr=1.1"]) > 0)
    for label, radius in radii.items():
        print(f"Circle {label}: r0={radius[0]:.6f}, r1={radius[1]:.6f}, "
              f"r12={radius[-1]:.6f} [PASS]")
    figure = plot_convergence(paths, 1, "Learning rate on x² + y²", metric="radius")
    figure.savefig(output_dir / "05_learning_rate.png", dpi=150)
    plt.close(figure)

    # 원본 8쪽의 학습률도 별도로 실행해 비교가 설정에 의존함을 확인한다.
    for optimizer in (VanillaGD(0.05), Momentum(0.05)):
        path = trajectory(optimizer, partial(quadratic_gradient, y_weight=10), start, 100)
        losses = quadratic_loss(path, 10)
        print(f"w=10, lr=0.05, {type(optimizer).__name__}: "
              f"loss={losses[-1]:.6e}, sustained step={sustained_step(losses, target)}")


if __name__ == "__main__":
    plt.switch_backend("Agg")
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)
    run_optimization(output_dir)
