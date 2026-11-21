"""행렬 변환, 거듭제곱 반복, 회색조 SVD 압축."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def polygon_area(points):
    """순서대로 나열한 꼭짓점의 면적을 신발끈 공식으로 구한다."""
    x, y = np.asarray(points).T
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def plot_transformations():
    """한 Figure에 단위 원과 변환 결과를 겹쳐 그리고 면적을 비교한다."""
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    transforms = {
        "Rotation (45°)": np.array([[c, -s], [s, c]]),
        "Scaling (2, 0.5)": np.diag([2, 0.5]),
        "Shear (k = 1)": np.array([[1, 1], [0, 1]]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    area = polygon_area(circle)
    stats = {}
    for ax, (name, matrix) in zip(axes, transforms.items()):
        mapped = circle @ matrix.T  # 각 행의 점에 열벡터 규약 A @ p를 적용한다.
        det = float(np.linalg.det(matrix))
        ratio = polygon_area(mapped) / area
        error = abs(ratio - abs(det)) / abs(det)
        stats[name] = det, ratio, error
        ax.plot(*np.vstack((mapped, mapped[0])).T, color="#d54c35", label="After")
        ax.plot(*np.vstack((circle, circle[0])).T, "--", color="#175c9e", label="Before")
        # 원의 회전은 외곽선이 겹치므로 같은 기준점을 양쪽에 표시한다.
        ax.scatter(*circle[0], facecolors="white", edgecolors="#175c9e", zorder=4)
        ax.scatter(*mapped[0], color="#d54c35", zorder=5, s=24)
        ax.set(title=f"{name}\n|det A| = {abs(det):.2f}   area ratio = {ratio:.2f}",
               xlabel="x", ylabel="y", xlim=(-2.25, 2.25), ylim=(-1.65, 1.65),
               aspect="equal")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower left")
    return fig, stats


def power_iteration(matrix, tol=1e-9, max_iter=1000, initial=None):
    """대칭 실수 행렬의 절댓값이 지배적인 고유쌍과 반복 횟수를 구한다.

    지배 고유값의 절댓값이 유일하고 시작 벡터에 해당 방향 성분이 있어야 한다.
    레일리 몫 λ = vᵀAv와 잔차 ||Av - λv||를 사용한다.
    """
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or not matrix.size or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T)):
        raise ValueError("유한한 실수 대칭 정사각 행렬을 입력하세요.")
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol과 max_iter는 양수여야 합니다.")
    vector = (np.random.normal(size=matrix.shape[0]) if initial is None
              else np.asarray(initial, dtype=float))
    if vector.shape != (matrix.shape[0],) or not np.isfinite(vector).all():
        raise ValueError("시작 벡터의 shape 또는 값이 유효하지 않습니다.")
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("시작 벡터는 영벡터일 수 없습니다.")
    vector = vector / length
    if not np.any(matrix):
        return 0.0, vector, 0
    for iteration in range(1, max_iter + 1):
        product = matrix @ vector
        length = np.linalg.norm(product)
        if length == 0:
            raise ValueError("시작 벡터가 행렬의 영공간에 있습니다.")
        vector = product / length
        transformed = matrix @ vector
        eigenvalue = float(vector @ transformed)
        if np.linalg.norm(transformed - eigenvalue * vector) <= tol:
            if vector[np.argmax(np.abs(vector))] < 0:
                vector = -vector
            return eigenvalue, vector, iteration
    raise RuntimeError("최대 반복 횟수 내에 고유쌍에 수렴하지 않았습니다.")


def compress_image(image, rank):
    """회색조 행렬에서 상위 min(rank, min(shape))개 SVD 성분을 보관한다."""
    image = np.asarray(image, dtype=float)
    if (image.ndim != 2 or not image.size or not np.isfinite(image).all()
            or not isinstance(rank, (int, np.integer)) or rank < 1):
        raise ValueError("유한한 2차원 이미지와 양의 정수 rank를 입력하세요.")
    u, singular_values, vt = np.linalg.svd(image, full_matrices=False)
    k = min(rank, singular_values.size)
    return u[:, :k].copy(), singular_values[:k].copy(), vt[:k].copy()


def restore_image(factors):
    """Uₖ diag(sₖ) Vₖᵀ로 이미지를 복원한다."""
    u, singular_values, vt = factors
    return (u * singular_values) @ vt


def plot_svd(image, ranks=(10, 50, 100)):
    """원본과 SVD 복원을 같은 명암 범위로 그리고 오차와 저장량을 반환한다."""
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(12, 3.5), layout="constrained")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original\n{image.shape[0]} × {image.shape[1]}")
    stats = {}
    norm = np.linalg.norm(image)
    for ax, rank in zip(axes[1:], ranks):
        factors = compress_image(image, rank)
        restored = restore_image(factors)
        error = np.linalg.norm(image - restored) / norm if norm else 0.0
        ratio = sum(part.size for part in factors) / image.size
        k = factors[1].size
        stats[rank] = k, float(error), ratio
        ax.imshow(restored, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"k = {rank} · used {k}\nerror {error:.2%} · size {ratio:.1%}", fontsize=10)
    for ax in axes:
        ax.axis("off")
    return fig, stats


def run_linear_algebra(output_dir):
    """면적, 고유쌍, SVD 복원 결과를 그림 및 수치로 확인한다."""
    figure, transformations = plot_transformations()
    figure.savefig(output_dir / "01_transformations.png", dpi=150)
    plt.close(figure)
    for name, (det, ratio, error) in transformations.items():
        assert error <= 0.01, name
        print(f"{name}: det={det:.6f}, ratio={ratio:.6f}, error={error:.3e} [PASS]")

    # 독립 확인: 단위 원(100각형)에 S(2, 0.5)를 적용한 결과를 타원 면적 공식 π·a·b와 비교한다.
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    ellipse_area = polygon_area(circle @ np.diag([2.0, 0.5]).T)
    formula = np.pi * 2.0 * 0.5
    gap = abs(ellipse_area - formula) / formula
    assert gap <= 0.01
    print(f"Scaling (2, 0.5) vs ellipse formula: polygon={ellipse_area:.6f}, "
          f"pi*a*b={formula:.6f}, error={gap:.3e} [PASS]")

    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    value, vector, iterations = power_iteration(matrix)
    # eig는 직접 구현한 결과의 비교에만 사용한다.
    ref_values, ref_vectors = np.linalg.eig(matrix)
    index = np.argmax(np.abs(ref_values))
    error = abs(value - ref_values[index]) / abs(ref_values[index])
    overlap = abs(vector @ ref_vectors[:, index])
    residual = np.linalg.norm(matrix @ vector - value * vector)
    assert error <= 0.05 and 1 - overlap <= 1e-8 and residual <= 1e-9
    print(f"Power Iteration: lambda={value:.10f}, v={vector}, steps={iterations}")
    print(f"eig={ref_values[index]:.10f}, error={error:.3e}, "
          f"|v @ v_ref|={overlap:.10f}, residual={residual:.3e} [PASS]")

    # 저장소에 포함된 회색조 픽셀을 읽어 0~1 범위로 바꾼다.
    image_path = Path(__file__).resolve().parents[1] / "data" / "grace_hopper_gray64.npy"
    image = np.load(image_path) / 255.0
    assert image.shape == (64, 64)
    figure, svd_stats = plot_svd(image)
    figure.savefig(output_dir / "02_svd.png", dpi=150)
    plt.close(figure)
    assert [result[0] for result in svd_stats.values()] == [10, 50, 64]
    errors = [result[1] for result in svd_stats.values()]
    assert errors[0] >= errors[1] >= errors[2] and errors[-1] <= 1e-12
    for request, (used, error, storage) in svd_stats.items():
        print(f"SVD: k={request}, used={used}, relative error={error:.6e}, "
              f"storage ratio={storage:.6f} [PASS]")


if __name__ == "__main__":
    plt.switch_backend("Agg")
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)
    np.random.seed(42)
    run_linear_algebra(output_dir)
