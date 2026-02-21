# import numpy as np
# from collections import deque
#
#
# class ModelPredictiveControl:
#     def __init__(self):
#         pass
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# (
#     𝑁
# )
# O(N)
# 复杂度的算法，比通用矩阵求解的
# 𝑂
# (
#     𝑁
#     3
# )
# O(N
# 3
# )
# 快得多，且省内存。
#
# 动态内存回避：所有矩阵操作都是手动展开的，避免了
# new / malloc，适合在单片机（STM32 / ESP32）上跑。
#
# 自适应
# Rho：代码中包含了
# RhoAdapter，这是一个高级特性，能根据收敛情况自动调整惩罚因子
# 𝜌
# ρ
# ，提高求解稳定性。
#
# Python
# 通用版
# MPC - ADMM
#
# 为了在
# Python
# 中实现高效且通用的版本（类似你的
# EKF
# 风格），我们不能直接照搬
# C + + 的
# Riccati
# 循环（因为
# Python
# 的
# for 循环很慢）。
#
# 最佳策略是： 使用
# 稀疏矩阵（Sparse
# Matrix） 技术。
# 虽然
# C + + 版用
# Riccati
# 避免了大矩阵，但在
# Python(NumPy / SciPy)
# 中，构建一个稀疏的大矩阵并调用
# splu
# 求解器，利用底层的
# C / Fortran
# 优化，速度反而比你写
# Python
# for 循环做 Riccati 倒推要快得多。
#
# 下面是按照你的
# EKF
# 风格重构的
# 通用
# MPC - ADMM
# 类。它支持：
#
# 通用模型：
# 𝑥
# 𝑘
# +
# 1
# =
# 𝐴
# 𝑥
# 𝑘
# +
# 𝐵
# 𝑢
# 𝑘
# +
# 𝑓
# x
# k + 1
# ​
#
# =Ax
# k
# ​
#
# +Bu
# k
# ​
#
# +f
# (支持线性化后的非线性系统)
#
# 任意约束：输入输出的
# Box
# 约束。
#
# 轨迹跟踪：支持动态更新参考轨迹
# ref。
#
# code
# Python
# download
# content_copy
# expand_less
# import numpy as np
# import scipy.sparse as sparse
# from scipy.sparse.linalg import splu
# import time
#
#
# class MPC_ADMM:
#     def __init__(self, A, B, Q, R, N, rho=1.0, f=None,
#                  u_min=None, u_max=None, x_min=None, x_max=None):
#         """
#         通用 MPC ADMM 求解器
#         模型: x_{k+1} = A x_k + B u_k + f
#         目标: min sum( (x-ref)^T Q (x-ref) + u^T R u )
#         """
#         self.A = A
#         self.B = B
#         self.N = N  # 预测步长
#         self.rho = rho
#
#         # 维度推断
#         self.nx = A.shape[0]
#         self.nu = B.shape[1]
#
#         # 仿射项 f (通常来自于线性化时的常数项), 默认为 0
#         self.f = f if f is not None else np.zeros(self.nx)
#
#         # 权重矩阵
#         self.Q = Q
#         self.R = R
#
#         # 约束 (处理 None)
#         self.u_min = u_min if u_min is not None else -1e10
#         self.u_max = u_max if u_max is not None else 1e10
#         self.x_min = x_min if x_min is not None else -1e10
#         self.x_max = x_max if x_max is not None else 1e10
#
#         # === 离线预计算 (Offline Setup) ===
#         # 对应 C++ 中的 tiny_setup 和 tiny_precompute_and_set_cache
#         self._build_kkt_system()
#
#         # 运行时变量 (Warm Start)
#         self.n_vars = (self.N) * self.nu + (self.N) * self.nx
#         self.z = np.zeros(self.n_vars)  # Slack variable (v, z in C++)
#         self.y = np.zeros(self.n_vars)  # Dual variable (g, y in C++)
#         self.w = np.zeros(self.n_vars)  # Primal variable (x, u in C++)
#
#         # 性能统计
#         self.solve_time = 0
#         self.iterations = 0
#
#     def _build_kkt_system(self):
#         """
#         构建稀疏 KKT 系统。
#         在 C++ TinyMPC 中，这一步对应 Riccati 增益的预计算。
#         在 Python 中，我们直接构建稀疏矩阵并做 LU 分解。
#         """
#         # 1. 构建 Hessian (代价函数矩阵 P)
#         # 变量顺序堆叠: w = [u0, ..., u_{N-1}, x1, ..., xN]
#         # 注意：x0 是常数，不作为优化变量
#
#         # R 块 (N 个)
#         R_list = [self.R] * self.N
#         # Q 块 (N 个)
#         Q_list = [self.Q] * self.N
#
#         # P = diag(R..., Q...)
#         self.P_mat = sparse.block_diag(R_list + Q_list, format='csc')
#
#         # 2. 构建动力学等式约束矩阵 A_eq
#         # 形式: A_eq * w = b_eq
#         # 约束展开:
#         # k=0: x1 - B u0 = A x0 + f
#         # k=1: x2 - A x1 - B u1 = f
#         # ...
#
#         # U 的系数 (-B)
#         # 这是一个块对角矩阵，对角线上全是 -B
#         B_list = [-self.B] * self.N
#         A_eq_u = sparse.block_diag(B_list)
#
#         # X 的系数 (I 和 -A)
#         # 这是一个双对角矩阵: 主对角线是 I, 下对角线是 -A
#         # I 矩阵 (Nx * N)
#         I_big = sparse.eye(self.N * self.nx)
#         # -A 矩阵 (在下对角线)
#         A_lower = sparse.kron(sparse.eye(self.N, k=-1), -self.A)
#         A_eq_x = I_big + A_lower
#
#         self.A_eq = sparse.hstack([A_eq_u, A_eq_x], format='csc')
#
#         # 3. KKT 左端矩阵 (LHS)
#         # [ P + rho*I   A_eq.T ]
#         # [ A_eq        0      ]
#
#         rho_I = self.rho * sparse.eye(self.P_mat.shape[0])
#         block_11 = self.P_mat + rho_I
#
#         KKT_left = sparse.vstack([
#             sparse.hstack([block_11, self.A_eq.T]),
#             sparse.hstack([self.A_eq, sparse.csc_matrix((self.A_eq.shape[0], self.A_eq.shape[0]))])
#         ], format='csc')
#
#         # 4. 预分解 (Factorization)
#         # 这是 ADMM 速度快的核心：矩阵逆只需要算一次
#         self.kkt_solver = splu(KKT_left)
#
#     def restart(self):
#         """重置求解器状态"""
#         self.z.fill(0)
#         self.y.fill(0)
#         self.w.fill(0)
#
#     def solve(self, x0, x_ref=None, u_ref=None, max_iter=50, tol=1e-3):
#         """
#         核心求解函数
#         :param x0: 当前状态 [nx]
#         :param x_ref: 参考轨迹 [N, nx] (可选)
#         :param max_iter: 最大迭代次数
#         """
#         t_start = time.time()
#
#         # === 1. 构建向量 q (线性代价项) ===
#         # Cost = 0.5 * x^T Q x + q^T x
#         # 如果有 ref, 则 q = -Q * x_ref
#         q_vec = np.zeros(self.n_vars)
#
#         # 填充 U 的 ref (如果有)
#         if u_ref is not None:
#             # 将 u_ref 展平填入前半部分
#             if u_ref.ndim > 1: u_ref = u_ref.flatten()
#             # q_u = -R * u_ref
#             # 这里简化处理，假设 R 是对角阵
#             for i in range(self.N * self.nu):
#                 q_vec[i] = -self.R[i % self.nu, i % self.nu] * u_ref[i]
#
#         # 填充 X 的 ref (如果有)
#         offset_x = self.N * self.nu
#         if x_ref is not None:
#             if x_ref.ndim == 1:  # 只有一个目标点
#                 x_ref = np.tile(x_ref, (self.N, 1))
#
#             x_ref_flat = x_ref.flatten()
#             # q_x = -Q * x_ref
#             # 同样利用 Kronecker 积的思想快速计算
#             # 但为了通用性，我们这里用循环或者矩阵乘法
#             # q_block = -Q @ x_ref_i
#             for k in range(self.N):
#                 start = offset_x + k * self.nx
#                 end = start + self.nx
#                 q_vec[start:end] = -self.Q @ x_ref[k]
#
#         # === 2. 构建向量 b_eq (动力学约束右端项) ===
#         # Ax = b
#         # b_eq = [A x0 + f; f; f; ...]
#         b_eq = np.zeros(self.N * self.nx)
#
#         # 第一步: A x0 + f
#         b_eq[0:self.nx] = self.A @ x0 + self.f
#         # 后续步骤: f
#         for k in range(1, self.N):
#             b_eq[k * self.nx: (k + 1) * self.nx] = self.f
#
#         # === 3. ADMM 迭代循环 ===
#         for i in range(max_iter):
#             w_prev = self.w.copy()
#
#             # --- Step 1: x-update (Primal Update) ---
#             # 对应 C++ 中的 forward_pass + backward_pass (Riccati)
#             # 这里是解稀疏线性方程组
#
#             # RHS = [ rho(z - y/rho) - q;  b_eq ]
#             rho_z_minus_y = self.rho * (self.z - self.y)
#             rhs_upper = rho_z_minus_y - q_vec
#             rhs = np.concatenate([rhs_upper, b_eq])
#
#             # 解方程
#             sol = self.kkt_solver.solve(rhs)
#             self.w = sol[:self.n_vars]  # 取出优化变量部分
#
#             # --- Step 2: z-update (Constraint Projection) ---
#             # 对应 C++ 中的 update_slack
#             # z = clip(w + y, min, max)
#
#             v = self.w + self.y
#
#             # 分离 U 和 X 进行截断
#             u_part = v[:offset_x]
#             x_part = v[offset_x:]
#
#             # 简单的 Box 截断
#             z_u = np.clip(u_part, self.u_min, self.u_max)
#             z_x = np.clip(x_part, self.x_min, self.x_max)
#
#             self.z = np.concatenate([z_u, z_x])
#
#             # --- Step 3: y-update (Dual Update) ---
#             # 对应 C++ 中的 update_dual
#             # y = y + (w - z)
#             self.y += (self.w - self.z)
#
#             # --- 收敛检查 ---
#             # 对应 C++ 中的 termination_condition
#             r_prim = np.linalg.norm(self.w - self.z)  # 原始残差
#             r_dual = np.linalg.norm(self.w - w_prev)  # 对偶残差 (简化版)
#
#             if r_prim < tol and r_dual < tol:
#                 self.iterations = i + 1
#                 break
#
#         self.solve_time = time.time() - t_start
#         self.iterations = i + 1
#
#         # === 4. 提取结果与移位 (Warm Start 准备) ===
#         # 提取第一个控制量 u0
#         u_opt = self.w[:self.nu]
#
#         # 移位操作 (Shift)
#         # 将解出来的轨迹往前挪一步，作为下一时刻的初始猜测
#         # w = [u0, u1, ..., uN-1, x1, ..., xN]
#
#         u_traj = self.w[:offset_x].reshape(self.N, self.nu)
#         x_traj = self.w[offset_x:].reshape(self.N, self.nx)
#
#         # U 移位: [u1, u2, ..., uN-1, uN-1]
#         u_shift = np.vstack([u_traj[1:], u_traj[-1]])
#         # X 移位: [x2, ..., xN, xN] (近似)
#         x_shift = np.vstack([x_traj[1:], x_traj[-1]])
#
#         self.w = np.concatenate([u_shift.flatten(), x_shift.flatten()])
#         self.z = self.w.copy()
#         # y 通常重置或者衰减，简单起见保持原样或清零
#         # self.y.fill(0)
#
#         return u_opt, x_traj
#
#
# 使用示例(main.py)
#
# 我们可以用一个简单的
# 二维平面车辆模型(Double
# Integrator) 来测试它，这和你做自瞄控制的物理场景很像（位置 + 速度）。
#
# code
# Python
# download
# content_copy
# expand_less
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     # 1. 定义物理模型 (例如：简单的 X-Y 运动)
#     # 状态: [x, y, vx, vy]
#     # 输入: [ax, ay]
#     dt = 0.05
#     A = np.array([
#         [1, 0, dt, 0],
#         [0, 1, 0, dt],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ])
#     B = np.array([
#         [0.5 * dt ** 2, 0],
#         [0.5 * dt ** 2, 0],
#         [dt, 0],
#         [0, dt]
#     ])
#     nx = 4
#     nu = 2
#     N = 20  # 预测20步 (1秒)
#
#     # 2. 定义代价矩阵
#     # 想要 x, y 也就是状态的前两维紧跟目标
#     Q = np.diag([10, 10, 1, 1])
#     # 想要加速度尽可能小
#     R = np.diag([0.1, 0.1])
#
#     # 3. 初始化 MPC
#     mpc = MPC_ADMM(
#         A, B, Q, R, N, rho=2.0,
#         u_min=-5.0, u_max=5.0,  # 加速度限制
#         x_min=-100, x_max=100
#     )
#
#     # 4. 模拟循环
#     x_curr = np.array([0, 0, 0, 0])  # 起点
#     x_target = np.array([10, 5, 0, 0])  # 目标点 (x=10, y=5, 静止)
#
#     history_x = []
#     history_u = []
#
#     print("开始仿真...")
#     for i in range(50):
#         # 求解 MPC
#         # 这里演示目标是固定点，如果是自瞄，x_target 可以是动态预测的轨迹
#         u_opt, pred_traj = mpc.solve(x_curr, x_ref=x_target, max_iter=20)
#
#         # 模拟物理更新
#         x_curr = A @ x_curr + B @ u_opt
#
#         history_x.append(x_curr)
#         history_u.append(u_opt)
#
#         print(
#             f"Step {i}: Pos=({x_curr[0]:.2f}, {x_curr[1]:.2f}) Time={mpc.solve_time * 1000:.2f}ms Iter={mpc.iterations}")
#
#     # 5. 绘图
#     hist_x = np.array(history_x)
#     hist_u = np.array(history_u)
#
#     plt.figure(figsize=(10, 8))
#
#     plt.subplot(2, 2, 1)
#     plt.plot(hist_x[:, 0], hist_x[:, 1], 'b-o')
#     plt.plot(0, 0, 'go', label='Start')
#     plt.plot(10, 5, 'rx', label='Target')
#     plt.title("2D Trajectory")
#     plt.grid(True)
#     plt.legend()
#
#     plt.subplot(2, 2, 2)
#     plt.plot(hist_x[:, 2], label='vx')
#     plt.plot(hist_x[:, 3], label='vy')
#     plt.title("Velocities")
#     plt.grid(True)
#     plt.legend()
#
#     plt.subplot(2, 1, 2)
#     plt.plot(hist_u[:, 0], label='ax')
#     plt.plot(hist_u[:, 1], label='ay')
#     plt.title("Control Inputs (Acceleration)")
#     plt.grid(True)
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
# 代码对比：TinyMPC
# C + + vs.Python
# ADMM
# 特性
# C + + (TinyMPC 源码)
# Python(我的实现)
# KKT
# 求解
# Riccati
# Recursion(前向 + 后向扫描)
# Sparse
# LU
# Decomposition(稀疏矩阵分解)
# 复杂度
# 𝑂
# (
#     𝑁
#     ⋅
#     𝑛
#     𝑥
#     3
# )
# O(N⋅n
# x
# 3
# ​
#
# )
# (实际上优化得很好)
# 𝑂
# (
#     𝑁
#     ⋅
#     𝑛
#     𝑥
#     3
# )
# O(N⋅n
# x
# 3
# ​
#
# )
# (底层C库实现)
# 内存
# 极低，无动态分配，静态数组
# 中等，依赖
# NumPy / SciPy
# 库
# Affine项
# fdyn
# 变量
# 整合进
# b_eq
# 向量
# 约束处理
# update_slack(投影)
# np.clip(投影)
# 对偶更新
# update_dual
# y += w - z
# Rho
# 自适应(Adaptive)
# 固定(Fixed) - 为了保持
# Python
# 代码简洁，可扩展
#
