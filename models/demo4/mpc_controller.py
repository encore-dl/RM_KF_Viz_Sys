import numpy as np
import scipy.sparse as sparse
import osqp


class MPCController:
    def __init__(self, dt, N, q_theta=100., q_dtheta=10., r_alpha=0.01,
                 alpha_max=30, theta_min=-1e9, theta_max=1e9):
        self.dt = dt
        self.N = N
        self.nx = 2
        self.nu = 1
        self.q_theta = q_theta
        self.q_dtheta = q_dtheta
        self.r_alpha = r_alpha
        self.alpha_max = alpha_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self._build_problem()

    def _build_problem(self):
        A = np.array([[1, self.dt], [0, 1]])
        B = np.array([[0.5*self.dt**2], [self.dt]])

        n_vars = self.N * self.nx + (self.N - 1) * self.nu

        P_diag = []
        for i in range(self.N):
            P_diag.extend([2*self.q_theta, 2*self.q_dtheta])
        for i in range(self.N - 1):
            P_diag.append(2*self.r_alpha)
        self.P = sparse.diags(P_diag, format='csc')

        n_eq = self.N * self.nx
        A_eq_data = []
        A_eq_rows = []
        A_eq_cols = []
        b_eq = np.zeros(n_eq)

        # 初始状态约束
        A_eq_rows.extend([0, 1])
        A_eq_cols.extend([0, 1])
        A_eq_data.extend([1, 1])

        for i in range(1, self.N):
            row = i * 2
            # x_i
            A_eq_rows.append(row)
            A_eq_cols.append(i*2)
            A_eq_data.append(1)
            # -A x_{i-1}
            A_eq_rows.append(row)
            A_eq_cols.append((i-1)*2)
            A_eq_data.append(-A[0,0])
            A_eq_rows.append(row)
            A_eq_cols.append((i-1)*2+1)
            A_eq_data.append(-A[0,1])
            # -B u_{i-1}
            u_idx = self.N * self.nx + (i-1)
            A_eq_rows.append(row)
            A_eq_cols.append(u_idx)
            A_eq_data.append(-B[0,0])
            # 下一行
            row += 1
            A_eq_rows.append(row)
            A_eq_cols.append(i*2+1)
            A_eq_data.append(1)
            A_eq_rows.append(row)
            A_eq_cols.append((i-1)*2)
            A_eq_data.append(-A[1,0])
            A_eq_rows.append(row)
            A_eq_cols.append((i-1)*2+1)
            A_eq_data.append(-A[1,1])
            A_eq_rows.append(row)
            A_eq_cols.append(u_idx)
            A_eq_data.append(-B[1,0])

        self.A_eq = sparse.csc_matrix((A_eq_data, (A_eq_rows, A_eq_cols)), shape=(n_eq, n_vars))

        # 不等式约束
        n_ineq_theta = self.N * 2
        n_ineq_u = (self.N - 1) * 2
        n_ineq = n_ineq_theta + n_ineq_u
        A_ineq_data = []
        A_ineq_rows = []
        A_ineq_cols = []
        l_ineq = []
        u_ineq = []

        # 角度边界
        for i in range(self.N):
            # theta_i >= min
            A_ineq_rows.append(len(l_ineq))
            A_ineq_cols.append(i*2)
            A_ineq_data.append(1)
            l_ineq.append(self.theta_min)
            u_ineq.append(np.inf)
            # theta_i <= max
            A_ineq_rows.append(len(l_ineq))
            A_ineq_cols.append(i*2)
            A_ineq_data.append(1)
            l_ineq.append(-np.inf)
            u_ineq.append(self.theta_max)

        # 角加速度边界
        for i in range(self.N - 1):
            u_idx = self.N * self.nx + i
            # alpha_i >= -max
            A_ineq_rows.append(len(l_ineq))
            A_ineq_cols.append(u_idx)
            A_ineq_data.append(1)
            l_ineq.append(-self.alpha_max)
            u_ineq.append(np.inf)
            # alpha_i <= max
            A_ineq_rows.append(len(l_ineq))
            A_ineq_cols.append(u_idx)
            A_ineq_data.append(1)
            l_ineq.append(-np.inf)
            u_ineq.append(self.alpha_max)

        self.A_ineq = sparse.csc_matrix((A_ineq_data, (A_ineq_rows, A_ineq_cols)), shape=(n_ineq, n_vars))
        self.l_ineq = np.array(l_ineq)
        self.u_ineq = np.array(u_ineq)

        self.A = sparse.vstack([self.A_eq, self.A_ineq])
        self.l = np.concatenate([b_eq, self.l_ineq])
        self.u = np.concatenate([b_eq, self.u_ineq])

        self.prob = osqp.OSQP()
        self.prob.setup(P=self.P, q=np.zeros(n_vars), A=self.A, l=self.l, u=self.u, verbose=False)

    def solve(self, x0, theta_ref, omega_ref=None, theta_min=None, theta_max=None):
        n_vars = self.P.shape[0]
        q = np.zeros(n_vars)
        for i in range(self.N):
            q[i*2] = -2 * self.q_theta * theta_ref[i]
            if omega_ref is not None:
                q[i * 2 + 1] = -2 * self.q_dtheta * omega_ref[i]
            else:
                q[i * 2 + 1] = 0.0

        self.l[0] = x0[0]
        self.u[0] = x0[0]
        self.l[1] = x0[1]
        self.u[1] = x0[1]

        if theta_min is not None:
            offset = self.A_eq.shape[0]
            for i in range(self.N):
                self.l[offset + 2*i] = theta_min[i]
                self.u[offset + 2*i + 1] = theta_max[i]

        self.prob.update(q=q, l=self.l, u=self.u)
        res = self.prob.solve()
        if res.info.status == 'solved':
            return res.x[self.N * self.nx]
        else:
            return 0.0