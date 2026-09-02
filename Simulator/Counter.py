from pyomo.environ import *
import numpy as np
import time



# class PolyHausdorffCalculator:
#     """
#     计算两个多面体之间的Hausdorff距离及其对约束参数的灵敏度
#
#     属性:
#         model: Pyomo模型对象
#         solver_name: 使用的求解器名称
#         M: 大M常数
#         delta_A: A矩阵的扰动幅度
#         delta_b: b向量的扰动幅度
#     """
#
#     def __init__(self, A_target, b_target, solver_name='gurobi', M=1e3, delta_A=1e-5, delta_b=1e-3):
#         """
#         初始化Hausdorff计算器
#
#         参数:
#             solver_name (str): 求解器名称 (默认: 'gurobi')
#             M (float): 大M常数 (默认: 1e3)
#             delta_A (float): A矩阵的扰动幅度 (默认: 1e-5)
#             delta_b (float): b向量的扰动幅度 (默认: 1e-3)
#         """
#         self.solver_name = solver_name
#         self.M = M
#         self.delta_A = delta_A
#         self.delta_b = delta_b
#         self.create_model(A_target, b_target)
#
#     def create_model(self, C, d):
#         """
#         创建Hausdorff距离计算模型
#
#         参数:
#             C (np.array): 固定多面体的约束矩阵 (k×n)
#             d (np.array): 固定多面体的约束向量 (k)
#             n (int): 变量维度
#
#         返回:
#             ConcreteModel: 创建好的Pyomo模型
#         """
#         k = C.shape[0]
#         n = C.shape[1]
#
#         model = ConcreteModel()
#
#         # 将A和b定义为参数（设为可变，mutable=True）
#         model.A = Param(range(k), range(n), mutable=True, initialize=0.0)
#         model.b = Param(range(k), mutable=True, initialize=0.0)
#
#         # 定义变量
#         model.x = Var(range(n), domain=Reals)  # 可变多面体中的点
#         model.y = Var(range(n), domain=Reals)  # 固定多面体中的点
#         model.lambda_AtoC = Var(range(k), domain=NonNegativeReals)  # 拉格朗日乘子
#         model.z_AtoC = Var(range(k), domain=Binary)  # 互补条件的二进制变量
#         model.lambda_CtoA = Var(range(k), domain=NonNegativeReals)  # 拉格朗日乘子
#         model.z_CtoA = Var(range(k), domain=Binary)  # 互补条件的二进制变量
#
#         # 目标函数：最大化距离平方
#         def objective_rule(model):
#             return sum((model.x[i] - model.y[i]) ** 2 for i in range(n))
#
#         model.obj = Objective(rule=objective_rule, sense=maximize)
#
#         # 约束条件
#         # 1. Ax ≤ b
#         def constraint_Ax_rule(model, i):
#             return sum(model.A[i, j] * model.x[j] for j in range(n)) <= model.b[i]
#
#         model.constr_Ax = Constraint(range(k), rule=constraint_Ax_rule)
#
#         # 2. Cy ≤ d (固定约束)
#         def constraint_Cy_rule(model, i):
#             return sum(C[i, j] * model.y[j] for j in range(n)) <= d[i]
#
#         model.constr_Cy = Constraint(range(k), rule=constraint_Cy_rule)
#
#         # 3. y - x + C^T λ = 0
#         def constraint_eq_rule_AtoC(model, j):
#             return model.y[j] - model.x[j] + sum(C[i, j] * model.lambda_AtoC[i] for i in range(k)) == 0
#
#         model.constr_eq_AtoC = Constraint(range(n), rule=constraint_eq_rule_AtoC)
#
#         # 4. 互补松弛条件的大M法处理
#         # 4a: λ_i ≤ z_i * M
#         def constraint_lambda_rule_AtoC(model, i):
#             return model.lambda_AtoC[i] <= model.z_AtoC[i] * self.M
#
#         model.constr_lambda_AtoC = Constraint(range(k), rule=constraint_lambda_rule_AtoC)
#
#         # 4b: d_i - C_i y ≤ (1-z_i)*M
#         def constraint_comp_rule_AtoC(model, i):
#             return d[i] - sum(C[i, j] * model.y[j] for j in range(n)) <= (1 - model.z_AtoC[i]) * self.M
#
#         model.constr_comp_AtoC = Constraint(range(k), rule=constraint_comp_rule_AtoC)
#
#         # 5. x - y + A^T λ = 0
#         def constraint_eq_rule_CtoA(model, j):
#             return model.x[j] - model.y[j] + sum(model.A[i, j] * model.lambda_CtoA[i] for i in range(k)) == 0
#
#         model.constr_eq_CtoA = Constraint(range(n), rule=constraint_eq_rule_CtoA)
#
#         # 6. 互补松弛条件的大M法处理
#         # 6a: λ_i ≤ z_i * M
#         def constraint_lambda_rule_CtoA(model, i):
#             return model.lambda_CtoA[i] <= model.z_CtoA[i] * self.M
#
#         model.constr_lambda_CtoA = Constraint(range(k), rule=constraint_lambda_rule_CtoA)
#
#         # 6b: b_i - A_i x ≤ (1-z_i)*M
#         def constraint_comp_rule_CtoA(model, i):
#             return model.b[i] - sum(model.A[i, j] * model.x[j] for j in range(n)) <= (1 - model.z_CtoA[i]) * self.M
#
#         model.constr_comp_CtoA = Constraint(range(k), rule=constraint_comp_rule_CtoA)
#
#         self.model = model
#         return
#
#     def solve(self):
#         """
#         求解Hausdorff距离模型
#
#         返回:
#             tuple: (距离平方值, 求解成功状态)
#         """
#         if self.model is None:
#             raise ValueError("模型尚未创建，请先调用create_model方法")
#
#         solver = SolverFactory(self.solver_name)
#
#         # 先求解A到C的方向
#         self.model.constr_eq_CtoA.deactivate()
#         self.model.constr_lambda_CtoA.deactivate()
#         self.model.constr_comp_CtoA.deactivate()
#
#         self.model.constr_eq_AtoC.activate()
#         self.model.constr_lambda_AtoC.activate()
#         self.model.constr_comp_AtoC.activate()
#
#         results = solver.solve(self.model)
#
#         distance_sq = []
#         if results.solver.termination_condition == TerminationCondition.optimal:
#             distance_sq.append(value(self.model.obj()))
#         else:
#             return None, False
#
#         # 再求解C到A的方向
#         self.model.constr_eq_AtoC.deactivate()
#         self.model.constr_lambda_AtoC.deactivate()
#         self.model.constr_comp_AtoC.deactivate()
#
#         self.model.constr_eq_CtoA.activate()
#         self.model.constr_lambda_CtoA.activate()
#         self.model.constr_comp_CtoA.activate()
#
#         results = solver.solve(self.model)
#
#         if results.solver.termination_condition == TerminationCondition.optimal:
#             distance_sq.append(value(self.model.obj()))
#         else:
#             return None, False
#
#         return max(distance_sq), True
#
#     def compute_sensitivity(self, A, b):
#         """
#         计算Hausdorff距离及其对A和b的灵敏度
#
#         参数:
#             A (np.array): 可变多面体的约束矩阵 (k×n)
#             b (np.array): 可变多面体的约束向量 (k)
#             C (np.array): 固定多面体的约束矩阵 (k×n)
#             d (np.array): 固定多面体的约束向量 (k)
#
#         返回:
#             dict: 包含距离、灵敏度矩阵等的结果
#         """
#         k, n = A.shape
#         start_time = time.time()
#
#         # 设置A和b的初始值
#         for i in range(k):
#             self.model.b[i] = b[i]
#             for j in range(n):
#                 self.model.A[i, j] = A[i, j]
#
#         # 求解基准情况
#         base_distance_sq, success = self.solve()
#         if not success:
#             return {'distance': None, 'sensitivity_A': None, 'sensitivity_b': None,
#                     'solver_status': 'solve_failed'}
#
#         base_distance = np.sqrt(base_distance_sq)
#         solve_time = time.time() - start_time
#
#         print(f"基准求解完成，距离: {base_distance:.6f}，耗时: {solve_time:.2f}秒")
#
#         # 扰动法计算灵敏度
#         sensitivity_A = np.zeros_like(A, dtype=float)
#         sensitivity_b = np.zeros_like(b, dtype=float)
#
#         # 对A的每个元素进行扰动
#         perturb_start = time.time()
#         for i in range(k):
#             original_A_i = np.array([self.model.A[i, jx].value for jx in range(n)])
#             for j in range(n):
#                 # 正向扰动并标准化
#                 self.model.A[i, j] = original_A_i[j] + self.delta_A
#                 row_norm = np.sqrt(
#                     np.linalg.norm(original_A_i) ** 2 - original_A_i[j] ** 2 + value(self.model.A[i, j]) ** 2)
#                 for jx in range(n):
#                     self.model.A[i, jx] = self.model.A[i, jx] / row_norm if row_norm > 1e-10 else 0
#                 pos_distance_sq, _ = self.solve()
#
#                 # 负向扰动
#                 self.model.A[i, j] = original_A_i[j] - self.delta_A
#                 row_norm = np.sqrt(
#                     np.linalg.norm(original_A_i) ** 2 - original_A_i[j] ** 2 + value(self.model.A[i, j]) ** 2)
#                 for jx in range(n):
#                     self.model.A[i, jx] = self.model.A[i, jx] / row_norm if row_norm > 1e-10 else 0
#                 neg_distance_sq, _ = self.solve()
#
#                 # 恢复原始值
#                 for jx in range(n):
#                     self.model.A[i, jx] = original_A_i[jx]
#
#                 if pos_distance_sq is not None and neg_distance_sq is not None:
#                     pos_distance = np.sqrt(pos_distance_sq)
#                     neg_distance = np.sqrt(neg_distance_sq)
#                     if pos_distance > base_distance and neg_distance > base_distance:
#                         sensitivity_A[i, j] = 0.0
#                     else:
#                         sensitivity_A[i, j] = (pos_distance - neg_distance) / (2 * self.delta_A)
#                 else:
#                     sensitivity_A[i, j] = np.nan
#
#         # 对b的每个元素进行扰动
#         for i in range(k):
#             # 备份原始值
#             original_b_i = self.model.b[i].value
#
#             # 正向扰动
#             self.model.b[i] = original_b_i + self.delta_b
#             pos_distance_sq, _ = self.solve()
#
#             # 负向扰动
#             self.model.b[i] = original_b_i - self.delta_b
#             neg_distance_sq, _ = self.solve()
#
#             # 恢复原始值
#             self.model.b[i] = original_b_i
#
#             if pos_distance_sq is not None and neg_distance_sq is not None:
#                 pos_distance = np.sqrt(pos_distance_sq)
#                 neg_distance = np.sqrt(neg_distance_sq)
#                 if pos_distance > base_distance and neg_distance > base_distance:
#                     sensitivity_b[i] = 0.0
#                 else:
#                     sensitivity_b[i] = (pos_distance - neg_distance) / (2 * self.delta_b)
#             else:
#                 sensitivity_b[i] = np.nan
#
#         total_time = time.time() - start_time
#         perturb_time = total_time - solve_time
#
#         print(f"灵敏度计算完成，总耗时: {total_time:.2f}秒（基准求解: {solve_time:.2f}秒，扰动求解: {perturb_time:.2f}秒）")
#
#         return {
#             'distance': base_distance,
#             'sensitivity_A': sensitivity_A,
#             'sensitivity_b': sensitivity_b,
#             'solve_time': solve_time,
#             'perturb_time': perturb_time,
#             'total_time': total_time,
#             'solver_status': 'success'
#         }
# # 示例用法
# if __name__ == "__main__":
#     # 定义问题数据
#     C = np.array([[1, 0], [0, 1], [-1, 0], [0, -1],
#                   [1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=float)
#     d = np.array([1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5], dtype=float)
#
#     A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
#     b = np.array([1.0, 1.0, 1, 1], dtype=float) - 0.50 / (np.sqrt(2) + 2)
#
#     # 创建计算器实例
#     calculator = PolyHausdorffCalculator(
#         solver_name='gurobi',  # 使用Gurobi求解器
#         A_target = C, b_target = d,
#         delta_A=1e-4,  # A的扰动幅度
#         delta_b=1e-4  # b的扰动幅度
#     )
#
#     # 计算Hausdorff距离和灵敏度
#     result = calculator.compute_sensitivity(A, b)
#
#     # 输出结果
#     print("\n计算结果:")
#     print(f"Hausdorff距离: {result['distance']:.6f}")
#     print("\n对b的灵敏度:")
#     print(result['sensitivity_b'])
#     print("\n对A的灵敏度矩阵:")
#     print(result['sensitivity_A'])
#     print(f"\n计算耗时: 基准求解 {result['solve_time']:.2f}秒, 扰动求解 {result['perturb_time']:.2f}秒, 总计 {result['total_time']:.2f}秒")



class PolyBallHausdorffCalculator:
    """
    计算多面体与单位球之间的Hausdorff距离

    属性:
        model: Pyomo模型对象
        solver_name: 使用的求解器名称
        M: 大M常数
        delta_A: A矩阵的扰动幅度
        delta_b: b向量的扰动幅度
    """

    def __init__(self, A,b, R = 1.0, solver_name='gurobi', M=1e3, delta_A=1e-5, delta_b=1e-3):
        """
        初始化计算器

        参数:
            solver_name (str): 求解器名称 (默认: 'gurobi')
            M (float): 大M常数 (默认: 1e3)
            delta_A (float): A矩阵的扰动幅度 (默认: 1e-5)
            delta_b (float): b向量的扰动幅度 (默认: 1e-3)
        """
        self.solver_name = solver_name

        self.ncons, self.dim = A.shape
        self.A = A
        self.b = b
        self.R = R
        self.M = M
        self.delta_A = delta_A
        self.delta_b = delta_b

    def create_model(self):
        """
        创建Hausdorff距离计算模型

        参数:
            A (np.array): 多面体的约束矩阵 (k×n)
            b (np.array): 多面体的约束向量 (k)
        """
        k, n = self.ncons, self.dim
        model_P2B = ConcreteModel()

        # 将A和b定义为参数（设为可变，mutable=True）
        model_P2B.A = Param(range(k), range(n), mutable=True, initialize=0.0)
        model_P2B.b = Param(range(k), mutable=True, initialize=0.0)

        # 定义变量
        model_P2B.x = Var(range(n), domain=Reals)  # 多面体中的点

        # ==================== 第一部分：多面体到单位球 ====================
        # 目标：max ||x||^2 (球心在原点)
        model_P2B.obj = Objective(
            expr=sum(model_P2B.x[i] ** 2 for i in range(n)),
            sense=maximize
        )

        # 多面体约束：Ax <= b
        def poly_constraint_rule(model, i):
            return sum( model_P2B.A[i, j] * model.x[j] for j in range(n)) <=  model_P2B.b[i]

        model_P2B.poly_constr = Constraint(range(k), rule=poly_constraint_rule)

        # ==================== 第二部分：单位球到多面体 ====================
        #理论计算，无需优化
        self.model_P2B = model_P2B


    def solve(self):
        """
        求解Hausdorff距离

        返回:
            tuple: (距离值, 求解状态)
        """
        solver = SolverFactory(self.solver_name)

        # ===== 第一部分：多面体到单位球 =====
        results = solver.solve(self.model_P2B,tee=False)
        if results.solver.termination_condition != TerminationCondition.optimal:
            return None, False

        # 计算实际距离 (||x|| - R, 最小为0)
        d1 = max(0.0,np.sqrt(value(self.model_P2B.obj))-self.R)

        # ===== 第二部分：单位球到多面体 =====
        row_norms = np.linalg.norm(self.A, axis=1)
        # 处理范数为0的行（避免除以0）
        zero_norm_mask = row_norms == 0
        row_norms[zero_norm_mask] = 1.0 # 若范数为0，缩放因子设为1（保持原值）
        # 标准化矩阵A：每行除以对应范数
        A_norm = self.A / row_norms[:, np.newaxis]  # 通过np.newaxis保持维度对齐
        # 标准化向量b：每个元素除以对应行的范数
        b_norm = self.b / row_norms
        # print(b_norm)
        d2 = np.max([0,np.max(self.R-b_norm)])


        # 最终Hausdorff距离
        # hausdorff_dist = max(d1, d2)
        hausdorff_dist = d1**2+d2**2 #真实目标函数
        return hausdorff_dist, {'feas':d1, 'opt':d2}, True

    def compute_hausdorff(self, sensitivity = False):
        """
        计算Hausdorff距离及其对A和b的灵敏度

        参数:
            A (np.array): 可变多面体的约束矩阵 (k×n)
            b (np.array): 可变多面体的约束向量 (k)

        返回:
            dict: 包含距离、灵敏度矩阵等的结果
        """
        A = self.A
        b = self.b
        k, n = A.shape
        start_time = time.time()

        # 设置A和b的初始值
        for i in range(k):
            self.model_P2B.b[i] = b[i]
            for j in range(n):
                self.model_P2B.A[i, j] = A[i, j]
        # 求解基准情况
        base_distance, single_distances, success = self.solve()
        if not success:
            return {'distance': None, 'sensitivity_A': None, 'sensitivity_b': None,
                    'solver_status': 'solve_failed'}

        solve_time = time.time() - start_time

        print(f"基准求解完成，距离: {base_distance:.6f}，耗时: {solve_time:.2f}秒")
        sensitivity_A = None
        sensitivity_b = None
        perturb_time = None
        total_time = solve_time
        if sensitivity:
            # 扰动法计算灵敏度
            sensitivity_A = np.zeros_like(A, dtype=float)
            sensitivity_b = np.zeros_like(b, dtype=float)

            # 对A的每个元素进行扰动
            perturb_start = time.time()
            for i in range(k):
                original_A_i = np.array([self.model_P2B.A[i, jx].value for jx in range(n)])
                for j in range(n):
                    # 正向扰动并标准化
                    self.model_P2B.A[i, j] = original_A_i[j] + self.delta_A

                    row_norm = np.sqrt(
                        np.linalg.norm(original_A_i) ** 2 - original_A_i[j] ** 2 + value(self.model_P2B.A[i, j]) ** 2)
                    for jx in range(n):
                        self.model_P2B.A[i, jx] = self.model_P2B.A[i, jx] / row_norm if row_norm > 1e-10 else 0
                    pos_distance, _, _ = self.solve()

                    # 负向扰动
                    self.model_P2B.A[i, j] = original_A_i[j] - self.delta_A
                    row_norm = np.sqrt(
                        np.linalg.norm(original_A_i) ** 2 - original_A_i[j] ** 2 + value(self.model_P2B.A[i, j]) ** 2)
                    for jx in range(n):
                        self.model_P2B.A[i, jx] = self.model_P2B.A[i, jx] / row_norm if row_norm > 1e-10 else 0
                    neg_distance, _, _ = self.solve()

                    # 恢复原始值
                    self.model_P2B.A[i, j] = original_A_i[j]

                    if pos_distance is not None and neg_distance is not None:
                        if pos_distance > base_distance and neg_distance > base_distance:
                            sensitivity_A[i, j] = 0.0
                        else:
                            sensitivity_A[i, j] = (pos_distance - neg_distance) / (2 * self.delta_A)
                    else:
                        sensitivity_A[i, j] = np.nan

            # 对b的每个元素进行扰动
            for i in range(k):
                # 备份原始值
                original_b_i = self.model_P2B.b[i].value

                # 正向扰动
                self.model_P2B.b[i] = original_b_i + self.delta_b
                pos_distance, _, _ = self.solve()

                # 负向扰动
                self.model_P2B.b[i] = original_b_i - self.delta_b
                neg_distance, _, _ = self.solve()

                # 恢复原始值
                self.model_P2B.b[i] = original_b_i

                if pos_distance is not None and neg_distance is not None:
                    if pos_distance > base_distance and neg_distance > base_distance:
                        sensitivity_b[i] = 0.0
                    else:
                        sensitivity_b[i] = (pos_distance - neg_distance) / (2 * self.delta_b)
                else:
                    sensitivity_b[i] = np.nan

            total_time = time.time() - start_time
            perturb_time = total_time - solve_time

            print(f"灵敏度计算完成，总耗时: {total_time:.2f}秒（基准求解: {solve_time:.2f}秒，扰动求解: {perturb_time:.2f}秒）")

        return {
            'distance': base_distance,
            'single_distances':single_distances,
            # 'distance_ideal': (np.sqrt(self.dim)-1)/(np.sqrt(self.dim)+1),
            'distance_ideal': (np.sqrt(self.dim)-1)**2/(self.dim+1),
            'sensitivity_A': sensitivity_A,
            'sensitivity_b': sensitivity_b,
            'solve_time': solve_time,
            'perturb_time': perturb_time,
            'total_time': total_time,
            'solver_status': 'success'
        }


# 示例用法
if __name__ == "__main__":
    # 定义多面体 (正方形 [-1,1]×[-1,1])
    dim = 50
    A = np.vstack([np.eye(dim),-np.eye(dim)])
    b = 2.0/(np.sqrt(dim)+1)*np.array(np.ones(2*dim),dtype=float)

    # 创建计算器
    calculator = PolyBallHausdorffCalculator(
        A = A,
        b = b,
        R = 1.0,
        solver_name='gurobi',  # 推荐使用支持二阶锥的求解器
        M=1e3,
        delta_A=1e-4,
        delta_b=1e-4
    )

    # 计算距离
    calculator.create_model()
    res = calculator.compute_hausdorff()
    print(f"Hausdorff距离: {res['distance']:.4f} (计算状态: {'成功' if res['solver_status'] else '失败'})")
    print(f"理论距离:{res['distance_ideal']:.4f}")
    # # 灵敏度分析 (可选)
    # sensitivity = calculator.compute_hausdorff()
    # print("\n灵敏度分析结果:")
    # print(f"对b的灵敏度: {sensitivity['sensitivity_b']}")
    # print(f"对A的灵敏度矩阵:\n{sensitivity['sensitivity_A']}")
