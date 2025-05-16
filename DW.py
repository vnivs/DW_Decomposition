from gurobipy import *
import numpy as np
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray


class MCTP:
    def __init__(self):
        self.NumSupply = 0
        self.NumDemand = 0
        self.NumCommodity = 0
        self.Supply = []
        self.Demand = []
        self.Capacity = []
        self.Cost = []
        self.sub_var = {}
        self.a = {}
        self.lamuda = {}
        self.mu = {}
        self.gama = {}

    def readData(self,datapath):
        with open(datapath,'r') as f:
            #把文件按照行读出来并且去除所有的空格行
            lines = [line.strip()for line in f if line.strip()]
            self.NumSupply = int(lines[0])
            self.NumDemand = int(lines[1])
            self.NumCommodity = int(lines[2])

            #读取supply数据
            idx = 3
            assert lines[idx] == "supply"
            idx += 1

            for _ in range(self.NumSupply):
                self.Supply.append(list(map(int,lines[idx].split())))
                idx += 1
            self.Supply = np.array(self.Supply)

            #读取demand数据
            assert lines[idx] == "demand"
            idx += 1
            for _ in range(self.NumDemand):
                self.Demand.append(list(map(int,lines[idx].split())))
                idx += 1
            self.Demand = np.array(self.Demand)

            #读取capacity数据
            assert lines[idx] == "capacity"
            idx += 1
            for _ in range(self.NumSupply):
                self.Capacity.append(list(map(int,lines[idx].split())))
                idx += 1
            self.Capacity = np.array(self.Capacity)

            #读取cost数据
            assert lines[idx] == "cost"
            idx += 1
            for _ in range(self.NumSupply * self.NumDemand):
                self.Cost.append(list(map(int,lines[idx].split())))
                idx += 1
            self.Cost = np.array(self.Cost)

    def subproblem(self,s,d):
        self.Subproblem = Model()
        self.Subproblem.setParam('OutputFlag',0)
        #设置子问题变量
        self.sub_var = {}
        for i in range(self.NumSupply):
            for j in range(self.NumDemand):
                self.sub_var[i,j] = self.Subproblem.addVar(vtype = GRB.CONTINUOUS, lb = 0, name = 'x'+str(i)+str(j))
        #设置子问题的约束
            #supply_constraint
        for i in range(self.NumSupply):
            self.Subproblem.addConstr(quicksum(self.sub_var[i,j] for j in range(self.NumDemand)) <= s[i])
            #demand_constraint
        for j in range(self.NumDemand):
            self.Subproblem.addConstr(quicksum(self.sub_var[i,j] for i in range(self.NumSupply)) >= d[j])


    def ColumnGeneration(self):
        sub1_solution = {}
        sub2_solution = {}
        sub3_solution = {}
        # 任意设置子问题的目标函数，得到各子问题多面体的一个极点作为辅助受限主问题的初始列
        #第一个子问题 s = 400 700 800; d = 300 300 100 75 650 225 250
        for i in range(self.NumCommodity):
            self.subproblem(self.Supply[:,i],self.Demand[:,i])
            self.Subproblem.setObjective(1,GRB.MINIMIZE)
            self.Subproblem.optimize()
            if i == 0:
                for key,value in self.sub_var.items():
                    sub1_solution[key] = value.X
            elif i == 1:
                for key, value in self.sub_var.items():
                    sub2_solution[key] = value.X
            else:
                for key,vaule in self.sub_var.items():
                    sub3_solution[key] = vaule.X
        return sub1_solution, sub2_solution, sub3_solution

    def Twophaseinitialization(self):
        x_1, x_2, x_3 = self.ColumnGeneration()
        #设置辅助受限主问题
        self.AuRMP = Model("RMP")
        self.AuRMP.setParam('OutputFlag',0)
        #设置人工变量
        s = {}
        a = {}
        for i in range(self.NumSupply):
            for j in range(self.NumDemand):
                self.a[i,j] = self.AuRMP.addVar(vtype = GRB.CONTINUOUS, lb = 0, obj = 1, name = 'a'+str(i)+str(j))
        #凸组合约束变量
        self.lamuda[0] = self.AuRMP.addVar(vtype = GRB.CONTINUOUS, lb = 0, name = 'lamuda_0')
        self.mu[0] = self.AuRMP.addVar(vtype = GRB.CONTINUOUS, lb = 0, name = 'mu_0')
        self.gama[0] = self.AuRMP.addVar(vtype = GRB.CONTINUOUS, lb = 0, name = 'gama_0')
        #设置约束
        self.cons_capacity = {}
        self.cons_convexity = {}
        for i in range(self.NumSupply):
            for j in range(self.NumDemand):
                self.cons_capacity[i,j] = self.AuRMP.addConstr(x_1[i,j]*self.lamuda[0] + x_2[i,j]*self.mu[0] +
                                x_3[i,j]*self.gama[0] - self.a[i,j] <= self.Capacity[i,j],name = f"({i},{j})")
        self.cons_convexity[0] = self.AuRMP.addConstr(self.lamuda[0] == 1)
        self.cons_convexity[1] = self.AuRMP.addConstr(self.mu[0] == 1)
        self.cons_convexity[2] = self.AuRMP.addConstr(self.gama[0] == 1)
        self.AuRMP.optimize()
        iter = 0
        while self.AuRMP.ObjVal > 1e-6:
            iter += 1
            print(f"-------------第{iter}次迭代---------------")
            pi = self.AuRMP.getAttr("PI",self.AuRMP.getConstrs())
            pi_capacity = np.array(pi[0:-self.NumCommodity]).reshape(self.NumSupply, self.NumDemand)
            sub_obj_vaule = {}
            sub_sulution = {}
            for i in range(self.NumCommodity):
                self.subproblem(self.Supply[:,i],self.Demand[:,i])
                obj = LinExpr()
                for m in range(self.NumSupply):
                    for n in range(self.NumDemand):
                        obj.addTerms(-pi_capacity[m,n],self.sub_var[m,n])
                self.Subproblem.setObjective(obj-pi[-self.NumCommodity + i],GRB.MINIMIZE)
                self.Subproblem.optimize()
                if i == 0:
                    sub_obj_vaule[i] = self.Subproblem.ObjVal
                    sub_sulution[i] = {}
                    for key,value in self.sub_var.items():
                        sub_sulution[i][key] = value.X
                elif i == 1:
                    sub_obj_vaule[i] = self.Subproblem.ObjVal
                    sub_sulution[i] = {}
                    for key,value in self.sub_var.items():
                        sub_sulution[i][key] = value.X
                else:
                    sub_obj_vaule[i] = self.Subproblem.ObjVal
                    sub_sulution[i] = {}
                    for key,value in self.sub_var.items():
                        sub_sulution[i][key] = value.X
            min_obj_sub = min(sub_obj_vaule, key = lambda k :sub_obj_vaule[k])
            new_col = Column()
            for i in range(self.NumSupply):
                for j in range(self.NumDemand):
                    new_col.addTerms(sub_sulution[min_obj_sub][i,j], self.cons_capacity[i,j])
            for i in range(self.NumCommodity):
                if i == min_obj_sub:
                    new_col.addTerms(1,self.cons_convexity[i])
                else:
                    new_col.addTerms(0,self.cons_convexity[i])
            if min_obj_sub == 0:
                self.lamuda[max(self.lamuda.keys()) + 1] = self.AuRMP.addVar(vtype=GRB.CONTINUOUS,
                                                 lb = 0, column=new_col, name = "lamuda"+str(max(self.lamuda.keys()) + 1))
            elif min_obj_sub == 1:
                self.mu[max(self.mu.keys()) + 1] = self.AuRMP.addVar(vtype=GRB.CONTINUOUS,
                                         lb = 0, column=new_col, name = "mu"+str(max(self.mu.keys()) + 1))
            else:
                self.gama[max(self.gama.keys()) + 1] = self.AuRMP.addVar(vtype=GRB.CONTINUOUS,
                                                                lb=0, column=new_col,
                                                                name="gama" + str(max(self.gama.keys()) + 1))
            self.AuRMP.optimize()
            self.AuRMP.write("RMP.lp")

    def optimize_RMP(self):
        #第一阶段结束，找到了一组基可行解，移除添加的人工变量
        for i in range(self.NumSupply):
            for j in range(self.NumDemand):
                self.AuRMP.remove(self.a[i,j])
        self.obj = LinExpr()
        for key in self.lamuda.keys():
            coef = []
            for name, constr in self.cons_capacity.items():
                coef.append(self.AuRMP.getCoeff(constr,self.lamuda[key]))
            coef = np.array(coef).reshape(self.NumSupply, self.NumDemand)
            for i in range(self.NumSupply):
                for j in range(self.NumDemand):
                    self.obj.addTerms(self.Cost[:,0].reshape(self.NumSupply,self.NumDemand)[i,j] * coef[i,j],self.lamuda[key])

        for key in self.mu.keys():
            coef = []
            for name, constr in self.cons_capacity.items():
                coef.append(self.AuRMP.getCoeff(constr,self.mu[key]))
            coef = np.array(coef).reshape(self.NumSupply, self.NumDemand)
            for i in range(self.NumSupply):
                for j in range(self.NumDemand):
                    self.obj.addTerms(self.Cost[:, 1].reshape(self.NumSupply, self.NumDemand)[i, j] * coef[i, j],self.mu[key])

        for key in self.gama.keys():
            coef = []
            for name, constr in self.cons_capacity.items():
                coef.append(self.AuRMP.getCoeff(constr,self.gama[key]))
            coef = np.array(coef).reshape(self.NumSupply, self.NumDemand)
            for i in range(self.NumSupply):
                for j in range(self.NumDemand):
                    self.obj.addTerms(self.Cost[:, 2].reshape(self.NumSupply, self.NumDemand)[i, j] * coef[i, j],self.gama[key])
        self.AuRMP.setObjective(self.obj, GRB.MINIMIZE)
        self.AuRMP.optimize()
        #求解受限主问题，检查子问题的目标函数，如果子问题目标函数都大于-1e-6（非常小），则无生成列，达到最优，求解完毕
        pi = list(value for value in self.AuRMP.getAttr("PI",self.cons_capacity).values())
        dual_capacity = np.array(pi).reshape(self.NumSupply, self.NumDemand)
        dual_convex = self.AuRMP.getAttr("PI",self.cons_convexity)
        self.sub_obj_vaule = {}
        sub_solution = {}
        for i in range(self.NumCommodity):
            cost_sub = np.array(self.Cost[:,i]).reshape(self.NumSupply, self.NumDemand)
            self.subproblem(self.Supply[:, i], self.Demand[:, i])
            obj = LinExpr()
            for m in range(self.NumSupply):
                for n in range(self.NumDemand):
                    obj.addTerms(cost_sub[m,n]-dual_capacity[m, n], self.sub_var[m, n])
            self.Subproblem.setObjective(obj - dual_convex[i], GRB.MINIMIZE)
            self.Subproblem.optimize()
            if i == 0:
                self.sub_obj_vaule[i] = self.Subproblem.ObjVal
                sub_solution[i] = {}
                for key, value in self.sub_var.items():
                    sub_solution[i][key] = value.X
            elif i == 1:
                self.sub_obj_vaule[i] = self.Subproblem.ObjVal
                sub_solution[i] = {}
                for key, value in self.sub_var.items():
                    sub_solution[i][key] = value.X
            else:
                self.sub_obj_vaule[i] = self.Subproblem.ObjVal
                sub_solution[i] = {}
                for key, value in self.sub_var.items():
                    sub_solution[i][key] = value.X
        self.min_obj_sub = min(self.sub_obj_vaule, key=lambda k: self.sub_obj_vaule[k])

        while self.sub_obj_vaule[self.min_obj_sub] < -1e-6:
            new_col = Column()
            for i in range(self.NumSupply):
                for j in range(self.NumDemand):
                    new_col.addTerms(sub_solution[self.min_obj_sub][i, j], self.cons_capacity[i, j])
            for i in range(self.NumCommodity):
                if i == self.min_obj_sub:
                    new_col.addTerms(1, self.cons_convexity[i])
                else:
                    new_col.addTerms(0, self.cons_convexity[i])
            if self.min_obj_sub == 0:
                self.lamuda[max(self.lamuda.keys()) + 1] = self.AuRMP.addVar(vtype=GRB.CONTINUOUS,
                                                                             lb=0, column=new_col, name="lamuda" + str(
                        max(self.lamuda.keys()) + 1))
            elif self.min_obj_sub == 1:
                self.mu[max(self.mu.keys()) + 1] = self.AuRMP.addVar(vtype=GRB.CONTINUOUS,
                                                                     lb=0, column=new_col,
                                                                     name="mu" + str(max(self.mu.keys()) + 1))
            else:
                self.gama[max(self.gama.keys()) + 1] = self.AuRMP.addVar(vtype=GRB.CONTINUOUS,
                                                                         lb=0, column=new_col,
                                                                         name="gama" + str(max(self.gama.keys()) + 1))
            #更新RMP的目标函数
            if self.min_obj_sub == 0:
                for i in range(self.NumSupply):
                    for j in range(self.NumDemand):
                        self.obj.addTerms(
                            self.Cost[:,0].reshape(3,7)[i,j]*sub_solution[0][i,j], self.lamuda[max(self.lamuda.keys())]
                        )
            elif self.min_obj_sub == 1:
                for i in range(self.NumSupply):
                    for j in range(self.NumDemand):
                        self.obj.addTerms(
                            self.Cost[:,1].reshape(3,7)[i,j]*sub_solution[1][i,j], self.mu[max(self.mu.keys())]
                        )
            else:
                for i in range(self.NumSupply):
                    for j in range(self.NumDemand):
                        self.obj.addTerms(
                            self.Cost[:,2].reshape(3,7)[i,j]*sub_solution[2][i,j],self.gama[max(self.gama.keys())]
                        )
            self.AuRMP.setObjective(self.obj, GRB.MINIMIZE)
            self.AuRMP.optimize()
            print("------当前最优解为-------")
            print(self.AuRMP.ObjVal)
            pi = list(value for value in self.AuRMP.getAttr("PI", self.cons_capacity).values())
            dual_capacity = np.array(pi).reshape(self.NumSupply, self.NumDemand)
            dual_convex = self.AuRMP.getAttr("PI", self.cons_convexity)
            self.sub_obj_vaule = {}
            sub_solution = {}
            for i in range(self.NumCommodity):
                cost_sub = np.array(self.Cost[:, i]).reshape(self.NumSupply, self.NumDemand)
                self.subproblem(self.Supply[:, i], self.Demand[:, i])
                obj = LinExpr()
                for m in range(self.NumSupply):
                    for n in range(self.NumDemand):
                        obj.addTerms(cost_sub[m, n] - dual_capacity[m, n], self.sub_var[m, n])
                self.Subproblem.setObjective(obj - dual_convex[i], GRB.MINIMIZE)
                self.Subproblem.optimize()
                if i == 0:
                    self.sub_obj_vaule[i] = self.Subproblem.ObjVal
                    sub_solution[i] = {}
                    for key, value in self.sub_var.items():
                        sub_solution[i][key] = value.X
                elif i == 1:
                    self.sub_obj_vaule[i] = self.Subproblem.ObjVal
                    sub_solution[i] = {}
                    for key, value in self.sub_var.items():
                        sub_solution[i][key] = value.X
                else:
                    self.sub_obj_vaule[i] = self.Subproblem.ObjVal
                    sub_solution[i] = {}
                    for key, value in self.sub_var.items():
                        sub_solution[i][key] = value.X
            self.min_obj_sub = min(self.sub_obj_vaule, key=lambda k: self.sub_obj_vaule[k])



if __name__ == "__main__":
    instance_MCTP = MCTP()
    instance_MCTP.readData("muticommdity.txt")
    instance_MCTP.Twophaseinitialization()
    instance_MCTP.optimize_RMP()






































        