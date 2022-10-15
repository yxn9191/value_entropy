class Organization:
    def __init__(self, id, model, members=None):
        self.id = id
        self.model = model
        self.members = members  # 成员列表,存储agent的id(没存agent的实例是为了节省空间)
        self.average_energy = self.cal_average_energy()
        self.model.num_organization = self.model.num_organization + 1

    def cal_average_energy(self):
        if self.members:
            total_energy = 0
            for member in self.members:
                for agent in self.model.schedule.agents:
                    if agent.unique_id == member:
                        total_energy += agent.energy
            return total_energy / len(self.members)
        else:
            return 0

    def destroy(self):
        # 解散组织
        return self.model.num_organization - 1  # 如果返回0代表组织已经解散

    def update(self):
        # 记得对组织操作后要调用更新函数
        self.average_energy = self.cal_average_energy()
        if not self.members:
            return self.destroy()
