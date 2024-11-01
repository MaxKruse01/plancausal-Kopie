class Operation:
    """Operation with job id, operation id, machine, duration, next operation"""
    def __init__(self, job_id, operation_id, machine_group_id, tool, duration, succ):
        self.job_id = job_id
        self.operation_id = operation_id
        self.req_machine_group_id = machine_group_id
        self.tool = tool
        self.plan_machine_id = None
        self.machine = None
        self.duration = duration
        self.successor = succ
        self.plan_start = None
        self.plan_end = None
        self.plan_duration = duration
        self.sim_start = None
        self.sim_duration = None
        self.sim_end = None
        self.successor_operation = None
        self.predecessor_operations = []

    def __repr__(self):
        """default print(operation) output"""
        return f"Operation(job_id='{self.job_id}', operation_id={self.operation_id}, plan_machine_id='{self.plan_machine_id}', " \
               f"duration={self.duration}, successor={self.successor}, plan_start={self.plan_start}, plan_end={self.plan_end})"


    def to_dict(self):
        """ used to create gantt charts """
        return {'job_id': self.job_id
                , 'operation_id': self.operation_id
                , 'machine': self.plan_machine_id
                , 'start_time': self.plan_start
                , 'duration' : self.duration
                , 'plan_duration' : self.plan_duration
                , 'end_time': self.plan_end }
    
    def to_dict_sim(self):
        """ used to create gantt charts """
        return {'job_id': self.job_id
                , 'operation_id': self.operation_id
                , 'machine': self.plan_machine_id
                , 'start_time': self.sim_start
                , 'duration' : self.sim_duration
                , 'plan_duration' : self.plan_duration
                , 'end_time': self.sim_end }
## Simulation relevant 
