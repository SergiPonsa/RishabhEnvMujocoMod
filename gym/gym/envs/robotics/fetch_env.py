import numpy as np
import time
from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class PID(object):
    def __init__(self,max_velocity = 30,kp=0.1,ki=0.0,kd=0.0):
        self.max_velocity = max_velocity

        self._kp = kp
        self._ki = ki
        self._kd = kd


        self._target_theta = 0.0
        self._sampling_time = 0.01

        self._theta0 = 0.0
        self._thetai = 0.0

    def init_status(self):
        self._theta0 = 0.0
        self._thetai = 0.0

    def set_target_theta(self, theta, degrees = True):
        if(degrees==True):
            self._target_theta = theta
        else:
            self._target_theta = theta*(180/3.14)

    def get_target_theta(self):
        return self._target_theta

    def get_velocity(self, theta , max_velocity = None):
        if(max_velocity == None):
            max_velocity = self.max_velocity

        error = self._target_theta - theta
        self._thetai += error * self._sampling_time
        dtheta = (error - self._theta0) / self._sampling_time
        self._theta0 = error


        duty_ratio = (error * self._kp + self._thetai * self._ki + dtheta * self._kd)/self._sampling_time

        if duty_ratio > max_velocity:
            duty_ratio = max_velocity
        elif duty_ratio < -max_velocity:
            duty_ratio = -max_velocity

        return duty_ratio

class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,object_pos_from_base=[0.4,0.0],\
        goal_pos_from_base=[0.5,0.2,0.0],joint_control=False
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.object_pos_from_base = object_pos_from_base
        self.goal_pos_from_base = goal_pos_from_base
        self.initial_qpos = initial_qpos
        self.joint_control = joint_control


        self._pid = [PID(),PID(),PID(),PID(),PID(),PID(),PID()]
        self._convertdeg2rad = 57.295779578552

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            #self.sim.data.set_joint_qpos('robot1:right_knuckle_joint', 0)
            #self.sim.data.set_joint_qpos('robot1:left_knuckle_joint', 0)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion {Vertical}
        #rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)

        if self.block_gripper:
           gripper_ctrl = np.zeros_like(gripper_ctrl)

        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.

        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)



    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_body_xpos('robot1:ee_link')
        self.grip_pos = grip_pos
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_body_xvelp('robot1:ee_link') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot1:ee_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        # if self.has_object:
        #     object_xpos = self.initial_gripper_xpos[:2]
        #     while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #         object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        #     object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos
        #     self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        #Define the Object position
        if self.has_object:
            Robot_Base = [0.0, 0.0, 0.0]
            #NOTE Change the object xpos, self.object_pos_from_base
            object_xpos = [Robot_Base[0]+self.object_pos_from_base[0],\
                            Robot_Base[1]+self.object_pos_from_base[1],\
                            0.025]
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:3] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)


        self.sim.forward()
        return True

    def _sample_goal(self):

        # Randomize position of the goal.
        # if self.has_object:
        #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        #     goal += self.target_offset
        #     goal[2] = self.height_offset
        #     if self.target_in_the_air and self.np_random.uniform() < 0.5:
        #         goal[2] += self.np_random.uniform(0, 0.45)
        # else:
        #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        #return goal.copy()

        #Define the goal position
        if self.has_object:
            Robot_Base = [0.0, 0.0, 0.0]
            #NOTE Change the object xpos, self.object_pos_from_base
            goal = [Robot_Base[0]+self.goal_pos_from_base[0],\
                            Robot_Base[1]+self.goal_pos_from_base[1],\
                            Robot_Base[2]+self.goal_pos_from_base[2]]
            goal = np.array(goal)
            print(goal.shape)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)

        if (self.joint_control == False):
            utils.reset_mocap_welds(self.sim)
            self.sim.forward()

            # Move end effector into position.

            Robot_Base = [0.0, 0.0, 0.0]
            Gripper_init = [0.138,0.0,0.55]
            gripper_target = np.array(Robot_Base)+np.array(Gripper_init)#+np.array([0.0, 0.0, -0.4])
            #gripper_target = np.array([0.436, 0.75, 1.15-0.4-0.2]) #+ self.sim.data.get_site_xpos('robot0:grip')
            #gripper_target =  np.array([0.0, 0.0,-0.4-0.2])+self.sim.data.get_body_xpos('robot1:ee_link')
            gripper_rotation = np.array([1., 0., 1., 0.] )
            self.sim.data.set_mocap_pos('robot1:mocap', gripper_target)
            self.sim.data.set_mocap_quat('robot1:mocap', gripper_rotation)

            for _ in range(10):
                self.sim.step()

            # Extract information for sampling goals.
            self.initial_gripper_xpos = self.sim.data.get_body_xpos('robot1:ee_link').copy() # Needs a change if using the gripper for goal generation
            if self.has_object:
                self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)

    def Reset_Save_Data_On_Environment(self):
        #Save Data
        try:
            #print(self.sergi_joints)
            #print(self.sergi_tcp)
            try:

                self.sergi_experiments_joints.append(self.sergi_joints.copy())
                self.sergi_experiments_tcp.append(self.sergi_tcp.copy())
                self.sergi_joints = []
                self.sergi_tcp = []
            except:
                self.sergi_experiments_joints = [self.sergi_joints.copy()]
                self.sergi_experiments_tcp = [self.sergi_tcp.copy()]
                self.sergi_joints = []
                self.sergi_tcp = []
        except:
            print("No data of joints and tcps was wanted to be recorded")

    def Save_Data_On_Environment(self):
        try:
            self.sergi_joints.append([self.sim.data.sensordata[0],\
                                    self.sim.data.sensordata[1],\
                                    self.sim.data.sensordata[2],\
                                    self.sim.data.sensordata[3],\
                                    self.sim.data.sensordata[4],\
                                    self.sim.data.sensordata[5],\
                                    self.sim.data.sensordata[6]])
            self.sergi_tcp.append([self.sim.data.sensordata[7],\
                                    self.sim.data.sensordata[8],\
                                    self.sim.data.sensordata[9],\
                                    self.sim.data.sensordata[10],\
                                    self.sim.data.sensordata[11],\
                                    self.sim.data.sensordata[12],\
                                    self.sim.data.sensordata[13]])
        except:
            #they don't exist yet create them
            self.sergi_joints = [ [self.sim.data.sensordata[0],\
                                    self.sim.data.sensordata[1],\
                                    self.sim.data.sensordata[2],\
                                    self.sim.data.sensordata[3],\
                                    self.sim.data.sensordata[4],\
                                    self.sim.data.sensordata[5],\
                                    self.sim.data.sensordata[6]] ]
            self.sergi_tcp = [ [self.sim.data.sensordata[7],\
                                    self.sim.data.sensordata[8],\
                                    self.sim.data.sensordata[9],\
                                    self.sim.data.sensordata[10],\
                                    self.sim.data.sensordata[11],\
                                    self.sim.data.sensordata[12],\
                                    self.sim.data.sensordata[13]] ]
    def step_joints_home(self):

        #Set the PID and PID set joint position
        if(self.joint_control == True):
            self.sim.model.neq = 0
        for i in range(500):
            for jointNum in range(7):
                theta = self.sim.data.sensordata[jointNum]
                target_theta = list(self.initial_qpos.values())[jointNum]
                self._pid[jointNum].set_target_theta(np.rad2deg(target_theta))
                linearVelocity = self._pid[jointNum].get_velocity(np.rad2deg(theta)) /self._convertdeg2rad
                self.sim.data.ctrl[jointNum] = linearVelocity

            self.sim.step()
            self._get_viewer('human').render()
            time.sleep(0.002)

    def step_joints_offset(self,action_offset):

        #Set the PID and PID set joint position
        print("\n")
        print(self.sim.data.ctrl)
        for jointNum in range(7):
            theta = self.sim.data.sensordata[jointNum]
            print(theta)
            target_theta = theta + action_offset[jointNum]
            self._pid[jointNum].set_target_theta(np.rad2deg(target_theta))
            linearVelocity = self._pid[jointNum].get_velocity(np.rad2deg(theta)) /self._convertdeg2rad
            print(linearVelocity)

            print("\n")
            self.sim.data.ctrl[jointNum] = linearVelocity

        print(self.sim.data.ctrl)
        self.sim.step()
        self._get_viewer('human').render()
        time.sleep(0.002)



    def Change_Mujoco_Parameters(self,parameters_to_change,parameters_to_change_values,\
                                actuators_to_change=["Shoulder_Link_motor","HalfArm1_Link_motor","HalfArm2_Link_motor","ForeArm_Link_motor",\
                                "SphericalWrist1_Link_motor","SphericalWrist2_Link_motor","Bracelet_Link_motor"],\
                                joints_to_change=["robot1:Actuator1","robot1:Actuator2","robot1:Actuator3","robot1:Actuator4",\
                                                "robot1:Actuator5","robot1:Actuator6","robot1:Actuator7"],\
                                bodies_to_change=["robot1:Shoulder_Link","robot1:HalfArm1_Link","robot1:HalfArm2_Link",\
                                                    "robot1:ForeArm_Link","robot1:SphericalWrist1_Link","robot1:SphericalWrist2_Link",\
                                                    "robot1:Bracelet_Link"]):
        """
        bodies_to_change=["robot1:Shoulder_Link","robot1:HalfArm1_Link","robot1:HalfArm2_Link"\
                            "robot1:ForeArm_Link","robot1:SphericalWrist1_Link","robot1:SphericalWrist2_Link",\
                            "robot1:Bracelet_Link","robot1:ee_link",\
                            "right_driver", "right_coupler", "right_follower_link", "right_spring_link",\
                            "gripper_central", "left_driver", "left_coupler", "left_follower_link", "left_spring_link"]):
        """

        parameters_actuators = ["ctrlrange","forcerange","gainprm"]
        #gainprm has 10 values but only the first one it's used.
        parameters_body = ["mass","inertia"]
        parameters_joints = ["stiffness"]
        parameters_all = parameters_actuators + parameters_body + parameters_joints

        #Check if the data provided is right
        dict_values_parameters = {}
        for parameters_pos in parameters_all:
            dict_values_parameters[parameters_pos] = 3 if (parameters_pos == "inertia")\
                                                    else 2 if (parameters_pos == "ctrlrange")\
                                                    else 2 if (parameters_pos == "forcerange")\
                                                    else 1 if (parameters_pos == "gainprm")\
                                                    else 1

        Expected_values = 0
        for parameter_change in parameters_to_change:
            if (parameter_change in parameters_all):
                number_of_elements = len(actuators_to_change) if (parameter_change in parameters_actuators)\
                                    else len(bodies_to_change) if (parameter_change in parameters_body)\
                                    else len(joints_to_change) if (parameter_change in parameters_joints)\
                                    else 0
                Expected_values += dict_values_parameters[parameter_change]*number_of_elements
            else:
                print(parameter_change, "is not a parameter allowed the parameters allowed are ", parameters_all)

        if (Expected_values != len(parameters_to_change_values)):

            print("Given ", len(parameters_to_change_values), "and expected ",Expected_values)

        #Make the actual Change

        for parameter_change in parameters_to_change:
            number_of_elements = len(actuators_to_change) if (parameter_change in parameters_actuators)\
                                else len(bodies_to_change) if (parameter_change in parameters_body)\
                                else len(joints_to_change) if (parameter_change in parameters_joints)\
                                else 0
            #Get the value
            value=[]
            for j in range(number_of_elements):

                if (dict_values_parameters[parameter_change] > 1):
                    for i in range (dict_values_parameters[parameter_change]):
                        value.append( parameters_to_change_values.pop(0) )
                elif (parameter_change == "gainprm"):
                    value_aux = [0]*10
                    value_aux[0] = parameters_to_change_values.pop(0)
                    value.append(value_aux)
                else:
                    value.append( parameters_to_change_values.pop(0) )

            value_x_element = np.array_split(value,number_of_elements)

            #Parameter assignation
            if (parameter_change in parameters_actuators):
                for actuator in actuators_to_change:
                    ind_act = list(self.sim.model.actuator_names).index(actuator)

                    if (parameter_change == "gainprm"):
                        self.sim.model.actuator_gainprm[ind_act] = np.array(list(value_x_element).pop(0)) #array of 10 values
                    elif (parameter_change == "ctrlrange"):
                        self.sim.model.actuator_ctrlrange[ind_act] = np.array(list(value_x_element).pop(0)) #array of 2 values
                    elif (parameter_change == "forcerange"):
                        self.sim.model.actuator_forcerange[ind_act] = np.array(list(value_x_element).pop(0)) #array of 2 values
                    else:
                        print(("\n error," + parameter_change + "it's not defined, modify the code")*20)

            if (parameter_change in parameters_joints):
                for joint in joints_to_change:
                    ind_joint = list(self.sim.model.joint_names).index(joint)

                    if(parameter_change == "stiffness"):
                        self.sim.model.joint_stiffness[ind_joint] = list(list(value_x_element).pop(0)).pop(0) #only a value
                    else:
                        print(("\n error," + parameter_change + "it's not defined, modify the code")*20)

            if (parameter_change in parameters_body):
                for body in bodies_to_change:
                    ind_body = list(self.sim.model.body_names).index(body)

                    if(parameter_change == "mass"):
                        self.sim.model.body_mass[ind_body] = list(list(value_x_element).pop(0)).pop(0) #only a value
                    elif (parameter_change == "inertia"):
                        self.sim.model.body_inertia[ind_body] = np.array(list(value_x_element).pop(0)) #array of 3 values
                    else:
                        print(("\n error," + parameter_change + "it's not defined, modify the code")*20)
