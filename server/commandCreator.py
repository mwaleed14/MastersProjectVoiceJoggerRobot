import copy
from word2number import w2n
from enum import Enum, unique
from text import TextClassifier
from sentence_transformers import SentenceTransformer
from misc import Command,MoveDirection,CommandMode ,GoalStatus, CommandMode, Command, Controller, MoveDirection, get_relative_orientation
import rospy
import numpy as np

import moveit_msgs.msg

import time
import sys
import copy
import rospy
import moveit_commander
import actionlib
import franka_gripper.msg
import franka_msgs.msg
from actionlib_msgs.msg import GoalStatusArray
import geometry_msgs.msg
from std_msgs.msg import String
import textFileHandler as tfh
import rospy
from controller_manager_msgs.srv import SwitchController
import franka_msgs.msg
from actionlib_msgs.msg import GoalStatusArray
from moveit_msgs.msg import RobotTrajectory
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal
from franka_msgs.msg import ErrorRecoveryActionGoal
from word2number import w2n
from franka_gripper.msg import ( GraspAction, GraspGoal,
                                 HomingAction, HomingGoal,
                                 MoveAction, MoveGoal,
                                 StopAction, StopGoal,
                                 GraspEpsilon )

from robotMover import RobotMover


class Velocity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
velocities = {
    Velocity.LOW: 0.05,
    Velocity.MEDIUM: 0.2,
    Velocity.HIGH: 1.0
}

class ControllerSwitcher:
    def __init__(self, active: Controller, stopped: Controller) -> None:
        """Initialize switch service"""
        self.active = active
        self.stopped = stopped
        self.strictness = 2
        self.start_asap = False
        self.timeout = 0.0

    def switch_controller(self, active: Controller, stop: Controller):
        if self.active == active:
            return
        rospy.wait_for_service('/controller_manager/switch_controller')
        try:
            switcher = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
            switcher([active.value], [stop.value], self.strictness, self.start_asap, self.timeout)
            self.active = active
            self.stopped = stop
            rospy.sleep(0.1)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)





SIMULATION = False

class Manipulator:
    """Robot Manipulator"""
    def __init__(self):
        """Initialize manipulator"""
        self.home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        #self.home_joints = [-0.10978979745454956, -0.7703535289764404, -0.05097640468462238, -2.3268556568809795, 0.00103424144312800801817, 1.5708663142522175, 0.7840747220798833]
        # initialize moveit commander
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        # Initialize servo controller publisher
        self.servo_pub = rospy.Publisher('/cartesian_controller/command', String, queue_size=1)
        # Set grasp tool as EE link
        self.move_group.set_end_effector_link("panda_hand_tcp")
        # Clients to send commands to the gripper
        self.grasp_action_client = actionlib.SimpleActionClient("/franka_gripper/grasp", franka_gripper.msg.GraspAction)
        self.move_action_client = actionlib.SimpleActionClient("/franka_gripper/move", franka_gripper.msg.MoveAction)
        # Clients for auto recovery
        self.error_recover_pub = rospy.Publisher("/franka_control/error_recovery/goal", franka_msgs.msg.ErrorRecoveryActionGoal, queue_size=1)
        self.robot_mode_sub = rospy.Subscriber("/franka_state_controller/franka_states",
            franka_msgs.msg.FrankaState, self.franka_state_callback,)
        # Transformation Matrices
        # Bring robot to home position during initialization
        self.moveit_home(wait=True)
        self.default_ee_pose = self.move_group.get_current_pose()
         # class constants

        # Initialize moveit_commander and a rospy node
        moveit_commander.roscpp_initialize(sys.argv)

        # Instantiate a RobotCommander object. Provides information such as the robot's
        # kinematic model and the robot's current joint states

        # Instantiate a PlanningSceneInterface object.  This provides a remote interface
        # for getting, setting, and updating the robot's internal understanding of the
        # surrounding world:
        scene = moveit_commander.PlanningSceneInterface()
        
        # Create a `DisplayTrajectory`_ ROS publisher which is used to display
        # trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
                "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        ) 
        
        # Subscriber for text commands
        text_command_subscriber = rospy.Subscriber("/text_commands", String, self.handle_received_command)
        text_command_subscriber = rospy.Subscriber("/text_commands_priority", String, self.handle_received_priority_command)

        # Publish to joint trajectory controller
        if SIMULATION:
            joint_trajectory_topic = '/effort_joint_trajectory_controller/follow_joint_trajectory/goal'
        else:
            joint_trajectory_topic = '/position_joint_trajectory_controller/follow_joint_trajectory/goal'
        self.joint_trajectory_goal_pub = rospy.Publisher(
                                      joint_trajectory_topic,
                                      FollowJointTrajectoryActionGoal, 
                                      queue_size = 10)

        # Publish to error recovery topic
        self.error_recovery_pub = rospy.Publisher(
                                        '/franka_control/error_recovery/goal',
                                        ErrorRecoveryActionGoal,
                                        queue_size = 10) 
     
   

        # class variables
        self.stopped = True
       
        self.cmd_param = None
        self.step_size = 0.05
        self.position1 = None
        self.position2 = None
        self.recording_task_name = None
        self.saved_positions = tfh.load_position()
        self.saved_tasks = tfh.load_task()
        # Updating waypoint is used to calculate final goal pose of AND command chain.
        self.updating_waypoint = []
        self.velocity = Velocity.MEDIUM
        self.waiting_for_tool_name = False
        self.current_tool = None

        # class constants
        self.pickup_area_offset = 0.05
        self.pickup_approach_height = 0.15
        self.default_pickup_height = 0.01
        self.object_size = 0.07
        self.home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

        # class variables
        self.scene = scene
        self.step_size = 0.05
        self.mode = 'STEP'  # step, distance
        self.position1 = None
        self.position2 = None
        self.recording_task_name = None
        self.saved_tasks = tfh.load_task()
        # Updating waypoint is used to calculate final goal pose of AND command chain.
        self.updating_waypoint = []
        self.saved_objects = tfh.load_object()
        self.velocity = Velocity.MEDIUM
        self.waiting_for_tool_name = False
        self.current_tool = None



    def robot_stop(self):
        # Replace current trajectory with stopping trajectory
        joint_values = self.move_group.get_current_joint_values()
        print(joint_values)
        stop_trajectory = JointTrajectory()
        stop_trajectory.joint_names = ['panda_joint1',
                                       'panda_joint2',
                                       'panda_joint3',
                                       'panda_joint4',
                                       'panda_joint5',
                                       'panda_joint6',
                                       'panda_joint7']
        stop_trajectory.points.append(JointTrajectoryPoint())
        stop_trajectory.points[0].positions = joint_values
        stop_trajectory.points[0].velocities = [0.0 for i in joint_values]
        stop_trajectory.points[0].time_from_start = rospy.Duration(1) # Stopping time
        stop_trajectory.header.frame_id = 'world'
        stop_goal = FollowJointTrajectoryActionGoal()
        stop_goal.goal.trajectory = stop_trajectory
        self.joint_trajectory_goal_pub.publish(stop_goal)

        # Stop gripper
        goal = StopGoal()
        self.stop_action_client.send_goal(goal)
        self.stopped = True
        rospy.loginfo("Stopped")

    def handle_received_priority_command(self, command):
        if type(command) == String:
            cmd = command.data.split(' ')
        elif type(command) == list:
            cmd = command
        #________________SAFETY COMMANDS___________________________
        if cmd[0] == "STOP_ROBOT":
            self.robot_stop()

    
    def error_recovery(self):
        error_recovery_goal = ErrorRecoveryActionGoal()
        self.error_recovery_pub.publish(error_recovery_goal)
        rospy.loginfo("Recovered from errors")

    def robot_start(self):
        self.stopped = False
        self.error_recovery()
        print("New approach 2")
        rospy.loginfo("Started")



    def handle_received_command(self, command):
        if type(command) == String:
            cmd = command.data.split(' ')
        elif type(command) == list:
            cmd = command
            
        if self.recording_task_name is not None:
            if cmd[0] == "FINISH":
                rospy.loginfo("Finished recording task with name %s", self.recording_task_name)

                # Write tasks to text file
                tfh.write_task(self.saved_tasks)
                # Load tasks from text file. 
                self.saved_tasks = tfh.load_task()

                self.recording_task_name = None
            #elif cmd[0] == "RECORD":
            #    pass
            #else:
                #self.saved_tasks[self.recording_task_name]["moves"].append(cmd)


        # Bool parameters for command chaining
        and_bool_parameter = False
        is_end_bool_parameter = False
        if cmd[0] == "UP" or cmd[0] == "DOWN" or cmd[0] == "LEFT" or cmd[0] == "RIGHT" \
                or cmd[0] == "FORWARD" or cmd[0] == "BACKWARD":
            if self.mode == 'STEP':
                if len(cmd) > 1:
                    if cmd[1] == "False":
                        and_bool_parameter = False
                    else:
                        and_bool_parameter = True
                if len(cmd) > 2:
                    if cmd[2] == "False":
                        is_end_bool_parameter = False
                    else:
                        is_end_bool_parameter = True
            elif self.mode == 'DISTANCE':
                if len(cmd) > 2:
                    if cmd[2] == "False":
                        and_bool_parameter = False
                    else:
                        and_bool_parameter = True
                if len(cmd) > 3:
                    if cmd[3] == "False":
                        is_end_bool_parameter = False
                    else:
                        is_end_bool_parameter = True

        print("This is handle")
        #________________STATUS COMMANDS_________________________
        if cmd[0] == 'START_ROBOT':
            print("New approach 1")
            self.robot_start()
        elif cmd[0] == 'RECOVER':
            self.error_recovery()

        #________________MOVE COMMANDS___________________________
        elif cmd[0] == "HOME":
            self.move_robot_home()


        elif cmd[0] == "MOVE_UP":
            is_end_bool_parameter = False
            is_end_bool_parameter = False

            stepSize = self.step_size
            if self.mode == 'STEP':
                stepSize = self.step_size
            if self.mode == 'DISTANCE':
                stepSize = float(cmd[2]) / 1000

            print("New approach 1 MOVE")
            self.move_robot_cartesian("up", stepSize, and_bool_parameter, is_end_bool_parameter) 
        elif cmd[0] == "MOVE_DOWN":
            is_end_bool_parameter = False
            is_end_bool_parameter = False

            stepSize = self.step_size
            if self.mode == 'STEP':
                stepSize = self.step_size
            if self.mode == 'DISTANCE':
                stepSize = float(cmd[2]) / 1000

            print("New approach 1 MOVE")
            self.move_robot_cartesian("down", stepSize, and_bool_parameter, is_end_bool_parameter)
        elif cmd[0] == "MOVE_LEFT":
            is_end_bool_parameter = False
            is_end_bool_parameter = False

            stepSize = self.step_size
            if self.mode == 'STEP':
                stepSize = self.step_size
            if self.mode == 'DISTANCE':
                stepSize = float(cmd[2]) / 1000

            print("New approach 1 MOVE")
            self.move_robot_cartesian("left", stepSize, and_bool_parameter, is_end_bool_parameter)
        elif cmd[0] == "MOVE_RIGHT":
            is_end_bool_parameter = False
            is_end_bool_parameter = False

            stepSize = self.step_size
            if self.mode == 'STEP':
                stepSize = self.step_size
            if self.mode == 'DISTANCE':
                stepSize = float(cmd[2]) / 1000

            print("New approach 1 MOVE")
            self.move_robot_cartesian("right", stepSize, and_bool_parameter, is_end_bool_parameter)
        elif cmd[0] == "MOVE_FRONT":
            is_end_bool_parameter = False
            is_end_bool_parameter = False

            stepSize = self.step_size
            if self.mode == 'STEP':
                stepSize = self.step_size
            if self.mode == 'DISTANCE':
                stepSize = float(cmd[2]) / 1000

            print("New approach 1 MOVE")
            self.move_robot_cartesian("forward", stepSize, and_bool_parameter, is_end_bool_parameter)
        elif cmd[0] == "MOVE_BACK":
            is_end_bool_parameter = False
            is_end_bool_parameter = False

            stepSize = self.step_size
            if self.mode == 'STEP':
                stepSize = self.step_size
            if self.mode == 'DISTANCE':
                stepSize = float(cmd[2]) / 1000

            print("New approach 1 MOVE")
            self.move_robot_cartesian("backward", stepSize, and_bool_parameter, is_end_bool_parameter)
            
        elif cmd[0] == "POSITION_NAME" : # move robot to saved position
            self.move_robot_to_position(cmd[1])

        elif cmd[0] == "MOVE" and cmd[1] == 'POSITION':
            stepSize = self.step_size
            if self.mode == 'STEP':
                stepSize = self.step_size
                print("Position " + cmd[2] + " saved.")
                self.move_robot_to_position(cmd[2])

        
        #________________GRIPPER COMMANDS_________________________
        #elif cmd[0] == "GRIPPER" or cmd[0] == "TOL":
         #   if len(cmd) == 2:
          #      if cmd[1] == "OPEiiN":
           #         self.open_gripper()
            #    elif cmd[1] == "CLOyyyySE":
             #       self.close_gripper()
             #   elif cmd[1] == "ROr55TATE" or cmd[1] == "TURN" or cmd[1] == "SPIN":
             #       self.rotate_gripper(self.step_size)
             #   elif cmd[1] == "HOM55E":
             #       self.move_gripper_home()
     #           else:
      #              try:
       #                 distance = float(cmd[1]) / 1000
        #                self.set_gripper_distance(distance)
         #           except ValueError:
          ##              print('Sending Command to ROS: STOP')
            #            rospy.loginfo('Invalid gripper command "%s" received, available commands are:', cmd[1])
             #           rospy.loginfo('OPEN, CLOSE, ROTATE or distance between fingers in units mm between 0-80')
              #          self.shake_gripper()
          #  if len(cmd) == 3:
           #     if cmd[1] == "ROTATE" or cmd[1] == "TURN" or cmd[1] == "SPIN":
            #        if cmd[2] == 'BACK':
             #           self.rotate_gripper(self.step_size, False)
                        
        elif cmd[0] == "OPEN_TOOL":
            self.open_gripper()
        elif cmd[0] == "CLOSE_TOOL":
            print(cmd)
            self.close_gripper()
        elif cmd[0] == "ROTATE_TOOL":
            self.rotate_gripper(self.step_size)
        elif cmd[0] == "ROTATE_TOOL_BACK":
            self.rotate_gripper(self.step_size,False)

        
        #________________CHANGE MODE_____________________________
        elif cmd[0] == "MODE":
            if cmd[1] == "STEP":
                self.mode = 'STEP'
                rospy.loginfo("Mode: STEP")
            elif cmd[1] == "DISTANCE":
                self.mode = 'DISTANCE'
                rospy.loginfo("Mode: DIRECTION AND DISTANCE.")
            else:
                rospy.loginfo("Command not found.")
                self.shake_gripper()


        #________________CHANGE STEP SIZE________________________
        elif cmd[0] =='STEP_SIZE': 
            self.set_stepsize_medium()


        #________________CHANGE VELOCITY________________________
        elif cmd[0] == 'VELOCITY':
            if cmd[1] == 'LOW':
                self.velocity = Velocity.LOW
                rospy.loginfo("Velocity LOW")
            elif cmd[1] == 'MEDIUM':
                self.velocity = Velocity.MEDIUM
                rospy.loginfo("Velocity MEDIUM")
            elif cmd[1] == 'HIGH':
                self.velocity = Velocity.HIGH
                rospy.loginfo("Velocity HIGH")
            else:
                rospy.loginfo("Command not found.")
                self.shake_gripper()


        #________________SAVE ROBOT POSITION_____________________
        #elif cmd[0] == "SAVE_POSITION":
         #   self.save_position1(cmd)
         #  print("save position",cmd[2])
         # print("SAVE Position executed")
                
            

        elif cmd[0] == 'SAVE' and cmd[1] == 'POSITION':
            if cmd[2] in self.saved_positions.keys():
                rospy.loginfo("There was already a stored position with the name %s so it was overwritten", cmd[2])
            self.saved_positions[cmd[2]] = self.move_group.get_current_pose().pose
            tfh.write_position(self.saved_positions)
            self.saved_positions = tfh.load_position()
            print("Position " + cmd[2] + " saved.")
            
            

        
        #               LOAD ROBOT POSITION_____________________
        elif cmd[0] == "LOAD_POSITION":
            print("load position length", len(cmd))
            print("load tool",cmd[2])
            self.move_robot_to_position(cmd[2])
            print("Load Position executed")
            
            
            
        #_______________REMOVE ROBOT POSITION____________________
        elif cmd[0] == 'REMOVE' and cmd[1] == 'POSITION':
            print("Remove position",cmd[2])
            if cmd[2] in self.saved_positions.keys():
                tfh.deleteItem('positions.txt', cmd[2])
                self.saved_positions = tfh.load_position()
                print("Position " + cmd[2] + " removed.")
            else:
                rospy.loginfo("Not enough arguments, expected REMOVE POSITION [position name]")
                self.shake_gripper()

        #________________SAVE TOOL POSITION_____________________
        elif cmd[0] == "SAVE_TOOL":
            print("Save tool",cmd[2])
            if cmd[2] in self.saved_objects.keys():
                rospy.loginfo("There was already a stored tool with the name %s so it was overwritten", cmd[2])
            self.saved_objects[cmd[2]] = [self.move_group.get_current_pose().pose, 1]
            tfh.write_object(self.saved_objects)
            self.saved_objects = tfh.load_object()
            print("Tool " + cmd[2] + " saved.")



        #_______________REMOVE TOOL POSITION____________________
        elif cmd[0] == "REMOVE_TOOL":
            print("remove tool",cmd[2])
            if cmd[2] in self.saved_objects.keys():
                tfh.deleteItem('objects.txt', cmd[2])
                self.saved_objects = tfh.load_object()
                print("Tool " + cmd[2] + " removed.")
            else:
                rospy.loginfo("Not enough arguments, expected REMOVE TOOL [tool name]")
                self.shake_gripper()

        #_______________ TAKE TOOL__________________________
        elif cmd[0] == 'TAKE':
            if cmd[1] == 'NEW':
                if len(cmd) == 2:
                    # Pickup new tool
                    self.pickup_tool()
                elif self.waiting_for_tool_name:
                    # Save tool with name and take it to empty position
                    self.save_tool(cmd[2])
                else:
                    print('No tool to name')
                    self.shake_gripper()
            elif len(cmd) > 1:
                self.pickup_tool(cmd[1])
            else:
                rospy.loginfo("Not enough arguments, expected TAKE [tool name] or TAKE NEW TOOL")
                self.shake_gripper()

        elif cmd[0] == 'RETURN':
            if not self.current_tool:
                rospy.loginfo("No current tool to return")
                self.shake_gripper()
            else:
                self.pickup_tool(self.current_tool)


        #___________________GIVE TOOL____________________________
        elif cmd[0] == 'GIVE':
            if len(cmd) > 1:
                self.give_tool(cmd[1])
                self.current_tool = cmd[1]
            else:
                rospy.loginfo("Not enough arguments, expected GIVE [tool name]")
                self.shake_gripper()

        #___________________DROP TOOL____________________________
        elif cmd[0] == 'DROP':
            if len(cmd) > 1:
                if cmd[1] == 'ALL':
                    self.drop_all()
                else:
                    self.drop_tool(cmd[1])
            else:
                rospy.loginfo("Not enough arguments, expected GIVE [tool name]")
                self.shake_gripper()

            
        #___________________TASK RECORDINGS______________________
        elif cmd[0] == 'RECORD':
            if len(cmd) > 1:
                rospy.loginfo("Begin recording task with name %s", cmd[1])
                self.recording_task_name = cmd[1]
                if self.recording_task_name not in self.saved_tasks.keys():
                    self.saved_tasks[self.recording_task_name] = {}
                    self.saved_tasks[self.recording_task_name]["moves"] = []
            else:
                print("RECORD error: give task name.")
                self.shake_gripper()
        
        elif cmd[0] == 'TASK' or cmd[0] == 'DO' or cmd[0] == 'PLAY':
            if len(cmd) > 1:
                if cmd[1] in self.saved_tasks.keys():

                    waypoints = []
                    for step in self.saved_tasks[cmd[1]]["moves"]:
                        if self.stopped:
                            break
                        if step[0] == 'pose':
                            waypoints.append(copy.deepcopy(step[1]))
                        elif len(waypoints) > 0:
                            self.move_robot(waypoints)
                            waypoints = []
                        if step[0] == 'gripper':
                            if step[1] == 'open':
                                self.open_gripper()
                            elif step[1] == 'distance':
                                self.set_gripper_distance(step[2])
                            elif step[1] == 'close':
                                self.close_gripper()
                    if len(waypoints) > 0:
                        self.move_robot(waypoints)
                else:
                    print("Executing task failed: Task name " + cmd[1] + " not in recorded tasks.")
                    self.shake_gripper()
            else:
                print("Executing task failed: Correct command: TASK/DO/PLAY [task name]")
                self.shake_gripper()


        #___________________TEXT FILE HANDLING______________________
        #___________________LIST TASKS/POSITIONS______________________
        elif cmd[0] == 'LIST':
            if len(cmd) != 1:
                if cmd[1] == 'TASKS':
                    print("")
                    print("Saved tasks:")
                    print("")
                    self.saved_tasks = tfh.load_task()
                    for taskname in self.saved_tasks:
                        print(taskname)    
                    print("")
                elif cmd[1] == 'POSITIONS':
                    print("")
                    print("Saved positions:")
                    print("")
                    self.saved_positions = tfh.load_position()
                    for position in self.saved_positions:
                        print(position)
                    print("")
                else:
                    print("Listing failed. List tasks: LIST TASKS. List positions: LIST POSITIONS.")
                    self.shake_gripper()
            else:
                print("Listing failed. List tasks: LIST TASKS. List positions: LIST POSITIONS.")
                self.shake_gripper()


        elif cmd[0] == 'REMOVE':
            if len(cmd) < 2:
                print("REMOVE error: give task name.")
                self.shake_gripper()
            elif len(cmd) > 2:
                print("REMOVE error: Too many arguments.")
                self.shake_gripper()
            else:
                if cmd[1] in self.saved_tasks.keys():
                    tfh.deleteItem("tasks.txt", cmd[1])
                    self.saved_tasks = tfh.load_task()
                    print("Task " + cmd[1] + " removed.")
                else:
                    print("REMOVE error: No task named " + cmd[1] + " found. Use LIST TASK command too see tasks.")
                    self.shake_gripper()
                
        else:
            print("Command not found.")
            self.shake_gripper()
          
        #___________________RECORD WAYPOINT______________________
        if self.recording_task_name is not None:
            pose = copy.deepcopy(self.move_group.get_current_pose().pose)
            self.saved_tasks[self.recording_task_name]["moves"].append(['pose', pose])  
        




    def save_position1(self,cmd):
        leng = len(cmd)
        print("Length of command" ,leng)
        if cmd[2] in self.saved_positions.keys():
            print("Save Position ",cmd[2])
            rospy.loginfo("There was already a stored position with the name %s so it was overwritten", cmd[2])
            self.saved_positions[cmd[2]] = self.move_group.get_current_pose().pose
            tfh.write_position(self.saved_positions)
            self.saved_positions = tfh.load_position()
            print("Position " + "man" + " saved.")
        
    def save_position(self):
        """Save position of end-effector pose"""
        if self.cmd_param is None:
            return
        self.saved_positions[self.cmd_param] = self.manipulator.move_group.get_current_pose().pose
    
    def load_position(self):
        """Load position of end-effector pose by executing cartesian trajectory"""
        if self.cmd_param is None or self.cmd_param not in self.saved_positions:
            return
        self.controller_switcher.switch_controller(Controller.MOVEIT, Controller.SERVO)
        self.manipulator.moveit_home(True)
        self.manipulator.moveit_execute_cartesian_path([self.saved_positions[self.cmd_param]])


    def rotate_gripper(self, stepSize, clockwise = True):
        if self.stopped:
            return
        # step size are: 0.01, 0.05, 0.1
        # Increase rotating step size
        stepSize = stepSize * 10
        if not clockwise:
            stepSize = stepSize * (-1)
        joint_goal = self.move_group.get_current_joint_values()

        # Joit 7 limits. max: 2.8973, min: -2.8973
        if joint_goal[6] + stepSize >= 2.8973:
            print("Joint 7 upper limit reached. Rotate counter-clockwise. Command: ROTATE BACK")
            self.shake_gripper()
        elif joint_goal[6] + stepSize <= -2.8973:
            print("Joint 7 lower limit reached. Rotate clockwise. Command: ROTATE")
            self.shake_gripper()
        else:
            joint_goal[6] = joint_goal[6] + stepSize
            value2decimals = "{:.2f}".format(joint_goal[6])
            rospy.loginfo("Gripper rotated. Joint 7 value: " + value2decimals + ". Max: 2.90, Min: -2.90.")
            print(joint_goal[6])
            self.move_group.set_max_velocity_scaling_factor(velocities[self.velocity])
            self.move_group.go(joint_goal, wait=True)


    def shake_gripper(self):
        return
        if self.stopped:
            return            
        joint_goal = self.move_group.get_current_joint_values()
        step_size = 0.1
        # Joit 7 limits. max: 2.8973, min: -2.8973
        if joint_goal[6] + step_size >= 2.8973:
            step_size = -1 * step_size
            
        joint_goal[6] = joint_goal[6] + step_size
        self.move_group.set_max_velocity_scaling_factor(velocities[Velocity.HIGH])
        self.move_group.go(joint_goal, wait=True)
        joint_goal[6] = joint_goal[6] - step_size
        self.move_group.go(joint_goal, wait=True)
        self.move_group.set_max_velocity_scaling_factor(velocities[self.velocity])
    def move_gripper_home(self):
        pose = geometry_msgs.msg.Pose()
        pose.position.x = 0.30701957
        pose.position.y = 0
        pose.position.z = 0.59026955
        pose.orientation.x = 0.92395569
        pose.orientation.y = -0.38249949
        pose.orientation.z = 0
        pose.orientation.w = 0

        self.move_robot([pose])


    def move_robot_home(self):
        self.move_group.set_max_velocity_scaling_factor(velocities[self.velocity])
        self.move_group.go(self.home, wait=True)
        self.move_group.stop()

    def move_robot(self, waypoints):
        if self.stopped:
            return
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)  # jump_threshold
        plan = self.move_group.retime_trajectory(self.move_group.get_current_state(), plan, velocity_scaling_factor = velocities[self.velocity])
        self.move_group.execute(plan, wait=True)
                
    def move_robot_to_position(self, position):
        # If there isn't saved position, do nothing but inform user
        if position in self.saved_positions.keys():
            target = self.saved_positions[position]
            rospy.loginfo("Robot moved to position " + position)
            waypoints = []
            waypoints.append(copy.deepcopy(target))
            self.move_robot(waypoints)
        else:
            rospy.loginfo("Position " + position + " not saved.")
            self.shake_gripper()
            
            
    def move_robot_cartesian(self, direction, stepSize, is_and=False, is_end_of_and=False):
        rospy.loginfo("Mode: " + self.mode + " " + direction + " " + str(stepSize) + " m")
        
        waypoints = []

        robot_pose = self.move_group.get_current_pose().pose

        # Calculate new position for updating waypoint.
        if is_and and len(self.updating_waypoint) != 0:
            if direction == "up":
                self.updating_waypoint[0].position.z += stepSize
            if direction == "down":
                self.updating_waypoint[0].position.z -= stepSize
            if direction == "left":
                self.updating_waypoint[0].position.y -= stepSize
            if direction == "right":
                self.updating_waypoint[0].position.y += stepSize
            if direction == "forward":
                self.updating_waypoint[0].position.x += stepSize
            if direction == "backward":
                self.updating_waypoint[0].position.x -= stepSize

            if not is_end_of_and:
                return

        else:
            # Calculate a new goal pose

            if direction == "up":
                robot_pose.position.z += stepSize
            if direction == "down":
                robot_pose.position.z -= stepSize
            if direction == "left":
                robot_pose.position.y -= stepSize
            if direction == "right":
                robot_pose.position.y += stepSize
            if direction == "forward":
                robot_pose.position.x += stepSize
            if direction == "backward":
                robot_pose.position.x -= stepSize

            waypoint = copy.deepcopy(robot_pose)
            waypoints.append(waypoint)


            if is_and:
                self.updating_waypoint.append(waypoint)
                if not is_end_of_and:
                    return

        if is_and and is_end_of_and:
            waypoints.append(self.updating_waypoint[0])

        self.move_robot(waypoints)

        self.updating_waypoint = []

    def franka_state_callback(self, msg: franka_msgs.msg.FrankaState):
        """Get franka state"""
        if msg.robot_mode == franka_msgs.msg.FrankaState.ROBOT_MODE_REFLEX:
            rospy.logwarn("Executing error recovery from Reflex mode...")
            self.error_recover_pub.publish(franka_msgs.msg.ErrorRecoveryActionGoal())
            rospy.logwarn("Franka robot mode recovered back to Move mode")

    def moveit_home(self, wait=True):
        """Goto home position"""
        # Clear existing pose targets
        self.move_group.clear_pose_targets()
        # Plan home joint values
        self.move_group.set_joint_value_target(self.home_joints)
        plan = self.move_group.plan()
        self.moveit_execute_plan(plan, wait)

    def moveit_execute_plan(self, plan, wait=True) -> None:
        """Execute a given plan through move group"""
        if isinstance(plan, RobotTrajectory):
            plan = [True, plan]
        if plan[0]:
            self.move_group.execute(plan[1], wait=True)
        else:
            rospy.logwarn("Could not plan trajectory from current pose to home pose")







    def set_stepsize_medium(self):
    
        self.step_size = 0.05
        rospy.loginfo("Step size MEDIUM (5 cm)")
          



        
    def moveit_execute_cartesian_path(self, waypoints):
        """Execute cartesian path with some safety checks regarding pose waypoints"""
        z_min, z_max = 0.02, 0.50
        for pose in waypoints:
            if pose.position.z < z_min:
                rospy.logwarn(f"{pose.position.z = } is invalid. Using {z_min} instead")
                pose.position.z = z_min
            if pose.position.z > z_max:
                rospy.logwarn(f"{pose.position.z = } is invalid. Using {z_max} instead")
                pose.position.z = z_max
        plan, _ = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)  # jump_threshold
        self.moveit_execute_plan(plan)
    
    #def open_gripper(self) -> None:
    #    """Open gripper"""
    #    goal = franka_gripper.msg.MoveGoal()
    #    goal.width = 0.08
    #    goal.speed = 0.1
    #    self.move_action_client.send_goal(goal)
    #    self.move_action_client.wait_for_result()

    #def close_gripper(self):
    #    """Grasp object by closing gripper"""
    #    goal = franka_gripper.msg.GraspGoal()
    #    goal.width = 0.00
    #    goal.speed = 0.1
    #    goal.force = 5  # limits 0.01 - 50 N
    #    goal.epsilon = franka_gripper.msg.GraspEpsilon(inner=0.08, outer=0.08)
    #    self.grasp_action_client.send_goal(goal)
    #    self.grasp_action_client.wait_for_result()





    def open_gripper(self, wait=True):
        if self.stopped:
            return
        if self.recording_task_name is not None:
            self.saved_tasks[self.recording_task_name]["moves"].append(['gripper', 'open'])
        movegoal = MoveGoal()
        movegoal.width = 0.08
        movegoal.speed = 0.05
        self.move_action_client.send_goal(movegoal)
        if wait:
            self.move_action_client.wait_for_result()

    def set_gripper_distance(self, distance):
        if self.stopped:
            return
        if self.recording_task_name is not None:
            self.saved_tasks[self.recording_task_name]["moves"].append(['gripper', 'distance', distance])
        movegoal = MoveGoal()
        movegoal.width = distance
        movegoal.speed = 0.05
        self.move_action_client.send_goal(movegoal)
        self.move_action_client.wait_for_result()

    def close_gripper(self):
        if self.stopped:
            return
        if self.recording_task_name is not None:
            self.saved_tasks[self.recording_task_name]["moves"].append(['gripper', 'close'])
        graspgoal = GraspGoal()
        graspgoal.width = 0.00
        graspgoal.speed = 0.05
        graspgoal.force = 2  # limits 0.01 - 50 N
        graspgoal.epsilon = GraspEpsilon(inner=0.08, outer=0.08)
        self.grasp_action_client.send_goal(graspgoal)
        self.grasp_action_client.wait_for_result()

    def servo_move(self, data):
        """Publish command to servo controller"""
        self.servo_pub.publish(data)



def get_number(words):
    number_words = copy.copy(words)
    print("print numbers: ", number_words)
    # Replace words that sound like number with numbers
    try:
        value = float(w2n.word_to_num(' '.join(number_words)))
        return value
    except Exception as a:
        print("Invalid number.")
        return 0
    
class CommandCreator(object):
    def __init__(self, debug_enabled = False):
        self.debug_enabled = debug_enabled

        # modes: STEP, DIRECTION AND DISTANCE
        #self.mode = 'STEP'

        # step sizes: LOW, MEDIUM, HIGH
        #self.step_size = 'MEDIUM'

        # original words from microphone_input after one speech
        self.original_words = []

        # current words. Amount of words decrease when buffering.
        self.current_words = []
        self.manipulator = Manipulator()
        self.mode = CommandMode.CONTINUOUS

        self.start_robot = False
        # Step size in meters
        self.step_size = 0.1
        # Commands that rely on numeric value use this parameter
        self.cmd_param = None
        # Controller switcher
        self.controller_switcher = ControllerSwitcher(active=Controller.MOVEIT, stopped=Controller.SERVO)
        # Cliport client that sends language input and expects pick-place poses from Cliport server
        # Saved positions
        self.saved_positions = {}
        self.manipulator.cmd_param = self.cmd_param

        # do buffering when true
        self.buffering_ok = True
        self.unknown_word = "[unk]"

        self.cmds = {
            Command.START_ROBOT: lambda: self.setup_robot(True),
            Command.STOP_ROBOT: lambda: self.setup_robot(False),
            Command.SET_MODE_STEP: lambda: self.change_mode(self.original_words),
            Command.SET_MODE_CONTINUOUS: lambda: self.set_mode(CommandMode.CONTINUOUS),
            Command.SET_MODE_MODEL: lambda: self.set_mode(CommandMode.MODEL),
            Command.MOVE_UP: lambda: self.manipulator.move_robot_cartesian("up",self.manipulator.step_size,False,False),
            Command.MOVE_DOWN: lambda: self.manipulator.move_robot_cartesian("down",self.manipulator.step_size,False,False),
            Command.MOVE_LEFT: lambda: self.move(MoveDirection.LEFT),
            Command.MOVE_RIGHT: lambda: self.move(MoveDirection.RIGHT),
            Command.MOVE_FRONT: lambda: self.move(MoveDirection.FRONT),
            Command.MOVE_BACK: lambda: self.move(MoveDirection.BACK),
            Command.STOP_EXECUTION: lambda: self.stop_execution(),
            Command.STEP_SIZE: lambda: self.set_stepsize(),
            Command.OPEN_TOOL: lambda: self.oc_gripper(True),
            Command.CLOSE_TOOL: lambda: self.oc_gripper(False),
            Command.ROTATE_TOOL: lambda: self.manipulator.rotate_gripper(self.manipulator.step_size),
            Command.ROTATE_TOOL_BACK: lambda: self.manipulator.rotate_gripper(self.manipulator.step_size,False),
            Command.SAVE_POSITION: lambda: self.manipulator.save_position(),
            Command.LOAD_POSITION: lambda: self.manipulator.load_position(),
            Command.HOME: lambda: self.manipulator.move_robot_home(),
            Command.SAVE_TOOL: lambda: self.manipulator.move_robot_home(),
            Command.REMOVE_TOOL: lambda: self.manipulator.move_robot_home(),
            Command.REMOVE_POSITION: lambda: self.manipulator.save_position1(),
            Command.MOVE_POSITION: lambda: self.manipulator.move_robot_to_position("mn"),
            
        }


        self.all_words_lookup_table = {
            'start' : 'START',
            'stop' : 'STOP',
            'panda': 'PANDA',
            'robot': 'ROBOT',
            'up' : 'UP',
            'open' : 'OPEN',
            'load' : 'LOAD',
            'save' : 'SAVE',
            'close' : 'CLOSE',
            'name' : 'NAME',
            'move' : 'MOVE',
            'down' : 'DOWN','close' : 'CLOSE',
            'again' : 'AGAIN',
            'left' : 'LEFT',
            'right' : 'RIGHT',
            'forward': 'FORWARD',
            'front': 'FORWARD',
            'backward': 'BACKWARD',
            'back': 'BACKWARD',
            'mode' : 'MODE',
            'distance' : 'DISTANCE',
            'direction' : 'DIRECTION',
            'step' : 'STEP',
            'low' : 'LOW',
            'medium' : 'MEDIUM',
            'high' : 'HIGH',
            'size' : 'SIZE',
            'tool' : 'TOOL',
            'grasp' : 'GRASP',
            'rotate' : 'ROTATE',
            'list' : 'LIST',
            'show' : 'SHOW',
            'task' : 'TASK',
            'play' : 'PLAY',
            'do' : 'DO',
            'remove' : 'REMOVE',
            'delete' : 'DELETE',
            'save' : 'SAVE',
            'home' : 'HOME',
            'finish' : 'FINISH',
            'record' : 'RECORD',
            'gripper' : 'GRIPPER',
            'grasp' : 'GRASP',
            'position' : 'POSITION',
            'spot' : 'SPOT',
            'other' : 'OTHER',
            'opposite' : 'OPPOSITE', 
            'counter' : 'COUNTER',
            'velocity' : 'VELOCITY',
            'speed' : 'VELOCITY',
            'recover' : 'RECOVER',
            'take' : 'TAKE',
            'give' : 'GIVE',
            'name' : 'NAME',
            'return' : 'RETURN',
            'drop' : 'DROP',
            'and' : 'AND',
            'then' : 'THEN'
        }


    def run(self, cmd, numeric=None):
        if self.start_robot is False and cmd != Command.START_ROBOT:
            print("Robot enter in to run command")
            rospy.logwarn(f"Command failed. Initialize robot with 'start robot' command before specifying any other command!")
            return
        rospy.loginfo(f"Running {cmd = }")

        self.cmd_param = numeric
        print("Entered in the before setup function")
        self.cmds[cmd]()

    def error_recovery(self):
        error_recovery_goal = ErrorRecoveryActionGoal()
        self.manipulator.error_recovery_pub.publish(error_recovery_goal)
        rospy.loginfo("Recovered from errors")

    def setup_robot(self, start):
        """Setup robot command"""
        print("Entered iin the setup function")
        self.start_robot = start
        self.stopped = False
        self.error_recovery()
        print("Entered iiiin the AFTER setup function")

    def set_mode(self, mode):
        """Set mode command"""
        self.mode = mode

    def set_stepsize(self):
        """Step size command (given in centimeters)"""
        if self.cmd_param is None:
            return
        self.step_size = abs(self.cmd_param) / 100

    def oc_gripper(self, open):
        """Open/Close gripper"""
        if open:
            self.manipulator.open_gripper()
        else:
            self.manipulator.close_gripper()

   
    
    def move(self, direction):
        """Move commands"""
        if self.mode == CommandMode.STEP:
            step_size = self.step_size
        elif self.mode == CommandMode.CONTINUOUS:
            step_size = 10
        else:
            return
        mapping = {
            MoveDirection.UP: f'Z,{step_size}',
            MoveDirection.DOWN: f'Z,-{step_size}',
            MoveDirection.LEFT: f'Y,-{step_size}',
            MoveDirection.RIGHT: f'Y,{step_size}',
            MoveDirection.FRONT: f'X,{step_size}',
            MoveDirection.BACK: f'X,-{step_size}',
        }
        print("Inside move function: " , self.mode)
        move_param = mapping[direction]
        self.controller_switcher.switch_controller(active=Controller.MOVEIT , stop=Controller.SERVO)

        self.manipulator.servo_move(move_param)
    
    def stop_execution(self):
        """Stop running execution"""
        if self.controller_switcher.active == Controller.SERVO:
            self.manipulator.servo_move("Z,0")
        elif self.controller_switcher.active == Controller.MOVEIT:
            self.manipulator.move_group.stop()
            self.manipulator.move_group.clear_pose_targets()

  
    def home(self):
        """Goto home joint values"""
        self.controller_switcher.switch_controller(Controller.MOVEIT, Controller.SERVO)
        self.manipulator.moveit_home(True)
    
    def pick_only(self):
        """Execute only a pick sequence"""
        if self.cmd_param is None:
            return
        self.home()
        # Get ee pose
        ee_pose = self.manipulator.move_group.get_current_pose()
        ee_wxyz = [ee_pose.pose.orientation.w,
                   ee_pose.pose.orientation.x,
                   ee_pose.pose.orientation.y,
                   ee_pose.pose.orientation.z]
        # Execute pick
        pick_wxyz = get_relative_orientation(ee_wxyz, self.cmd_param['pick_rotation'])
        pick_xyz = self.cmd_param['pick_xyz']
        self._pick(pick_xyz, pick_wxyz)


    def pick_place(self):
        """Execute pick/place sequence given the poses"""
        if self.cmd_param is None:
            return
        self.home()
        # Get ee pose
        ee_pose = self.manipulator.move_group.get_current_pose()
        ee_wxyz = [ee_pose.pose.orientation.w,
                   ee_pose.pose.orientation.x,
                   ee_pose.pose.orientation.y,
                   ee_pose.pose.orientation.z]
        # Execute pick
        pick_wxyz = get_relative_orientation(ee_wxyz, self.cmd_param['pick_rotation'])
        pick_xyz = self.cmd_param['pick_xyz']
        self._pick(pick_xyz, pick_wxyz)
        # Execute place
        place_wxyz = get_relative_orientation(ee_wxyz, self.cmd_param['place_rotation'])
        place_xyz = self.cmd_param['place_xyz']
        self._place(place_xyz, place_wxyz)

    def _place(self, xyz, wxyz):
        """Execute place sequence"""
        # This is used to execute up movement before dropping the target
        z_offset_up = 0.15

        pose = geometry_msgs.msg.Pose()
        pose.position.x = xyz[0]
        pose.position.y = xyz[1]
        pose.position.z = xyz[2] + z_offset_up
        pose.orientation.w = wxyz[0]
        pose.orientation.x = wxyz[1]
        pose.orientation.y = wxyz[2]
        pose.orientation.z = wxyz[3]

        # Move above object and open gripper
        rospy.loginfo("Moving towards place object and opening gripper")
        self.manipulator.moveit_execute_cartesian_path([pose])
        self.oc_gripper(True)
        self.home()
    
    def _pick(self, xyz, wxyz):
        """Execute pick sequence"""
        # This is used to execute up-down movement when grasping the target
        z_offset_up = 0.035
        z_offset_up_2 = 0.20
        z_offset_down = 0.015

        pose = geometry_msgs.msg.Pose()
        pose.position.x = xyz[0]
        pose.position.y = xyz[1]
        pose.position.z = xyz[2]
        pose.orientation.w = wxyz[0]
        pose.orientation.x = wxyz[1]
        pose.orientation.y = wxyz[2]
        pose.orientation.z = wxyz[3]

        # Move above object and open gripper
        rospy.loginfo("Moving towards pick object and opening gripper")
        pose_up = copy.deepcopy(pose)
        pose_up.position.z += z_offset_up
        self.manipulator.moveit_execute_cartesian_path([pose_up])
        self.oc_gripper(True)
        # Move down and grasp object
        rospy.loginfo("Moving down and grasping pick object")
        pose_down = copy.deepcopy(pose)
        pose_down.position.z -= z_offset_down
        self.manipulator.moveit_execute_cartesian_path([pose_down])
        self.oc_gripper(False)
        # Move up again
        rospy.loginfo("Moving up again after picking object")
        pose_up_2 = copy.deepcopy(pose)
        pose_up_2.position.z += z_offset_up_2
        self.manipulator.moveit_execute_cartesian_path([pose_up_2])



    def getCommand(self, first_call):
        allwords = copy.copy(self.original_words)

        if first_call:
            self.buffering_ok = True
            # Filtering words
            filtered_words = []
            for word in self.original_words:
                # Check is word inside all_words_lookup_table
                checked_word = self.all_words_lookup_table.get(word)

                # Can't do buffering if one of these words exists
                if checked_word != None:
                    if checked_word in ["MOVE", "AGAIN", "RECORD", "REMOVE", "DELETE", 
                    "TASK", "DO", "PLAY", "SAVE", "POSITION", "SPOT"]:
                        self.buffering_ok = False

                # Check is word number
                checked_number = self.get_number([word])

                # Can't do buffering if number exists
                if checked_number != None:
                    self.buffering_ok = False

                # Add word to filtered_words
                # word is known word and number
                if self.unknown_word == word:
                    continue
                filtered_words.append(word)

            if self.buffering_ok:
                self.current_words = filtered_words
            else:
                self.current_words = []

            if self.original_words[0] != "":
                if self.debug_enabled:
                    print(80*"-")
                    print("All recorded words: ")
                    print(self.original_words)
                    print("")
                    print("Filtered_words: ")
                    print(filtered_words)
                    print("")
                else:
                    print(80*"-")
                    print('Detected words:')
                    print(' '.join(filtered_words))

        words = []
        if first_call:
            if self.buffering_ok:
                words = self.current_words
            else:
                words = self.original_words
        else:
            words = self.current_words

        # Take a new word from the words-list until word is found from all_words_lookup_table
        if len(words) > 0:
            command = self.all_words_lookup_table.get(words.pop(0))
            print("received command ",command)
        else:
            return None

        while(command == None):
            if len(words) == 0:
                return None
            else:
                command = self.all_words_lookup_table.get(words.pop(0))
                
        if command == "START":
            return self.get_start_command(words)
        elif command == "STOP":
            return self.get_stop_command(words)
        elif command == "RECOVER":
            return ["RECOVER"]
        elif command == "HOME":
            return ["HOME"]
        elif command == "MOVE":
            self.mode = 'STEP'
            if self.mode == 'STEP':
                print("received step command ",command)
                return self.get_move_command_step_mode(words)
            else:
                print("received continious command ",command)
                return self.get_move_command_direction_and_distance_mode(words)
        elif command == "MODE":
            return self.change_mode(words)
        elif command == "VELOCITY":
            return self.change_velocity(words)
        elif command == "STEP":
            return self.change_step_size(words)
        elif command == "TOOL":
            return self.get_tool_command(words)

        #___________________MOVING COMMANDS____________________________
        elif command in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD']:
            # Check distance
            print("received up command ",command)

            if len(words) > 0:
                distance = self.get_number(words)
                if self.buffering_ok:
                    return [command]
                elif not self.buffering_ok and distance != None:
                    return [command, distance]
                else:
                    print("Invalid moving command. ")
                    return None
            return [command]

        #___________________ROTATE TOOL________________________________
        elif command == "ROTATE":
            if len(words) == 0:
                return ["ROTATE"]
            else:
                if self.all_words_lookup_table.get(words[0], '') in ['OTHER', 'COUNTER', 'OPPOSITE']:
                    words.pop(0)
                    return ["ROTATE", "BACK"]
                elif self.all_words_lookup_table.get(words[0], '') in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD']:
                    return ["ROTATE"] 
                else:
                    print("Invalid " + command + " command.")
                    return ["ROTATE"] 


        #___________________RECORD TASKNAME_____________________________
        elif command == "RECORD":
            task_name = self.get_name(words)
            if task_name is not None:
                return ["RECORD", task_name]
            else:
                print('Invalid ' + command + ' command. Correct form: RECORD [task name]')
                return None

        #___________________REMOVE/DELETE TASK/POSITION______________________
        elif command == "REMOVE" or command == 'DELETE':
            # Remove position
            if self.all_words_lookup_table.get(words[0], '') in ['POSITION', 'SPOT']:
                words.pop(0)
                position_name = self.get_name(words)
                return ["REMOVE", "POSITION", position_name]
            else:
                task_name = self.get_name(words)
                if task_name is not None:
                    return ["REMOVE", task_name]
                else:
                    print('Invalid ' + command + ' command. Correct form: REMOVE/DELETE [task name]')
                    return None

        #___________________PLAY/DO/TASK TASKNAME______________________
        elif command == 'TASK' or command == 'DO' or command == 'PLAY':
            task_name = self.get_name(words)
            if task_name is not None:
                return ["TASK", task_name]
            else:
                print('Invalid ' + command + ' command. Correct form: TASK/DO/PLAY [task name]')
                return None

        #___________________LIST/SHOW TASKS/POSITIONS__________________
        elif command == "LIST" or command == "SHOW":
            if len(words) != 1:
                print('Invalid command ' + command + '. Correct form: LIST/SHOW TASK/TASKS/POSITION/POSITIONS')
                return None
            else:
                # list tasks
                if self.all_words_lookup_table.get(words[0], '') in ['TASK']:
                    return ['LIST', 'TASKS']
                elif self.all_words_lookup_table.get(words[0], '') in ['POSITION', 'SPOT']:
                    return ['LIST', 'POSITIONS']
                else:
                    print('Invalid command ' + words[0] + '. Correct form: LIST/SHOW TASK/TASKS/POSITION/POSITIONS')
                    return None

        #___________________SAVE POSITION______________________________
        elif command == "SAVE":
            cmd = self.all_words_lookup_table.get(words.pop(0), '')
            if cmd in ['POSITION', 'SPOT']:
                position_name = self.get_name(words)
                if position_name is not None:
                    return ['SAVE', 'POSITION', position_name]
                else:
                    print('Invalid ' + command + ' command. Correct form: SAVE POSITION/SPOT [position name]')
                    return ['SAVE', 'POSITION', position_name]
                
            elif cmd == 'CORNER':
                corner_num = self.get_number(words)
                if corner_num in [1,2]:
                    return ['SAVE', 'POSITION', 'CORNER' + str(corner_num)]
                else:
                    print('Invalid ' + command + ' command. Correct form: SAVE CORNER [1/2]')
                    return None
            elif cmd == 'TOOL':
                tool_name = self.get_name(words)
                if tool_name is not None:
                    return ['SAVE', 'TOOL', tool_name]
                else:
                    print('Invalid ' + command + ' command. Correct form: SAVE TOOL [tool name]')
                    return ['SAVE', 'TOOL', tool_name]
            else:
                print('Invalid ' + command + ' command. Correct form: SAVE TOOL [tool name]')
                return None
            

        elif command == "LOAD":
            cmd = self.all_words_lookup_table.get(words.pop(0), '')
            if cmd in ['POSITION', 'SPOT']:
                position_name = self.get_name(words)
                if position_name is not None:
                    return ['LOAD', 'POSITION', position_name]
                else:
                    print('Invalid ' + command + ' command. Correct form: LOAD POSITION/SPOT [position name]')
                    return None

        #___________________MOVE TO POSITION___________________________
        elif command == "POSITION" or command == "SPOT":
            position_name = self.get_name(words)
            return ["POSITION", position_name]

        #___________________TAKE TOOL________________________
        elif command == "TAKE":
            tool = self.get_name(words)
            if tool == "NEWtool":
                print("Give name for tool with command: 'NAME' [tool name]")
                return ["TAKE", "NEW"]
            elif tool != None:
                return ["TAKE", tool]
        
        elif command == "RETURN":
            return ["RETURN"]
        
        elif command == "NAME":
            tool = self.get_name(words)
            if tool != None:
                print("Tool name is: ", tool)
                return ["TAKE", "NEW", tool]
            else:
                print('Invalid ' + command + ' command. Correct form: NAME [tool name]')
                return None

        #___________________GIVE TOOL_________________________
        elif command == "GIVE":
            tool = self.get_name(words)
            if tool != None:
                return ["GIVE", tool]
            else:
                print('Invalid ' + command + ' command. Correct form: GIVE [tool name]')
                return None                


        #___________________DROP TOOL_________________________
        elif command == "DROP":
            tool = self.get_name(words)
            if tool != None:
                return ["DROP", tool]
            else:
                print('Invalid ' + command + ' command. Correct form: DROP [tool name]')
                return None


        elif type(command) == str:
            # Command from all_words_lookup_table
            cmd = [command]
            for word in words:
                cmd.append(word)
            return cmd
        else:
            # The first word not in all_words_lookup_table
            return allwords


    def get_name(self, words):
        words = self.words_before_chain(words)
        # no name
        if len(words) < 1:
            return None

        # only name. E.g. TASK
        elif len(words) == 1:
            # if name is only number
            name_is_number = self.get_number(words)
            if name_is_number is None:
                return words[0].upper()
            else:
                return name_is_number

        # number at the end of name. E.g. TASK1
        elif len(words) > 1:
            task_name = words.pop(0).upper()
            # Check does task_name has number
            number = self.get_number(words)
            if number is None:
                # Other word at the end of name. E.g. TASKPICK
                extra_name = ''.join(words)
                return task_name + extra_name
            else:
                # number at the end of name. E.g. TASK1
                task_name = task_name + str(number)
                return task_name


    def get_start_command(self, words):
        try:
            if len(words) != 1:
                return None
            robot_name = self.all_words_lookup_table.get(words.pop(0), '')
            if robot_name not in ['ROBOT']:
                raise ValueError('Invalid robot name specified in start command')
            
            return ['START', robot_name]
        
        except Exception as e:
            print('Invalid start command arguments received')
            print(e)
            return None

    def get_stop_command(self, words):
        try:
            if len(words) != 1:
                return None
            robot_name = self.all_words_lookup_table.get(words.pop(0), '')
            if robot_name not in ['ROBOT']:
                raise ValueError('Invalid robot name specified in stop command')
            return ['STOP', robot_name]

        except Exception as e:
            print('Invalid stop command arguments received')
            print(e)
            return None

    def get_move_command_direction_and_distance_mode(self, words):
        try:
            if len(words) < 2:
                return None
            direction = self.all_words_lookup_table.get(words.pop(0), '')
            if direction not in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD']:
                raise ValueError('Invalid direction specified in move command')
            
            value = self.get_number(words)
            if value is None:
                raise ValueError('Could not convert value to number in move command')
            
            return ['MOVE', direction, value]
        
        except Exception as e:
            print('Invalid move command arguments received')
            print(e)
            return None

    def get_number(self, words):
        number_words = words.copy()
        # Replace words that sound like number with numbers
        for i,word in enumerate(words):
            new_word = self.all_words_lookup_table.get(word, None)
            if new_word:
                number_words[i] = new_word
        try:
            value = w2n.word_to_num(' '.join(number_words))
            return value
        except Exception as a:
            pass


    def get_move_command_step_mode(self, words):
        try:
            if len(words) < 1:
                return None

            direction = self.all_words_lookup_table.get(words.pop(0), '')
            if direction not in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD']:
                raise ValueError('Invalid direction specified in move command')

            return ['MOVE', direction]

        except Exception as e:
            print('Invalid move command arguments received')
            print(e)
            return None

    def change_mode(self, words):
        try:
            if len(words) != 1:
                return None
            mode = self.all_words_lookup_table.get(words.pop(0), '')
            if mode not in ['STEP', 'DISTANCE']:
                raise ValueError('Mode: ', mode, ' not valid mode. Valid modes are STEP and DISTANCE.')

            self.mode = mode

            return ['MODE', mode]

        except Exception as e:
            print('Invalid mode change.')
            print(e)
            return None

    def change_velocity(self, words):
        try:
            if len(words) < 1:
                return None
            velocity = self.all_words_lookup_table.get(words.pop(0), '')
            print("velocity: ", velocity)
            if velocity not in ['LOW', 'MEDIUM', 'HIGH']:
                raise ValueError(velocity)
            return ['VELOCITY', velocity]

        except Exception as e:
            print('Invalid velocity change. ', e)
            return None


    def change_step_size(self, words):
        try:
            if len(words) < 2:
                return None
            word2 = self.all_words_lookup_table.get(words.pop(0), '')
            if word2 != 'SIZE':
                raise ValueError('word2: ' + word2)
            size = self.all_words_lookup_table.get(words.pop(0), '')
            print("sizeee: ", size)
            if size not in ['LOW', 'MEDIUM', 'HIGH']:
                raise ValueError(word2)

            self.step_size = size

            return ['STEP', 'SIZE', size]

        except Exception as e:
            print('Invalid step size change. ', e)
            return None
        
    def get_tool_command(self, words):
        try:
            if len(words) != 1:
                return None
            word = words.pop(0)
            tool_state = self.all_words_lookup_table.get(word, '')
            if tool_state not in ['OPEN', 'CLOSE']:
                raise ValueError('Command: ', tool_state, 
                     ' not valid command for gripper tool. Valid commands are'
                      ' OPEN and CLOSE.')
            
            return ['TOOL', tool_state]
        
        except Exception as e:
            print('Invalid tool command. ', e)
            return None

    def check_if_chained(self, words):
        # Checks if there is chaining command AND or THEN and which one it is
        is_and = False
        is_end_of_and = True
        for i in range(len(words)):
            if self.all_words_lookup_table.get(words[i]) == "AND" or self.all_words_lookup_table.get(words[i]) == "THEN":
                if self.all_words_lookup_table.get(words[i]) == "AND":
                    is_and = True
                    is_end_of_and = False

                return words[i+1:], is_and, is_end_of_and
        return None, is_and, is_end_of_and


    def words_before_chain(self, words):
        for i in range(len(words)):
            if self.all_words_lookup_table.get(words[i]) == "AND" or self.all_words_lookup_table.get(words[i]) == "THEN":
                return words[:i]
        return words
        
        
        