"""Utility functions"""
from enum import Enum, unique

import numpy as np

"""Utility functions"""
from enum import Enum, unique
from transforms3d._gohlketransforms import quaternion_matrix, euler_matrix, quaternion_from_matrix



class GoalStatus(Enum):
    PENDING = 0   # The goal has yet to be processed by the action server
    ACTIVE  = 1   # The goal is currently being processed by the action server
    PREEMPTED  = 2   # The goal received a cancel request after it started executing
                     #   and has since completed its execution (Terminal State)
    SUCCEEDED = 3   # The goal was achieved successfully by the action server (Terminal State)
    ABORTED = 4   # The goal was aborted during execution by the action server due
                  #    to some failure (Terminal State)
    REJECTED = 5   # The goal was rejected by the action server without being processed,
                   #    because the goal was unattainable or invalid (Terminal State)
    PREEMPTING = 6   # The goal received a cancel request after it started executing
                     #    and has not yet completed execution
    RECALLING = 7   # The goal received a cancel request before it started executing,
                    #    but the action server has not yet confirmed that the goal is canceled
    RECALLED = 8   # The goal received a cancel request before it started executing
                   #    and was successfully cancelled (Terminal State)
    LOST = 9   # An action client can determine that a goal is LOST. This should not be
                #    sent over the wire by an action server


class CommandMode(Enum):
    CONTINUOUS = 'continuous'
    STEP = 'step'
    MODEL = 'model'
"""Utility functions"""
from enum import Enum, unique

import numpy as np
from transforms3d._gohlketransforms import quaternion_matrix, euler_matrix, quaternion_from_matrix




class MoveDirection(Enum):
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'
    FRONT = 'front'
    BACK = 'back'

class GoalStatus(Enum):
    PENDING = 0   # The goal has yet to be processed by the action server
    ACTIVE  = 1   # The goal is currently being processed by the action server
    PREEMPTED  = 2   # The goal received a cancel request after it started executing
                     #   and has since completed its execution (Terminal State)
    SUCCEEDED = 3   # The goal was achieved successfully by the action server (Terminal State)
    ABORTED = 4   # The goal was aborted during execution by the action server due
                  #    to some failure (Terminal State)
    REJECTED = 5   # The goal was rejected by the action server without being processed,
                   #    because the goal was unattainable or invalid (Terminal State)
    PREEMPTING = 6   # The goal received a cancel request after it started executing
                     #    and has not yet completed execution
    RECALLING = 7   # The goal received a cancel request before it started executing,
                    #    but the action server has not yet confirmed that the goal is canceled
    RECALLED = 8   # The goal received a cancel request before it started executing
                   #    and was successfully cancelled (Terminal State)
    LOST = 9   # An action client can determine that a goal is LOST. This should not be
                #    sent over the wire by an action server


class CommandMode(Enum):
    CONTINUOUS = 'continuous'
    STEP = 'step'
    MODEL = 'model'

@unique
class Command(Enum):
    START_ROBOT = 0
    STOP_ROBOT = 1
    SET_MODE_STEP = 2
    SET_MODE_CONTINUOUS = 3
    SET_MODE_MODEL = 4
    MOVE_UP = 5
    MOVE_DOWN = 6
    MOVE_LEFT = 7
    MOVE_RIGHT = 8
    MOVE_BACK = 9
    MOVE_FRONT = 10
    STOP_EXECUTION = 11
    STEP_SIZE = 12
    OPEN_TOOL = 13
    CLOSE_TOOL = 14
    ROTATE_TOOL = 15
    ROTATE_TOOL_BACK = 16
    SAVE_POSITION = 17
    LOAD_POSITION = 18
    HOME = 19
    SAVE_TOOL = 20
    REMOVE_TOOL = 21
    REMOVE_POSITION = 22
    POSITION_NAME = 23
    TAKE_NEW = 24 
    PICK_POSITION = 25 
 
    def __str__(self):
        return self.name
    

class Controller(Enum):
    MOVEIT = "position_joint_trajectory_controller"
    SERVO = "cartesian_controller"

def get_relative_orientation(reference, yaw_rotation):
    """Get orientation relative to reference. Reference is in quaternion (WXYZ) while rotation is given in yaw degrees. Returned orientation is in quaternion (WXYZ)"""
    ref_matrix = quaternion_matrix(reference)
    rel_matrix = euler_matrix(0, 0, np.radians(yaw_rotation), 'sxyz')
    res_matrix = np.dot(rel_matrix, ref_matrix)
    
    return quaternion_from_matrix(res_matrix)
