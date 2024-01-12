"""Main server code for handling Android->Speech2text->ROS. Use Ctrl-C to stop script"""
import queue

from commandCreator import CommandCreator
from udp_handler import UDPReceiver
from model_handler import Recognizer, SpeechRecognizer
from text import TextClassifier
from commandCreator import Command
from pathlib import Path
import time
import numpy as np

from pathlib import Path
import time
from std_msgs.msg import String
import time
import moveit_msgs.msg
import geometry_msgs.msg
from sentence_transformers import SentenceTransformer
from misc import Command

import geometry_msgs.msg

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal
from franka_msgs.msg import ErrorRecoveryActionGoal
from word2number import w2n
from franka_gripper.msg import ( GraspAction, GraspGoal,
                                 HomingAction, HomingGoal,
                                 MoveAction, MoveGoal,
                                 StopAction, StopGoal,
                                 GraspEpsilon )
import textFileHandler as tfh
import copy
import rospy
import moveit_commander
import actionlib
from controller_manager_msgs.srv import SwitchController
import franka_msgs.msg
from actionlib_msgs.msg import GoalStatusArray
from moveit_msgs.msg import RobotTrajectory
from misc import GoalStatus, CommandMode, Command, Controller, MoveDirection, get_relative_orientation
from enum import Enum
from nlihrc.cliport_client import CliportClient
import rospy


from std_msgs.msg import String
# Configs:
# if Rate = 16000, then Chunk = 1280
# if Rate = 8000, then Chunk = 640 (8kHz is not supported with speech model. Can be added later with upsampling before passing to recognizer)

RATE = 16000
CHUNK = 1280
PORT = 50005
ROS_ENABLED = True
DEBUG = False


class Server:

    def __init__(self):
        if ROS_ENABLED:
            # ROS
            import rospy
            from std_msgs.msg import String

            rospy.init_node('text_command_transmitter')
            self.pub = rospy.Publisher("/text_commands", String, queue_size=10) # queue_size gives time for subscriber to process data it gets
            self.pub_priority = rospy.Publisher("/text_commands_priority", String, queue_size=10) # queue_size gives time for subscriber to process data it gets
            
        # UDP Receiver (Handles Android App comm.)
        q = queue.Queue()
        self.udp = UDPReceiver(q, CHUNK, "0.0.0.0", PORT)


         # modes: STEP, DIRECTION AND DISTANCE
        self.mmode = 'STEP'

        # step sizes: LOW, MEDIUM, HIGH
        self.mstep_size = 'MEDIUM'
        # Speech Recognizer (Handles speech to text)
        rec = Recognizer('model', RATE, DEBUG)


        textclassifier = TextClassifier()

        # Command Creator (Handles words to command logic)
        self.commandCreator = CommandCreator(debug_enabled=DEBUG)

        # Start udp thread
        self.udp.start()

        cmd = None
        comd = None
        self.start_robot = False
        self.is_chain_going = False
        self.previous_and = False
        self.prev_cmd = None
        try:

            while True:
                # We need this to make this process shutdown when rospy is being used
                if ROS_ENABLED and rospy.is_shutdown():
                    break
                if q.empty():
                    continue
                data = q.get()

                # Speech to text conversion
                words, number = rec.speech_to_text(data)
                if len(words) > 0:
                    rospy.loginfo(f'Recognized words: {" ".join(words)}')
                    if number is not None:
                        rospy.loginfo(f'Recognized number: {number}')
                else:
                    continue
                # Text to command classification
                sentence = ' '.join(words)
                cmd = textclassifier.find_match(sentence, 0.7)
                self.commandCreator.original_words = words
                comd = self.commandCreator.getCommand(True)

                if comd is not None:
                    #start_robot means start sending commands
                    if cmd.name == "START_ROBOT":
                            print("Converted from ENUM to STRING ",cmd.name) 
                            if comd[0] == 'START':    
                                self.start_robot = True
                                print('Starting with command: ', comd)
                                print(f'Sending configuration to ROS. mode: {self.commandCreator.mode} step_size: {self.commandCreator.step_size}')
                                if ROS_ENABLED:
                                    # Send robot configuration to ROS.
                                    self.pub.publish('MODE ' + self.mmode)
                                    self.pub.publish('STEP SIZE ' + self.mstep_size)
                    elif comd[0] == 'STOP':
                        print('Sending Command to ROS: STOP')
                        if ROS_ENABLED:
                            self.pub_priority.publish('STOP')
                        self.start_robot = False
                        print('Stopping with command: ', comd)

                    # cmds are published after the robot is started
                    if self.start_robot and comd is not None:                
                        if comd[0] !='AGAIN':
                            self.prev_cmd = comd
                            cmdString = ' '.join(map(str, comd))
                            print('Sending Command to ROS: ', cmdString)
                            if ROS_ENABLED:
                                print("Length of command" ,len(comd))
                                if len(comd) > 2:
                                    self.pub.publish(cmdString)
                                else:
                                    self.pub.publish(cmd.name)
                    else:
                        cmd = None
                        comd = None



                    if cmd is None:
                        rospy.logwarn(f"Couldn't classify given {sentence = } to any command")
                        continue
                    # Command to robot
                    if cmd is not None:
                        if cmd in self.commandCreator.cmds:
                            print(cmd)
                            self.commandCreator.cmds[cmd]()
                            print(cmd)       
                            print('Sending Command to ROS: ', cmd)
                            self.commandCreator.run(cmd, number)
                    elif words != None and ROS_ENABLED:
                        self.pub.publish('INVALID')
                    cmd = None
                    comd = None

                
        except Exception as e:
            print('Exception', e)
        finally:
            print('Shutting down...')
            print('Closing UDP thread')
            self.udp.close_thread = True
            self.udp.join()
            print('Clearing queue')
            while not q.empty():
                q.get()





server = Server()
