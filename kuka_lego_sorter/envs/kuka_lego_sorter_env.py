# Kuka Environment
import pybullet as p
import pybullet_data
import gym
# from pybullet_envs.bullet.kuka import Kuka
# from gym.utils import seeding
from kuka_lego_sorter.resources.KukaModel import Kuka
import glob

# required library
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import numpy as np
import math
import time
import random
from typing import Dict
from typing import Tuple

# display option
import cv2  # ..  avoid rendering and overwork 

## this environment has been developed to test find target object which is distinguised by colour using image observation

class KukaLegoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                urdfRoot=pybullet_data.getDataPath(), 
                objRoot = '../kuka_lego_sorter/resources/urdf',
                renders=False, 
                isDiscrete=False, 
                isTarget=False,
                actionRepeat=50,
                maxSteps=20,
                width=128,
                height=128,
                numOfObjs = 3,
                detectingColour = 'red'):

        self._urdfRoot = urdfRoot
        self._objRoot = objRoot
        self._renders = renders
        self._isDiscrete = isDiscrete
        self._isTarget = isTarget
        self._actionRepeat = actionRepeat
        self._maxSteps = maxSteps
        self._timeStep = 1. / 240.
        self._envStepCounter = 0
        self.terminated = 0
        self._numOfobjs = numOfObjs
        self._cam_info = {'dist': 1.3, 'yaw': 180, 'pitch':-30}
        self._width = width
        self._height = height
        self._grasped_colour = None
        colourList = {'red' : 0, 'blue' : 1, 'green' : 2}
        self._detecting_colour = colourList[detectingColour]

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0 :
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_info['dist'], self._cam_info['yaw'], self._cam_info['pitch'], [0.52, -0.2, -0.33])

        else:
            cid = p.connect(p.DIRECT)
        self.seed()
        
        if self._isDiscrete:
            self.action_space = gym.spaces.Discrete(7) # +x, -x, +y, -y, -z, +-endEffectorAngle,
        else:
            self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, )) # dx, dy, dz, R, P, Y, finger
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, 4), dtype=np.uint8) # (h,w) * [r,g,b, depth]

        self._kuka = None
        self._objectIDs = None
        self._target_obj_IDs = None
        self._target_obj_Z = []
        self._end_effector_pos = None
        self._end_effector_orn = None

    def reset(self):
        look = [0.23, 0.2, 0.54]
        distance = 1
        pitch = -56
        yaw = 245
        roll = 0
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        fov = 20.
        aspect = self._width / self._height
        self._near = 0.01
        self._far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self._near, self._far)
        
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.630000,
                0.000000, 0.000000, 0.0, 1.0)
        p.setGravity(0, 0, -10)
        self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        self._attempted_grasp = False
        p.stepSimulation()
        self._objectIDs, self._target_obj_IDs = self._get_objectsID()
        observation = self._get_observation()
        info = {'grasped colour': self._grasped_colour}

        return observation, info

    def _get_objectsID(self):
        # IDs = getObject(self._p, self._numOfobjs)
        urdf_name = ["lego_1_1_", "lego_1_2_", "lego_1_3_", "lego_1_4_", "lego_2_2_"]# "man_", "sign_", "bar_", "corn_"]
        urdf_colour = ["red", "blue", "green"]
        IDs = []
        target_obj_IDs =[]
        for i in range(self._numOfobjs):
            xpos = 0.4 + 0.3*random.random()
            ypos = 0.3 * (random.random() - .5)
            angle = np.pi / 2 + 0.3 * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            nameInd = random.randrange(0,len(urdf_name))
            colourInd = random.randrange(0,len(urdf_colour))
            f_name = urdf_name[nameInd] + urdf_colour[colourInd] + ".urdf"
            urdf_path = os.path.join(self._objRoot, f_name)
            uid = p.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])
            IDs.append(uid)
            if colourInd == self._detecting_colour:
                target_obj_IDs.append(uid)
            for _ in range(200):
                p.stepSimulation()
        return IDs, target_obj_IDs

    def _get_observation(self):
        '''return image '''
        img_arr = p.getCameraImage(width=self._width, 
                                    height=self._height,
                                    viewMatrix=self._view_matrix,
                                    projectionMatrix=self._proj_matrix)  #-> return [width, height, rgbPixels(size=w,h,4), depth(size=w,h), segmentmask(size = w,h)]
        rgb = img_arr[2] # rbg[h, w] = [r, g, b, a]
        image = np.reshape(rgb, (self._height, self._width, 4)) 
        mask = np.reshape(img_arr[4], (self._height, self._width, 1))
        for row in range(self._height):
            for col in range(self._width):
                image[row,col,3] = 0
                for target in (self._target_obj_IDs):
                    if mask[row, col, 0] == target:
                        image[row, col, 3] = 255
        return image # np.concatenate((image, segment), axis=2)  # shape (height, width, 4

    def step(self, action):
        
        dv = 0.05
        # a = 0.05*np.pi/180
        if self._isDiscrete:
            dx = [0, dv, -dv,  0,   0,   0, 0,  0][action]
            dy = [0,  0,   0, dv, -dv,   0, 0,  0][action]
            dz = [0,  0,   0,  0,   0, -dv, 0,  0][action]
            da = [0,  0,   0,  0,   0,   0, dv, -dv][action]
            f = 0.29
        else:
            if self._isTarget:
                dx, dy, dz, da, f  = self._get_action()
            else:
                dx = action[0]
                dy = action[1]
                dz = action[2]
                da = action[3]
                f = action[4]

        return self._step_action([dx, dy, dz, da, f])
        
    def _step_action(self, action):
        self._envStepCounter += 1

        gripperState = self._kuka.getObservation()
        # print(gripperState)
        if gripperState[2] < 0.25:
            fingerAngle = 0.29
            for _ in range(200):
                action = [0, 0, 0, 0, fingerAngle]
                self._kuka.applyAction(action)
                p.stepSimulation()
                fingerAngle -= 0.3/100.
                if fingerAngle < 0:
                    fingerAngle = 0
                if self._renders:
                    time.sleep(self._timeStep)
                if self._termination():
                    break
            for _ in range(200):
                action = [0, 0, 0.001, 0, fingerAngle]
                self._kuka.applyAction(action)
                p.stepSimulation()
                fingerAngle -= 0.3/100.
                if fingerAngle < 0:
                    fingerAngle = 0
                if self._renders:
                    time.sleep(self._timeStep)
                if self._termination():
                    break
            self._attempted_grasp = True
        
        else:
            self._kuka.applyAction(action)
            for _ in range(self._actionRepeat):
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                if self._termination():
                    break

        observation = self._get_observation()
        reward = self._reward()
        done = self._termination()
        info = {'grasped colour': self._grasped_colour}

        return observation, reward, done, info
    
    def _get_action(self):
        gripperState = self._kuka.getObservation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self._objectIDs[0])
        blockOrn = p.getEulerFromQuaternion(blockOrn)
        ''' initially roll has diffence pi '''
        dx = blockPos[0] - gripperState[0]
        dy = blockPos[1] - gripperState[1]
        da = blockOrn[2] - gripperState[5]
        if abs(dx) < 0.05 and abs(dy) < 0.05:
            dz = blockPos[2] - gripperState[2]
        else: 
            dz = 0
        if abs(dz) > 0 and abs(dz) < 0.25:
            f = 1 
            self._attempted_grasp += 1
        else:
            f = 0
        print([dx,dy,dz,da, f])
        return dx, dy, dz, da, f
        
    def _reward(self):
        ''' 
        return reward
        if there is no more targeted colour object, return 100
        else negative total distance between gripper and all targeted colour objects
         '''
    
        reward = 0
        if self._attempted_grasp:
            for target in self._target_obj_IDs:
                objPos, _ = p.getBasePositionAndOrientation(target)
                if objPos[2] >= 0.3:
                    p.removeBody(target)
                    self._target_obj_IDs.remove(target)
            self._attempted_grasp = False
        kukaState = self._kuka.getObservation()
        self._end_effector_pos = [kukaState[0], kukaState[1], kukaState[2]]
        dis = 0
        for target in self._target_obj_IDs:
            objPos, _ = p.getBasePositionAndOrientation(target)
            dis +=  np.sqrt( (objPos[0] - self._end_effector_pos[0])**2 + 
                        (objPos[1] - self._end_effector_pos[1])**2 +
                        (objPos[2] - self._end_effector_pos[2])**2 )
        reward -= dis

        if len(self._target_obj_IDs) == 0: 
            reward = 100

        return reward
    
    def _termination(self): 
        return len(self._target_obj_IDs) == 0 or self._envStepCounter >= self._maxSteps

    def render(self):
        pass

    def close(self):
        p.disconnect()
        
    def seed(self, seed=None): 
        pass

    