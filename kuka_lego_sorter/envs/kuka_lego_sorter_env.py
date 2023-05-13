# Kuka Environment
import pybullet as p
import pybullet_data
import gym
from pybullet_envs.bullet.kuka import Kuka
# from gym.utils import seeding
import glob

# required library
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import numpy as np
import math
import time
import random

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
                actionRepeat=1,
                maxSteps=1000,
                width=120,
                height=120,
                numOfObjs = 1):

        self._urdfRoot = urdfRoot
        self._objRoot = objRoot
        self._renders = renders
        self._isDiscrete = isDiscrete
        self._actionRepeat = actionRepeat
        self._maxSteps = maxSteps
        self._timeStep = 1. / 240.
        self._envStepCounter = 0
        self.terminated = 0
        self._numOfobjs = numOfObjs
        self._cam_info = {'dist': 1.3, 'yaw': 180, 'pitch': -40}
        self._width = width
        self._height = height
        self._grasped_colour = None
        

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0 :
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])

        else:
            cid = p.connect(p.DIRECT)
        self.seed()
        
        if self._isDiscrete:
            self.action_space = gym.spaces.Discrete(7) # +x, -x, +y, -y, +z, -z, endEffectorAngle, fingergrasping
        else:
            self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, )) # dx, dy, dz, da, finger
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8) # (h,w) * [r,g,b, Mask]
        self._kuka = None
        self._objectIDs = None
        self._end_effector_pos = None
        self._end_effector_orn = None

    def _get_objectsID(self):
        urdf_name = ['lego_1_2_', 'lego_2_2_', 'lego_1_3_', 'man_']
        urdf_colour = ['red','blue','green']
        IDs = []
        for i in range(self._numOfobjs):
            xpos = 0.4 + 0.3*random.random()
            ypos = 0.3 * (random.random() - .5)
            angle = np.pi / 2 + 0.3 * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            randInd = random.randrange(0,len(urdf_name))
            # urdf_path = os.path.join(self._objRoot, urdf_name[0], urdf_colour[0], '.urdf')
            urdf_path =  os.path.join(self._objRoot, 'lego_1_2_red.urdf')
            uid = p.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])
            IDs.append(uid)
            for _ in range(200):
                p.stepSimulation()
        return IDs

    def _get_observation(self):
        '''return image '''
        img_arr = p.getCameraImage(width=self._width, 
                                    height=self._height,
                                    viewMatrix=self._view_matrix,
                                    projectionMatrix=self._proj_matrix)  #-> return [width, height, rgbPixels(size=w,h,4), depth(size=w,h), segmentmask(size = w,h)]
        rgb = img_arr[2] # rbg[h, w] = [r, g, b, a]
        image = np.reshape(rgb, (self._height, self._width, 4))     
        segment = np.reshape(img_arr[3], (self._height, self._width, 1))   #sement[h,w] = object ID + (linkIndes+1) << 24

        return image[:, :, :3] # np.concatenate((image, segment), axis=2)  # shape (height, width, 4)

    def step(self, action):
        
        # dv = 0.05
        # if self._isDiscrete:
        #     dx = [dv, -dv,  0,   0,  0,   0, 0, 0][action]
        #     dy = [ 0,   0, dv, -dv,  0,   0, 0, 0][action]
        #     dz = [ 0,   0,  0,   0, dv, -dv, 0, 0][action]
        #     da = [ 0,   0,  0,   0,  0,   0, 2*np.pi*dv, 0][action]
        #     f =  1 if action == 7 else 0
        # else:
        #     dx = action[0]
        #     dy = action[1]
        #     dz = action[2]
        #     da = action[3]
        #     f = action[4]

        ##temp
        obs = self._kuka.getObservation()
        gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
        gripperPos = gripperState[0]
        gripperOrn = gripperState[1]
        blockPos, blockOrn = p.getBasePositionAndOrientation(self._objectIDs[0])

        invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)

        blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                    blockPos, blockOrn)
        blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
        dx = -blockPosInGripper[0]
        dy = -blockPosInGripper[1]
        da = 0#blockEulerInGripper[2]
        dz = 0
        f = 0.2

        return self._step_action([dx,dy,dz,da,f])
        
    def _step_action(self, action):

        self._envStepCounter += 1
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

    def reset(self):
        look = [0.23, 0.2, 0.54]
        distance = 1
        pitch = -56
        yaw = 245
        roll = 0
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        fov = 20.
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                0.000000, 0.000000, 0.0, 1.0)
        p.setGravity(0, 0, -10)
        self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._objectIDs = self._get_objectsID()
        self._observation = self._get_observation()
        info = {'grasped colour': self._grasped_colour}
        return np.array(self._observation), info
        
    def _reward(self):
        ''' return negative distance between robot end effector and red object '''
        numOfRedObj = int(self._numOfobjs/3)+1
        minDis = 100000; redInd = 0
        self._end_effector_pos = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)[0]
        for _ in range(numOfRedObj):
            objPos, _ = p.getBasePositionAndOrientation(self._objectIDs[redInd])
            redInd += 1
            dis = np.sqrt((objPos[0] - self._end_effector_pos[0])**2 + (objPos[1] - self._end_effector_pos[1])**2 )
            if dis < minDis:
                mindDis = dis

        if minDis <= 0.01:
            reward = 50
        else:
            reward = -minDis
        return reward

    def _termination(self): 
        self._end_effector_pos = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)[0]
        return self._end_effector_pos[2] <= 0.1 or self._envStepCounter >= self._maxSteps

    def render(self):
        pass

    def close(self):
        p.disconnect()
        
    def seed(self, seed=None): 
        pass

    