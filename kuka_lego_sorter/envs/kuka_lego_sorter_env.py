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
# from pkg_resources import parse_version

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
                actionRepeat=20,
                maxSteps=100,
                width=256,
                height=256,
                numOfObjects = 3,
                targetColour = 'red',
                observeImage=False,
                velocity=0.05):

        self._urdfRoot = urdfRoot
        self._objRoot = objRoot
        self._renders = renders
        self._isDiscrete = isDiscrete
        self._dv = velocity
        self._actionRepeat = actionRepeat
        self._maxSteps = maxSteps
        self._timeStep = 1. / 240.
        self._envStepCounter = 0
        self._cam_info = {'dist': 1.3, 'yaw': 180, 'pitch':-30}
        self._width = width
        self._height = height
        self._observeImage = observeImage
        self._numOfObjects = numOfObjects if numOfObjects <= 10 else 10    #  Maximum 10
        self._draw_frame = False
        self.vid = []

        ''' Client '''
        self._p = p
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if self.cid < 0 :
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_info['dist'], self._cam_info['yaw'], self._cam_info['pitch'], [0.52, -0.2, -0.33])

        else:
            self.cid = p.connect(p.DIRECT)
        self.seed()
        
        ''' Action Space '''
        if self._isDiscrete:
            self.action_space = gym.spaces.Discrete(7) # -z, +x, -x, +y, -y, +endEffectorYaw, -endEffectorYaw
        else:
            self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, )) # dx, dy, dz, R, P, Y, finger
        ''' Observation Space '''
        if self._observeImage:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)  # (h,w) * [r,g,b, depth]
        else:
            self.observation_space = (gym.spaces.Box(low=-5., high=5., shape=(6, ), dtype=np.float32))     
                                                        # gym.spaces.Box(low=-np.pi, high=np.pi, shape=(3, ), dtype=np.float32))) # # number of left target bricks # position     orientation ( end-effector pos related the closest target brick )

        ''' Kuka environment '''
        self._kuka = None
        self._gripper_state = None

        ''' object environment '''
        self._objectIDs = None
        self._target_IDs = None
        self._closest_target_ID = None
        self._target_number = 0
        self.urdf_colour = ["red", "green", "blue"]
        self.urdf_name = ["lego_1_1_", "lego_1_2_", "lego_1_3_", "lego_1_4_", "lego_2_2_"]#, "man_", "sign_", "bar_", "corn_"]
        self.target_colour_index = self.urdf_colour.index(targetColour)
        
        self.reset()

    def reset(self):
        self.info = []
        ''' Set rendering camera '''
        look = [0.53, 0, 0.2]
        distance = 2.5
        pitch =-30 
        yaw = 90
        roll = 0
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        fov = 20.
        aspect = self._width / self._height
        self._near = 0.01
        self._far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self._near, self._far)
        ''' Set simulation environment'''
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.630000,
                0.000000, 0.000000, 0.0, 1.0)
        p.setGravity(0, 0, -10)
        ''' Set Kuka '''
        self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        self._attempted_grasp = 0
        self._prev_attempted_grasp = 0
        self._success_target_grasping = None
        self._attemption_results = []
        p.stepSimulation()
        self._gripper_state = self._kuka.getObservation()
        ''' Set Objects'''
        self._objectIDs, self._target_IDs = self._get_objectsIDs()
        self._closest_target_ID = self._find_closest_target(self._gripper_state[:3])
        observation = self._get_observation()

        self.info = {'target number': self._target_number, 'grasping attemption results': self._attemption_results, 'video': self.vid}
        return observation, self.info

    def step(self, action):
        dv = self._dv
        if self._isDiscrete:
            dx = [  0, dv, -dv,  0,   0,  0,   0][action]
            dy = [  0,  0,   0, dv, -dv,  0,   0][action]
            dz = [-dv,  0,   0,  0,   0,  0,   0][action]
            da = [  0,  0,   0,  0,   0, dv, -dv][action]
            f = 0.29
        else:
            dx = action[0]
            dy = action[1]
            dz = action[2]
            da = action[3]
            f = action[4]

        return self._step_action([dx, dy, dz, da, f])
        
    def _step_action(self, action):
        self._envStepCounter += 1

        self._gripper_state = self._kuka.getObservation()
        if self._draw_frame:
            p.addUserDebugLine([0,0,0], [0.1, 0, 0], 
                    lineColorRGB=[1, 0, 0], 
                    lineWidth=1., 
                    lifeTime=15,
                    parentObjectUniqueId=self._kuka.kukaUid, 
                    parentLinkIndex=7)
            p.addUserDebugLine([0,0,0], [0, 0.1, 0], 
                    lineColorRGB=[0, 1, 0], 
                    lineWidth=1., 
                    lifeTime=15,
                    parentObjectUniqueId=self._kuka.kukaUid, 
                    parentLinkIndex=7)
            p.addUserDebugLine([0,0,0], [0, 0, 0.1], 
                    lineColorRGB=[0, 0, 1], 
                    lineWidth=1., 
                    lifeTime=15,
                    parentObjectUniqueId=self._kuka.kukaUid, 
                    parentLinkIndex=7)

        if self._gripper_state[2] < 0.25:
            self._grasping_action()
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
        self.info = {'target number': self._target_number, 'grasping attemption results': self._attemption_results, 'video': self.vid}
        return observation, reward, done, self.info

    def _grasping_action(self):
        ''' enter grasping mode'''
        self._attempted_grasp += 1

        fingerAngle = self._kuka.get_fingerAngle()

        for t in range(400):
            action = [0, 0, 0, 0, fingerAngle]
            self._kuka.applyAction(action)
            p.stepSimulation()
            fingerAngle -= 0.3/100.
            if fingerAngle < 0:
                fingerAngle = 0        
            if self._renders:
                time.sleep(self._timeStep)
            if  t%15 == 0:
                img = p.getCameraImage(width=self._width, 
                                        height=self._height,
                                        viewMatrix=self._view_matrix,
                                        projectionMatrix=self._proj_matrix)
                self.vid.append(np.reshape(img[2], (self._height, self._width, 4)))
        for t in range(400):
            action = [0, 0, 0.00085, 0, fingerAngle]
            self._kuka.applyAction(action)
            p.stepSimulation()
            fingerAngle -= 0.3/100.
            if fingerAngle < 0:
                fingerAngle = 0
            if self._renders:
                time.sleep(self._timeStep)
            if t%10 == 0 :
                img = p.getCameraImage(width=self._width, 
                                            height=self._height,
                                            viewMatrix=self._view_matrix,
                                            projectionMatrix=self._proj_matrix)
                self.vid.append(np.reshape(img[2], (self._height, self._width, 4)))
        ''' check if grasped target '''
        self._gripper_state = self._kuka.getObservation()
        gripperPos = self._gripper_state[:3]
        for target in self._target_IDs:
            objPos, _ = p.getBasePositionAndOrientation(target)
            if objPos[2] > 0.2: 
                if np.linalg.norm(np.array(gripperPos) - np.array(objPos)) <= 0.25:        # if within length of fingers
                    self._success_target_grasping = target
                    break
        
    def _reward(self):
        ''' 
        return reward
        if there is no more targeted colour object, return 100
        else negative total distance between gripper and all targeted colour objects
        
        Conditions 1)
                        no target objects in the first place    => 0      >> terminate to reset env 
        Conditions 2)
                        continously close/far to the target     => -1
        Conditions 3) 
                        attempted grasping but failed           => -10.
        Conditions 4) 
                        successfully grasping target            => 100 * total target number / remain target number
        Conditions 5)   
                        successfully grasping LAST target       => 100 * total target number        >> all atepmtions successed, maximise reward. if earlier success ultimate Max // left attemps = total objects number - current attempstions

        '''
        if  self._target_number == 0:
            return 0

        if self._success_target_grasping is not None:
            p.removeBody(self._success_target_grasping)
            self._target_IDs.remove(self._success_target_grasping)
            self._success_target_grasping = None
            if bool(self._target_IDs):
                print(f"""
                target_number : {self._target_number}
                attepmted number : {self._attempted_grasp}
                SUCCESS : {True}
                return value : {100. / self._target_number}
                """)
                return 100. * self._target_number / len(self._target_IDs)
            else:
                print(f"""
                target_number : {self._target_number}
                attepmted number : {self._attempted_grasp}
                SUCCESS : {True} and LAST
                return value : { 100.}
                """)
                return 100. * (self._target_number + 1)
        
        else:
            
            # self._gripper_state = self._kuka.getObservation()
            # self._closest_target_ID = self._find_closest_target(self._gripper_state[:3])
            # objPos, _ = p.getBasePositionAndOrientation(self._closest_target_ID)

            if self._attempted_grasp == self._prev_attempted_grasp:
            #     ''' no attempted'''
                return -1.  # * np.exp(np.linalg.norm(np.array(self._gripper_state[:3]) - np.array(objPos)))
            else:
            #     print(f"""
            #     target_number : {self._target_number}
            #     attepmted number : {self._attempted_grasp}
            #     SUCCESS : {False}
            #     return value : {-1. * np.exp( self._attempted_grasp - (self._target_number - len(self._target_IDs)) ) }
            #    """)

            #     ''' if attpmted but failed '''
                self._prev_attempted_grasp = self._attempted_grasp
                return -10. #*  np.exp( self._attempted_grasp - (self._target_number - len(self._target_IDs)) ) 
    
    def _termination(self): 
        '''ternimate '''
        return len(self._target_IDs) == 0 or self._envStepCounter >= self._maxSteps 

    def render(self):
        pass

    def close(self):
        p.disconnect()
        
    def seed(self, seed=None): 
        pass

    # if parse_version(gym.__version__) < parse_version('0.9.6'):
    #     _reset = reset
    #     _step = step

    def _get_observation(self):
        ''' return  
            1) Observe Image
            image masked excpeted target blocks and the robot    /// TODO: if wants depth?
            2) Observe coordinate
            number of left targets,[ end-effector's position ], [end-effector's orientation in Euler] which is related the cloaset target pos [x, y, z, yaw]  '''
        if self._observeImage:
            img_arr = p.getCameraImage(width=self._width, 
                                        height=self._height,
                                        viewMatrix=self._view_matrix,
                                        projectionMatrix=self._proj_matrix)  #-> return [width, height, rgbPixels(size=w,h,4), depth(size=w,h), segmentmask(size = w,h)]
            rgb = img_arr[2] # rbg[h, w] = [r, g, b, a]
            # depth = np.reshape(img_arr[3], (self._height, self._width)) 
            image = np.reshape(rgb, (self._height, self._width, 4)) 
            # mask = np.reshape(img_arr[4], (self._height, self._width, 1))
            # for row in range(self._height):
            #     for col in range(self._width):
            #         image[row,col, 3] = 0
            #         if mask[row, col, 0] == self._kuka.kukaUid:
            #             image[row, col, 3] = 255
            #         else:
            #             for target in (self._target_IDs):
            #                 if mask[row, col, 0] == target:
            #                     image[row, col, 3] = 255

            # for row in range(self._height):
            #     for col in range(self._width):
            #         if image[row, col, 3] == 0:
            #             image[row, col, :] = 0
            return image[:, :, :3] # np.concatenate((image, segment), axis=2)  # shape (height, width, 4
        
        else:
            img = p.getCameraImage(width=self._width, 
                                        height=self._height,
                                        viewMatrix=self._view_matrix,
                                        projectionMatrix=self._proj_matrix)
            self.vid.append(np.reshape(img[2], (self._height, self._width, 4)))
        
            if self._target_number == 0:
                return [0, 0, 0, 0, 0, 0]

            self._gripper_state = self._kuka.getObservation()
            self._closest_target_ID = self._find_closest_target(self._gripper_state[:3])

            if self._closest_target_ID is None:     # removed target, return NULL
                return [0, 0, 0, 0, 0, 0]
            else:
                objPos, objOrn = p.getBasePositionAndOrientation(self._closest_target_ID)
                gripperPos, gripperOrn = self._gripper_state[:3], p.getQuaternionFromEuler(self._gripper_state[3:])
                invObjPos, invObjOrn = p.invertTransform(objPos, objOrn)
                observedPos, observedOrn = p.multiplyTransforms(invObjPos, invObjOrn, gripperPos, gripperOrn)
                observedOrnEuler = p.getEulerFromQuaternion(observedOrn)
                observed = []
                observed.extend(list(observedPos))
                observed.extend(list(observedOrnEuler))
                return observed


    def _get_objectsIDs(self):
        IDs = []
        target_IDs =[]
        random_colour_weights = np.ones((len(self.urdf_colour), 1)) * 0.5 / self._numOfObjects
        random_colour_weights[self.target_colour_index] = 0.5           
        
        for i in range(self._numOfObjects):
            xpos = 0.4 + 0.3*random.random()
            ypos = 0.3 * (random.random() - .5)
            angle = np.pi / 2 + 0.3 * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])

            # f_name = random.choices(self.urdf_name)[0] + random.choices(self.urdf_colour, random_colour_weights)[0] + ".urdf"
            f_name = random.choices(self.urdf_name)[0] + self.urdf_colour[i] + ".urdf"
            urdf_path = os.path.join(self._objRoot, f_name)
            uid = p.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])
            IDs.append(uid)
            if self.urdf_colour[self.target_colour_index] in f_name:
                target_IDs.append(uid)
            for _ in range(200):
                p.stepSimulation() 

        self._target_number = len(target_IDs)
        return IDs, target_IDs

    def _find_closest_target(self, gripperPos):
        closestTarget = None
        minDis = 1000
        for target in self._target_IDs:
            objPos, _ = p.getBasePositionAndOrientation(target)
            dis = np.linalg.norm(np.array(gripperPos) - np.array(objPos))
            if dis < minDis:
                minDis = dis
                closestTarget = target

        if self._draw_frame and closestTarget is not None:
            p.addUserDebugLine([0,0,0], [0.1, 0, 0], 
                    lineColorRGB=[1, 0, 0], 
                    lineWidth=1., 
                    lifeTime=15,
                    parentObjectUniqueId=closestTarget, 
                    parentLinkIndex=-1)
            p.addUserDebugLine([0,0,0], [0, 0.1, 0], 
                    lineColorRGB=[0, 1, 0], 
                    lineWidth=1., 
                    lifeTime=15,
                    parentObjectUniqueId=closestTarget, 
                    parentLinkIndex=-1)
            p.addUserDebugLine([0,0,0], [0, 0, 0.1], 
                    lineColorRGB=[0, 0, 1], 
                    lineWidth=1., 
                    lifeTime=15,
                    parentObjectUniqueId=closestTarget, 
                    parentLinkIndex=-1)
        return closestTarget