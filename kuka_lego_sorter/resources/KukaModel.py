import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data


class Kuka:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 1
    self.useOrientation = 1
    self.kukaEndEffectorIndex = 6
    self.kukaGripperIndex = 7
    #lower limits for null space
    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    #upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    #joint ranges for null space
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    #restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    #joint damping coefficents
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001, 0.00001, 0.00001
    ]
    # self.flag = True
    self.reset()

  def reset(self):
    objects = p.loadSDF(os.path.join(self.urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf"))
    self.kukaUid = objects[0]
    #for i in range (p.getNumJoints(self.kukaUid)):
    #  print(p.getJointInfo(self.kukaUid,i))
    p.resetBasePositionAndOrientation(self.kukaUid, [-0.100000, 0.000000, 0.070000],
                                      [0.000000, 0.000000, 0.000000, 1.000000])
    self.jointPositions = [
        0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
        -0.3, 0.000000, -0.000043, 0.3, 0.000000, -0.000200
    ]
    self.numJoints = p.getNumJoints(self.kukaUid)
    for jointIndex in range(self.numJoints):
      p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
      p.setJointMotorControl2(self.kukaUid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)

    # self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.640000,
    #                           0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)
    observation = self.getObservation()
    self.endEffectorPos = [observation[0], observation[1], observation[2]]
    self.endEffectorOrn = [observation[3], observation[4], observation[5]]
    # if self.flag:
    #   print(f"""
    #           gripper pos :  {self.endEffectorPos}
    #           gripper orientation : {self.endEffectorOrn}""")
    self.fingerAngle = 0.29
    self._attempted_grasp = 0 

    self.motorNames = []
    self.motorIndices = []

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.kukaUid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.kukaUid, self.kukaGripperIndex)
    pos = state[0]
    orn = state[1]
    euler = p.getEulerFromQuaternion(orn)
    
    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation
  
  def _normalize(self, angle):
    '''
    return value between -PI and PI
    '''
    if angle < -np.pi or angle > np.pi:
      angle = np.arcsin(np.sin(angle))
    return angle

  def get_fingerAngle(self):
    self.fingerAngle = (abs(p.getJointState(self.kukaUid, 8)[0]) + abs(p.getJointState(self.kukaUid, 11)[0]))/2
    return self.fingerAngle

  
  def applyAction(self, motorCommands):
    if (self.useInverseKinematics):

      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      da = motorCommands[3]
      self.fingerAngle = motorCommands[4]

      # kukaState = self.getObservation()
      # self.endEffectorPos = [kukaState[0], kukaState[1], kukaState[2]]
      # self.endEffectorOrn = [kukaState[3], kukaState[4], kukaState[5]]

      self.endEffectorPos[0] = self.endEffectorPos[0] + dx 
      if (self.endEffectorPos[0] > 0.65):
        self.endEffectorPos[0] = 0.65
      if (self.endEffectorPos[0] < 0.50):
        self.endEffectorPos[0] = 0.50
      self.endEffectorPos[1] = self.endEffectorPos[1] + dy
      if (self.endEffectorPos[1] < -0.17):
        self.endEffectorPos[1] = -0.17
      if (self.endEffectorPos[1] > 0.22):
        self.endEffectorPos[1] = 0.22
      self.endEffectorPos[2] = self.endEffectorPos[2] + dz
      if (self.endEffectorPos[2] < 0.23):
        self.endEffectorPos[2] = 0.23

      self.endEffectorOrn[0] = np.pi#self._normalize(R + observation[3])
      self.endEffectorOrn[1] = 0 #self._normalize(P + observation[4])
      self.endEffectorOrn[2] = self._normalize(self.endEffectorOrn[2] + da)
      
      pos = self.endEffectorPos
      orn = p.getQuaternionFromEuler([self.endEffectorOrn[0], self.endEffectorOrn[1], self.endEffectorOrn[2]])

      ''' get joint poses '''
      if (self.useNullSpace == 1):
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaGripperIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp)
        else:
          jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaGripperIndex,
                                                    pos,
                                                    lowerLimits=self.ll,
                                                    upperLimits=self.ul,
                                                    jointRanges=self.jr,
                                                    restPoses=self.rp)
      else:
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaGripperIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd)
        else:
          jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaGripperIndex, pos)

      ''' set joint poses [ARM] '''
      if (self.useSimulation):
        for i in range(self.kukaGripperIndex+1):
          p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)

      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.kukaUid, i, jointPoses[i])
      ''' set joint poses [GRIPPER] '''
      if self.fingerAngle >= 0.28:
        # p.setJointMotorControl2(self.kukaUid,
        #                         7,
        #                         p.POSITION_CONTROL,
        #                         targetPosition=jointPoses[7], #self.endEffectorOrn[2],#
        #                         force=self.maxForce)
        p.setJointMotorControl2(self.kukaUid,
                                8,
                                p.POSITION_CONTROL,
                                targetPosition=-self.fingerAngle,
                                force=self.maxForce)
        p.setJointMotorControl2(self.kukaUid,
                                11,
                                p.POSITION_CONTROL,
                                targetPosition=self.fingerAngle,
                                force=self.maxForce)

        p.setJointMotorControl2(self.kukaUid,
                                10,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.fingerTipForce)
        p.setJointMotorControl2(self.kukaUid,
                                13,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.fingerTipForce)
      else:
        # p.setJointMotorControl2(self.kukaUid,
        #                         7,
        #                         p.POSITION_CONTROL,
        #                         targetPosition=jointPoses[7], #self.endEffectorOrn[2],#
        #                         force=self.maxForce)
        p.setJointMotorControl2(self.kukaUid,
                                8,
                                p.POSITION_CONTROL,
                                targetPosition=-self.fingerAngle,
                                force=self.maxForce)
        p.setJointMotorControl2(self.kukaUid,
                                11,
                                p.POSITION_CONTROL,
                                targetPosition=self.fingerAngle,
                                force=self.maxForce)

        p.setJointMotorControl2(self.kukaUid,
                                10,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.maxForce)
        p.setJointMotorControl2(self.kukaUid,
                                13,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.maxForce)
    else:
      for action in range(len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.kukaUid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)