import gym
import pybullet as p
import kuka_lego_sorter
from PIL import Image
import cv2

env = gym.make("KukaLegoSorter-v0", renders=True)


frame = env.reset()
frames = []
frames.append(frame)

for t in range(100):
    print(f'\rtimestep {t}...', end='')
    frame, reward, done, info = env.step([0,0, 0, 0, 0.2])
    frames.append(frame)

env.close()
Image.fromarray(frame).show()

# cv2.namedWindow('vid', cv2.WINDOW_NORMAL)
# cnt = len(frames)
# idx = 0
# while idx<cnt:
#     cv2.imshow('vid',frames[idx])
#     if cv2.waitKey(100) == 27:
#         break
#     idx += 1
    # if idx >= cnt:
    #     idx = 0

# p.disconnect()