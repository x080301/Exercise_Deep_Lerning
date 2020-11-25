print(__file__)

import sys
import os

print(os.path.realpath(__file__))
print(os.path.split(os.path.realpath(__file__)))
#sys.path.append(os.path.split(os.path.realpath(__file__))[0])#添加路径，这个是临时的