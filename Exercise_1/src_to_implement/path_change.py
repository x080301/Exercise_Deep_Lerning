import os

#print(os.getcwd())
#print(os.path.split(os.path.realpath(__file__))[0])
os.chdir(os.path.split(os.path.realpath(__file__))[0])  #更改路径，''里面为更改的路径
#print(os.getcwd())
#sys.path.append(os.path.split(os.path.realpath(__file__))[0])#添加路径，这个是临时的
