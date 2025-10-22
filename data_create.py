import os
import numpy as np
import pandas as pd

def CSI_CI_generate(locatMat1, locatMat2, num):
    '''
    生成CSI矩阵,瑞利衰落信道模型
    输出：num X locatMat1.shape[1]的矩阵
        每列对应一个用户的CSI
    '''
    location1 = locatMat1.T[:, np.newaxis, :]  # Shape: (UUnum, 1, 3)
    location2 = locatMat2.T[np.newaxis, :, :]  # Shape: (1, DUnum, 3)
    distanceMat = np.linalg.norm(location1 - location2, axis=2)  # Shape: (UUnum, DUnum)
    RandomMat = (np.random.randn(locatMat1.shape[1], locatMat2.shape[1])
                + 1j * np.random.randn(locatMat1.shape[1], locatMat2.shape[1]))/np.sqrt(2) 
    CSIMat = (RandomMat * (np.sqrt(10**(-3)*(distanceMat**-2.67))
                        ))
    return CSIMat

def CSI_generate(locatMat1, locateMat2, num):
    '''
    生成CSI矩阵,瑞利衰落信道模型
    输出：num X locatMat1.shape[1]的矩阵
        每列对应一个用户的CSI
    '''
    distanceMat = np.linalg.norm(locatMat1-locateMat2,axis=0)
    RandomMat = (np.random.randn(num, locatMat1.shape[1])
                + 1j * np.random.randn(num, locatMat1.shape[1]))/np.sqrt(2) 
    CSIMat = (RandomMat * (np.sqrt(10**(-3)*(distanceMat**-2.67))
                        ))
    return CSIMat

class BSparameter:
    '''
    基站参数设置 num_trans = real, num_rece = real , powerBudget=real ,location=1Darr(3)
    '''
    def __init__(self, num_trans, num_rece, powerBudget,location):
        self.num_trans = num_trans
        self.num_rece = num_rece
        self.powerBudget = powerBudget
        self.location=location.reshape(-1,1);
    def info(self):
        print(self.__dict__)

class TAparameter:
    '''
    目标参数设置 num= real, angleRange = 1Darr(2), powerGainRange = 1Darr(2)
    返回值: angleArr = 1Darr(M), powerGainArr = 1Darr(M)
    '''
    def __init__(self, num, angleRange, powerGainRange):
        self.num = num
        self.angleRange = angleRange
        self.powerGain = powerGainRange
        self.angleArr = None
        self.powerGainArr = None
    def info(self):
        print(self.__dict__)
    def random(self):
        '''
        随机生成TA参数
        '''
        self.angleArr = np.random.uniform(self.angleRange, self.num)
        self.powerGainArr = np.random.uniform(self.powerGain, self.num)
        return self.angleArr, self.powerGainArr
        
class INparameter:
    '''
    干扰参数设置 num= real, angleRange = 1Darr(2), powerGainRange = 1Darr(2)
    返回值: angleArr = 1Darr(M), powerGainArr = 1Darr(M)
    '''
    def __init__(self, num, angleRange, powerGainRange):
        self.num = num
        self.angleRange = angleRange
        self.powerGain = powerGainRange
        self.angleArr = None
        self.powerGainArr = None
    def info(self):
        print(self.__dict__)
    def random(self):
        '''
        随机生成IN参数
        '''
        self.angleArr = np.random.uniform(self.angleRange, self.num)
        self.powerGainArr = np.random.uniform(self.powerGain, self.num)
        return self.angleArr, self.powerGainArr
    
class UUparameter:
    '''
    上行参数设置 num= real, powerBudget= real, locatRange = 2Darr(3,2)
    返回值: powerBudgetArr = 1Darr(N), locatMat = 2Darr(3,UUnum)
    '''
    def __init__(self, num, powerBudget, locatRange):
        self.num = num
        self.powerBudget = powerBudget
        self.xlocatRange = locatRange[0]
        self.ylocatRange = locatRange[1]
        self.zlocatRange = locatRange[2]
        self.powerBudgetArr = None
        self.CSIMat = None
        self.locatMat = None
    def random(self):
        '''
        随机生成UU参数
        '''
        self.powerBudgetArr=np.full(self.num, self.powerBudget)
        locatMat = np.stack([np.random.uniform(self.xlocatRange[0], self.xlocatRange[1], self.num),
                                 np.random.uniform(self.ylocatRange[0], self.ylocatRange[1], self.num),
                                 np.random.uniform(self.zlocatRange[0], self.zlocatRange[1], self.num)],
                                 axis=1).T
        self.locatMat = locatMat
        return self.powerBudgetArr,self.locatMat
    def info(self):
        print(self.__dict__)
class DUparameter:
    '''
    下行参数设置 num= real, locatRange = 2Darr(3,2)
    返回值: locatMat = 2Darr(3,DUnum)
    '''
    def __init__(self, num, locatRange):
        self.num = num
        self.xlocatRange = locatRange[0]
        self.ylocatRange = locatRange[1]
        self.zlocatRange = locatRange[2]
        self.locatMat = None
        self.CSImat = None
    def random(self):
        '''
        随机生成DU参数
        '''
        locatMat = np.stack([np.random.uniform(self.xlocatRange[0], self.xlocatRange[1], self.num),
                                 np.random.uniform(self.ylocatRange[0], self.ylocatRange[1], self.num),
                                 np.random.uniform(self.zlocatRange[0], self.zlocatRange[1], self.num)],
                                 axis=1).T
        self.locatMat = locatMat
        return self.locatMat
    def info(self):
        print(self.__dict__)
class systemParameter:
    def __init__(self, BS:BSparameter, TA:TAparameter, IN:INparameter, UU:UUparameter, DU:DUparameter):
        """
        初始化系统参数，包含所有子组件参数
        直接传入已实例化的组件对象
        :param BS: BSparameter实例
        :param TA: TAparameter实例
        :param IN: INparameter实例
        :param UU: UUparameter实例
        :param DU: DUparameter实例
        """
        self.BS = BS
        self.TA = TA
        self.IN = IN
        self.UU = UU
        self.DU = DU
        self.CIMat = None
        self.noise2DI = None
        self.noise2BS = None
    def random(self):
        """为所有组件生成随机参数"""
        self.TA.random()
        self.IN.random()
        self.UU.random()
        self.DU.random()
    def envParaSet(self, noise2DI, noise2BS):
        '''
        设置环境参数
        '''
        self.noise2DI = noise2DI
        self.noise2BS = noise2BS

    def data_create(self):
        '''
        生成随机场景数据
        '''
        #UU的CSI矩阵 
        self.UU.CSIMat = CSI_generate(self.UU.locatMat, self.BS.location, self.BS.num_rece)
        #DU的CSI矩阵
        self.DU.CSIMat = CSI_generate(self.DU.locatMat, self.BS.location, self.BS.num_trans)
        #CI矩阵,其中CIMat[i,j]表示第i个DU与第j个UU之间的CSI
        self.CIMat = CSI_CI_generate(self.UU.locatMat, self.DU.locatMat, self.DU.num)
    def info(self):
        print(self.__dict__)
        print("BS\n")
        BS.info()
        print("TA\n")
        TA.info()
        print("IN\n")
        IN.info()
        print("UU\n")
        UU.info()
        print("DU\n")
        DU.info()


if __name__ == "__main__":
    #测试代码
    BS = BSparameter(6, 6, 100, np.array([0,0,0]))
    TA = TAparameter(5, np.array([0,180]), np.array([-30,-30]))
    IN = INparameter(3, np.array([0,180]), np.array([-30,-30]))
    UU = UUparameter(4, 10, np.array([[0,100],[0,100],[0,10]]))
    DU = DUparameter(4, np.array([[0,100],[0,100],[0,10]]))
    BS.info()
    system = systemParameter(BS,TA,IN,UU,DU)
    system.random()
    system.data_create()
    print(system.__dict__)