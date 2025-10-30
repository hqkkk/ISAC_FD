import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
def CSI_CI_generate(locatMat1, locatMat2):
    '''
    UU和DU之间相互干扰的CSI矩阵
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
        self.BS = BS
        self.TA = TA
        self.IN = IN
        self.UU = UU
        self.DU = DU
        self.CIMat = None
        self.noise2DI = None
        self.noise2BS = None
    def random(self):
        self.TA.random()
        self.IN.random()
        self.UU.random()
        self.DU.random()
    def envParaSet(self, noise2DI, noise2BS):
        self.noise2DI = noise2DI
        self.noise2BS = noise2BS
    def data_create(self):
        '''
        生成随机场景数据
        '''
        self.TA.random()
        self.IN.random()
        self.UU.random()
        self.DU.random()
        #UU的CSI矩阵 
        UUCSIMat = CSI_generate(self.UU.locatMat, self.BS.location, self.BS.num_rece)
        #DU的CSI矩阵
        DUMat = CSI_generate(self.DU.locatMat, self.BS.location, self.BS.num_trans)
        #CI矩阵,其中CIMat[i,j]表示第i个DU与第j个UU之间的CSI
        CIMat = CSI_CI_generate(self.UU.locatMat, self.DU.locatMat)
        UUCSIMat = torch.from_numpy(UUCSIMat).type(torch.complex64)
        UUPower = torch.from_numpy(self.UU.powerBudgetArr).type(torch.complex64)
        UUMat = torch.cat([UUCSIMat, UUPower.unsqueeze(0)], dim=0)#?
        DUMat = torch.from_numpy(DUMat).type(torch.complex64)
        CIMat = torch.from_numpy(CIMat).type(torch.complex64)
        INMat = torch.from_numpy(np.stack([self.IN.angleArr, self.IN.powerGainArr], axis=1)).type(torch.complex64)
        TAMat = torch.from_numpy(np.stack([self.TA.angleArr, self.TA.powerGainArr], axis=1)).type(torch.complex64)
        return (UUMat, DUMat, INMat, TAMat, CIMat)

uu_list, du_list, in_list, ta_list, ci_list = [], [], [], [], []
def generate_samples(num_samples: int, MAT):
    for i in range(num_samples):
        global uu_list, du_list, in_list, ta_list, ci_list
        UUMat, DUMat, INMat, TAMat, CIMat = MAT
        
        # 收集到各自的列表中
        uu_list.append(UUMat)
        du_list.append(DUMat)
        in_list.append(INMat)
        ta_list.append(TAMat)
        ci_list.append(CIMat)
    uu_list = torch.stack(uu_list, axis=0)
    du_list = torch.stack(du_list, axis=0)
    in_list = torch.stack(in_list, axis=0)
    ta_list = torch.stack(ta_list, axis=0)
    ci_list = torch.stack(ci_list, axis=0)
    print(uu_list.shape,du_list.shape,in_list.shape,ta_list.shape,ci_list.shape)
    return uu_list, du_list, in_list, ta_list, ci_list

def save_unsupervised_dataset(MAT,data_path="unsupervised_dataset", num_samples=1000, 
                             num_trans=4, num_rece=4, embed_dim=16):
    """保存无监督数据集到文件"""
    os.makedirs(data_path, exist_ok=True)
    
    UUMat, DUMat, INMat, TAMat, CIMat = MAT
    print(UUMat.shape,DUMat.shape,INMat.shape,TAMat.shape,CIMat.shape)
    dataset = TensorDataset(UUMat, DUMat, INMat, TAMat, CIMat)
    
    torch.save(dataset, os.path.join(data_path, "dataset.pt"))
    
    data_info = {
        'num_samples': num_samples,
        'num_trans': num_trans,
        'num_rece': num_rece,
        'embed_dim': embed_dim
    }
    torch.save(data_info, os.path.join(data_path, "data_info.pt"))
    
    print(f"无监督数据集已保存到 {data_path}")
    print(f"数据集大小: {len(dataset)}")
    
    return dataset

if __name__ == "__main__":
    #测试代码
    BS = BSparameter(4, 4, 100, np.array([0,0,20]))
    TA = TAparameter(5, np.array([0,180]), np.array([-30,-30]))
    IN = INparameter(3, np.array([0,180]), np.array([-30,-30]))
    UU = UUparameter(4, 10, np.array([[0,100],[0,100],[0,10]]))
    DU = DUparameter(4, np.array([[0,100],[0,100],[0,10]]))
    system = systemParameter(BS,TA,IN,UU,DU)
    save_unsupervised_dataset(generate_samples(1000, system.data_create()))