import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

def dB2linear(dB):
    return 10 ** (dB / 10)

def dBm2watt(dbm):
    return 10 ** (dbm / 10)*1e-3

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
    def __init__(self, num, angleRange, powerGainArr):
        self.num = num
        self.angleRange = angleRange
        self.powerGainArr = powerGainArr
    def info(self):
        print(self.__dict__)
    def random(self):
        '''
        随机生成IN参数
        '''
        self.angleArr = np.random.uniform(self.angleRange, self.num)
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
    def envParaSet(self, noise2DI, noise2BS,alpha_SI,num_trans,num_rece):
        self.noise2DI = noise2DI
        self.noise2BS = noise2BS
        self.alpha_SI = alpha_SI
        self.num_trans = num_trans
        self.num_rece = num_rece
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

        # UUCSIMat: shape (N_r, UUnum) from CSI_generate; convert to tensor and transpose
        UUCSIMat = torch.from_numpy(UUCSIMat).type(torch.complex64).permute(1, 0).contiguous()
        # set UUMat as (UUnum, N_r)
        UUMat = UUCSIMat

        # DUMat: CSI_generate returned shape (N_t, DUnum); convert to (DUnum, N_t)
        DUMat = torch.from_numpy(DUMat).type(torch.complex64).permute(1, 0).contiguous()

        # CIMat: shape (UUnum, DUnum) is already correct
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

def save_unsupervised_dataset(MAT,data_path="dataset", num_samples=1000, 
                             num_trans=4, num_rece=4, noise2DU = dBm2watt(-70),
                            noise2BS = dBm2watt(-70), alpha_SI = dB2linear(-110), 
                            power_BS = dBm2watt(18),
                            ):
    '''
    保存无监督数据集到文件
    '''
    os.makedirs(data_path, exist_ok=True)
    UUMat, DUMat, INMat, TAMat, CIMat = MAT
    print(UUMat.shape,DUMat.shape,INMat.shape,TAMat.shape,CIMat.shape)
    # 保存为纯张量字典（避免保存 Dataset/TensorDataset 对象，便于跨环境加载）
    data = {
        'UUMat': UUMat,
        'DUMat': DUMat,
        'INMat': INMat,
        'TAMat': TAMat,
        'CIMat': CIMat,
        'noise2DU': noise2DU,
        'noise2BS': noise2BS,
        'alpha_SI': alpha_SI,
    }
    torch.save(data, os.path.join(data_path, "dataset.pt"))
    
    data_info = {
        'num_samples': num_samples,
        'num_trans': num_trans,
        'num_rece': num_rece,
        'noise2DU': noise2DU,
        'noise2BS_dBm': noise2BS,
        'alpha_SI_dB': alpha_SI,
        'power_BS': power_BS
    }
    torch.save(data_info, os.path.join(data_path, "data_info.pt"))
    
    num_samples_saved = UUMat.shape[0] if (hasattr(UUMat, 'shape') and len(UUMat.shape) > 0) else 1
    print(f"无监督数据集已保存到 {data_path}")
    print(f"数据集大小: {num_samples_saved}")
    
    return data



if __name__ == "__main__":
    #生成数据参数设定
    noise2DU_dBm = -70  # dBm
    noise2BS_dBm = -70  # dBm
    alpha_SI_dB = -110  # dB
    num_trans = 4
    num_rece = 4
    power_BS = dBm2watt(18)  # 基站功率预算
    BSlocation_XYZ = np.array([0,20,0])

    ta_num = 2
    ta_angleRange = np.array([45,135])
    #功率增益的平方
    ta_powerGainArr = np.full((ta_num),dB2linear(-30)*dB2linear(noise2BS_dBm))
    
    in_num = 2
    in_angleRange = np.array([45,135])
    in_powerGainArr = np.full((in_num),dB2linear(20)*dB2linear(noise2BS_dBm))
    
    uu_num = 4
    uu_powerBudget = dBm2watt(5)
    uu_locatRange = np.array([[0,100],[0,10],[0,100]])
    
    du_num = 4
    du_locatRange = np.array([[0,100],[0,10],[0,100]])

    BS = BSparameter(num_trans, num_rece, power_BS, BSlocation_XYZ)
    TA = TAparameter(ta_num, ta_angleRange, ta_powerGainArr)
    IN = INparameter(in_num, in_angleRange, in_powerGainArr)
    UU = UUparameter(uu_num, uu_powerBudget, uu_locatRange)
    DU = DUparameter(du_num, du_locatRange)
    system = systemParameter(BS,TA,IN,UU,DU)
    system.envParaSet(dBm2watt(noise2DU_dBm),dBm2watt(noise2BS_dBm),dB2linear(alpha_SI_dB),num_trans,num_rece)
    save_unsupervised_dataset(generate_samples(10000, system.data_create()))