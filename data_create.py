import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


def dBm2watt(dbm):
    return 10 ** (dbm / 10)*1e-3
def dBW2watt(dbW):
    return 10 ** ((dbW+30) / 10)*1e-3
def dB2linear(dB):
    return 10 ** (dB / 10)
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
def powerGain_create(num, distanceRange,test = True):
    distanceArr = np.random.uniform(distanceRange[0], distanceRange[1], num)
    RandomMat = (np.random.randn(num)
                + 1j * np.random.randn(num))/np.sqrt(2)
    powerGain=(RandomMat * (np.sqrt(10**(-3)*(distanceArr**-2.67))
                            ))
    return powerGain

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
    def __init__(self, num, angleRange, powerGainRange,ta_distanRange, setAngle = False):
        if(setAngle):
            self.num = num
            self.angleArr = angleRange
            self.powerGainArr = powerGainRange
            self.setAngle = setAngle
            self.distanceArr = ta_distanRange
            self.powerGain = None
        else:
            self.num = num
            self.angleRange = angleRange
            self.powerGain = powerGainRange
            self.setAngle = setAngle
            self.angleArr = None
            self.distanceArr = ta_distanRange
            self.powerGainArr = None
    def info(self):
        print(self.__dict__)
    def random(self):
        '''
        随机生成TA参数
        '''
        if(self.setAngle):
            return self.angleArr, self.powerGainArr
        self.powerGainArr = powerGain_create(self.num, self.distanceArr)
        self.angleArr =np.array([np.random.uniform(low, high) for low, high in self.angleRange])#角度不要相距太近
        return self.angleArr, self.powerGainArr
        
class INparameter:
    '''
    干扰参数设置 num= real, angleRange = 1Darr(2), powerGainRange = 1Darr(2)
    返回值: angleArr = 1Darr(M), powerGainArr = 1Darr(M)
    '''
    def __init__(self, num, angleRange, powerGainArr, distanRange, setAngle = False):
        if(setAngle):
            self.num = num
            self.angleArr = angleRange
            self.setAngle = setAngle
            self.distanceArr = distanRange
        else:
            self.num = num
            self.angleRange = angleRange
            self.setAngle = setAngle
            self.angleArr = None
            self.distanceArr = distanRange
        self.powerGainArr = powerGainArr
    def info(self):
        print(self.__dict__)
    def random(self):
        '''
        随机生成IN参数
        '''
        self.powerGainArr = powerGain_create(self.num, self.distanceArr)
        if(self.setAngle):
            pass
        else:
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
        pb_np = np.array(self.UU.powerBudgetArr).reshape(-1, 1)
        pb_t = torch.from_numpy(pb_np).float().type(torch.complex64)
        UUMat = torch.cat([UUMat, pb_t], dim=1)
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
                            noise2BS = dBm2watt(-70), alpha_SI = dBm2watt(-110), 
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
        # 单位均为瓦 (W) / 线性值
        'noise2DU_W': noise2DU,
        'noise2BS_W': noise2BS,
        'alpha_SI_linear': alpha_SI,
        'power_BS_W': power_BS
    }
    torch.save(data_info, os.path.join(data_path, "data_info.pt"))
    
    num_samples_saved = UUMat.shape[0] if (hasattr(UUMat, 'shape') and len(UUMat.shape) > 0) else 1
    print(f"无监督数据集已保存到 {data_path}")
    print(f"数据集大小: {num_samples_saved}")
    
    return data

def printDataSamples(system):
    print("系统参数：")
    system.BS.info()
    print("TA参数：")
    system.TA.info()
    print("IN参数：")
    system.IN.info()
    print("UU参数：")
    system.UU.info()
    print("DU参数：")
    system.DU.info()
    print("环境参数：")
    print(f"噪声功率DU侧: {system.noise2DI},噪声功率BS侧: {system.noise2BS}, 自干扰衰减: {system.alpha_SI}")    
    UUMat, DUMat, INMat, TAMat, CIMat = system.data_create()
    print("生成的数据样本：")
    print(f"UUMat shape: {UUMat.shape}",UUMat)
    print(f"DUMat shape: {DUMat.shape}",DUMat)
    print(f"INMat shape: {INMat.shape}",INMat)
    print(f"TAMat shape: {TAMat.shape}",TAMat)
    print(f"CIMat shape: {CIMat.shape}",CIMat)


if __name__ == "__main__":
    #生成数据参数设定
    noise2DU_dBm = -70  # dBm
    noise2BS_dBm = -70  # dBm
    alpha_SI_dB = -110  # dB
    noise2DU = np.sqrt(dBm2watt(noise2DU_dBm))
    noise2BS = np.sqrt(dBm2watt(noise2BS_dBm))
    alpha_SI = dB2linear(alpha_SI_dB)
    num_trans = 4
    num_rece = 4
    power_BS = dBm2watt(18)  # 基站功率约0.063W
    BSlocation_XYZ = np.array([0,20,0])

    ta_num = 2
    ta_angleRange = np.array([[45,75],[105,135]])
    #功率增益的平方，增益随机生成即可
    ta_powerGainArr = np.full((ta_num),np.sqrt(dBm2watt(noise2BS_dBm-30)))
    ta_distanceRange = np.array([75,125])#此时数量级较为统一
    in_num = 2
    in_angleRange = np.array([45,135])
    in_powerGainArr = np.full((in_num),np.sqrt(dBm2watt(noise2BS_dBm+20)))
    in_distanceRange = np.array([75,125])

    uu_num = 4
    uu_powerBudget = dBm2watt(5)#约为0.003W
    uu_locatRange = np.array([[0,100],[0,10],[0,100]])
    
    du_num = 4
    du_locatRange = np.array([[0,100],[0,10],[0,100]])

    BS = BSparameter(num_trans, num_rece, power_BS, BSlocation_XYZ)
    TA = TAparameter(ta_num, ta_angleRange, ta_powerGainArr, ta_distanceRange, setAngle=False)
    IN = INparameter(in_num, in_angleRange, in_powerGainArr, in_distanceRange, setAngle=False)
    UU = UUparameter(uu_num, uu_powerBudget, uu_locatRange)
    DU = DUparameter(du_num, du_locatRange)
    system = systemParameter(BS,TA,IN,UU,DU)
    system.envParaSet(np.sqrt(dBm2watt(noise2DU_dBm)),np.sqrt(dBm2watt(noise2BS_dBm)),alpha_SI,num_trans,num_rece)
    # 生成样本并保存，注意：将实际使用的参数传递给保存函数，避免默认参数覆盖
    num_samples_to_generate = 10000
    samples = generate_samples(num_samples_to_generate, system.data_create())
    save_unsupervised_dataset(
        samples,
        data_path="dataset",
        num_samples=num_samples_to_generate,
        num_trans=num_trans,
        num_rece=num_rece,
        noise2DU=noise2DU,
        noise2BS=noise2BS,
        alpha_SI=alpha_SI,
        power_BS=power_BS,
    )
    printDataSamples(system)
