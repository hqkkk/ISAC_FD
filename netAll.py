import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import netInitial
import netUpdate
import netOutput

class netAll(nn.Module):
    def __init__(self, num_trans, num_rece, num_heads, embed_dim):
        super().__init__()
        self.Init=netInitial.InitailLayer(embed_dim, num_trans, num_rece)
        self.Update=netUpdate.UpdateLayer(embed_dim, num_heads)
        self.Update2=netUpdate.UpdateLayer(embed_dim, num_heads)
        self.Update3=netUpdate.UpdateLayer(embed_dim, num_heads)
        self.Output=netOutput.OutputLayer(embed_dim)
    def forward(self,UUMat,DUMat,INMat,TAMat,CIMat):
        UUMat1, DUMat1, INMat1, TAMat1= self.Init(UUMat,DUMat,INMat,TAMat)
        UUMat2, DUMat2, INMat2, TAMat2= self.Update(UUMat1,DUMat1,INMat1,TAMat1,CIMat)
        UUMat1, DUMat1, INMat1, TAMat1= self.Update2(UUMat2,DUMat2,INMat2,TAMat2,CIMat)
        UUMat2, DUMat2, INMat2, TAMat2= self.Update3(UUMat1,DUMat1,INMat1,TAMat1,CIMat)
        UUPower, DUComMat, INComMat, TAPower= self.Output(UUMat2,DUMat2,INMat2,TAMat2)
        return UUPower, DUComMat, INComMat, TAPower
    
