# combine the registration with the pinns
from non_rigid_registration_modified_v10 import MLPNet,options,prostateset,PointNet_features,chamfer_loss,save_checkpoint, MLPNet_wo_relu,get_sample,convert_list_into_array
from pinns_modified_v10 import pinn_loss, PINNsNet, pinns_layers,pinn_loss_lame_vector,get_lame_vectors_torch
import torch
import numpy as np
import argparse
import torch.utils.data
import logging
import os
import time

def check_derivative(der, t):
    if der is None:
       return  torch.zeros_like(t, requires_grad=True)
    else:
       der.requires_grad_()
       return der 

# check if we have imported properly
'''
scale  = 1
mlp_h1 = [int(64/scale), int(64/scale)]
h1 = MLPNet(3, mlp_h1, b_shared=True).layers
print(h1)
'''

class PhysicsInformedRegistration(torch.nn.Module):
    '''
    Combine the pinns and registration;
    Please refer to the 'class deform' in non_rigid_registration.py;
    and 'class PhysicsInformed' in pinns.py. 
    '''
    def __init__(self, nch_input=64, num_neurons =[40]*4 ,  dim_k=1024): 
        # ux, uy, uz are the known displacements (of the boundary points)
        super().__init__()
        '''
        the deformation related
        '''
        self.ptfeatures = PointNet_features()
        # the MLP layers
        mlp_list_layers2= [int(1024), int(512), int(256), int(128), int(64)]
        self.list_layers2 = MLPNet(2*dim_k+3, mlp_list_layers2, b_shared=True).layers
        # the last layer
        last_layer      = [int(3)]
        self.last_layer = MLPNet_wo_relu(64, last_layer, b_shared=True).layers   
        '''
        the pinns related
        '''  
        
        self.ptfeatures_pinns = PointNet_features()

        self.pinns_sxx1       = MLPNet(2*dim_k+3, mlp_list_layers2, b_shared=True).layers
        self.pinns_sxx2       = MLPNet_wo_relu(64, [int(1)], b_shared=True).layers  
        
        self.pinns_syy1       = MLPNet(2*dim_k+3, mlp_list_layers2, b_shared=True).layers
        self.pinns_syy2       = MLPNet_wo_relu(64, [int(1)], b_shared=True).layers  
        
        self.pinns_szz1       = MLPNet(2*dim_k+3, mlp_list_layers2, b_shared=True).layers
        self.pinns_szz2       = MLPNet_wo_relu(64, [int(1)], b_shared=True).layers         
        
        self.pinns_sxy1       = MLPNet(2*dim_k+3, mlp_list_layers2, b_shared=True).layers
        self.pinns_sxy2       = MLPNet_wo_relu(64, [int(1)], b_shared=True).layers            
                      
        self.pinns_sxz1       = MLPNet(2*dim_k+3, mlp_list_layers2, b_shared=True).layers
        self.pinns_sxz2       = MLPNet_wo_relu(64, [int(1)], b_shared=True).layers   
                       
        self.pinns_syz1       = MLPNet(2*dim_k+3, mlp_list_layers2, b_shared=True).layers  
        self.pinns_syz2       = MLPNet_wo_relu(64, [int(1)], b_shared=True).layers 
        
         
    def forward(self, data):
        
        source = data[:,0,:,:]
        target = data[:,1,:,:]       

        # get the x,y,z coordinates
        source_x_coordinates = source[:,:,0]                                 #batch*number_of_points_in_each_batch
        source_y_coordinates = source[:,:,1]
        source_z_coordinates = source[:,:,2]
        x = torch.reshape(source_x_coordinates, (-1,1))                      #reshape it into a column vector
        y = torch.reshape(source_y_coordinates, (-1,1))
        z = torch.reshape(source_z_coordinates, (-1,1))
        
#        print('shape of x', x.shape) #[2048, 1]
#        print('shape of y', y.shape) #[2048, 1]       
#        print('shape of z', z.shape) #[2048, 1]
        
        # set the x,y,z to be 
        x.requires_grad= True
        y.requires_grad= True       
        z.requires_grad= True    
        
        xyz = torch.cat((x,y,z), 1)
#        xyz.requires_grad= True
#        print('shape of xyz', xyz.shape)

        # if we use one network, we have to utilise the source data re-generated from the x,y,z..
        batch_number   = data.shape[0]
        num_points     = source.shape[1]
        source_used_x  = torch.reshape(x, (batch_number,-1))
        source_used_y  = torch.reshape(y, (batch_number,-1))
        source_used_z  = torch.reshape(z, (batch_number,-1))
#        print('the device of source_used_x', source_used_x.get_device())
#        print('the device of source_used_y', source_used_y.get_device())
#        print('the device of source_used_z', source_used_z.get_device())
        source_used    = torch.zeros(batch_number,num_points,3, device=torch.device('cuda:0')) # Make Sure that THIS TENSOR is GPU
#        print('the device of source_used', source_used.get_device())
        source_used[:,:,0] = source_used_x 
        source_used[:,:,1] = source_used_y
        source_used[:,:,2] = source_used_z
                         
#        print('shape of source_used_x', source_used_x.shape)
#        print('shape of source_used_y', source_used_y.shape)
#        print('shape of source_used_z', source_used_z.shape)
#        print('shape of source_used',   source_used.shape) # [2,1024,3])
#        print('shape of source', source.shape)
        
        pffeat_src   = self.ptfeatures(source_used)       # source_used or source
#        print('pffeat_src shape', pffeat_src.shape)        # [2,1024])
        pffeat_target= self.ptfeatures(target)       
        
        global_feature = torch.cat( (pffeat_src,pffeat_target), -1)        
        num_source     = source.shape[1]       
        global_feature_repeated                 = global_feature.unsqueeze(1).repeat(1, num_source, 1)       
        global_feature_repeated_conca           = torch.cat((global_feature_repeated, source) , -1)      
#        print('shape of global_feature_repeated_conca.permute(0,2,1)', global_feature_repeated_conca.permute(0,2,1).shape)  #[2, 2051, 1024]
        displacements_source_before_last_layer  = self.list_layers2(global_feature_repeated_conca.permute(0,2,1))           #[2, 64,   1024]
#        print('shape of displacements_source_before_last_layer', displacements_source_before_last_layer.shape)     
        displacements_source         = self.last_layer(displacements_source_before_last_layer)        # (batch, 3, num_points_source), i.e., [2,3,1024]     
#        print('shape of displacements_source', displacements_source.shape)                            
        displacements_source_reshape = displacements_source.permute(0,2,1)   # (batch, num_points_source, 3), which can be directly added to source
#        print('shape of displacements_source_reshape', displacements_source_reshape.shape)            # [2, 1024,3]       
        deformed_source              = source + displacements_source_reshape #  (batch, num_points_source, 3)
        
        # the physics-informed network
        disp_x_coordinates = displacements_source_reshape[:,:,0]             #batch*number_of_points_in_each_batch
        disp_y_coordinates = displacements_source_reshape[:,:,1]
        disp_z_coordinates = displacements_source_reshape[:,:,2]
        disp_x = torch.reshape(disp_x_coordinates,  (-1,1))                  #reshape it into a column vector
        disp_y = torch.reshape(disp_y_coordinates,  (-1,1))
        disp_z = torch.reshape(disp_z_coordinates,  (-1,1))                                           
        ux_target =disp_x
        uy_target =disp_y
        uz_target =disp_z
        
        
        # in this case, we utilise one sole network to predict the displacement vectors.
        Ux  = ux_target
        Uy  = uy_target
        Uz  = uz_target
        
#        print('shape of Ux', Ux.shape) #[2048,1]
#        print('shape of Uy', Uy.shape) #[2048,1]
#        print('shape of Uz', Uz.shape) #[2048,1]
                        
        '''
        the pinns related
        '''  
        pffeat_src_pinns     = self.ptfeatures_pinns(source_used)       
        pffeat_target_pinns  = self.ptfeatures_pinns(target)       
        
        global_feature_pinns = torch.cat( (pffeat_src_pinns,pffeat_target_pinns), -1)        
        num_source     = source.shape[1]       
        global_feature_repeated_pinns                 = global_feature_pinns.unsqueeze(1).repeat(1, num_source, 1)       
        global_feature_repeated_conca_pinns           = torch.cat((global_feature_repeated_pinns, source) , -1) 

        Sxx  =self.pinns_sxx2( self.pinns_sxx1(global_feature_repeated_conca_pinns.permute(0,2,1)))         
        Syy  =self.pinns_syy2( self.pinns_syy1(global_feature_repeated_conca_pinns.permute(0,2,1)))    
        Szz  =self.pinns_szz2( self.pinns_szz1(global_feature_repeated_conca_pinns.permute(0,2,1)))    
        Sxy  =self.pinns_sxy2( self.pinns_sxy1(global_feature_repeated_conca_pinns.permute(0,2,1)))    
        Sxz  =self.pinns_sxz2( self.pinns_sxz1(global_feature_repeated_conca_pinns.permute(0,2,1)))    
        Syz  =self.pinns_syz2( self.pinns_syz1(global_feature_repeated_conca_pinns.permute(0,2,1)))    
        
        
#        print('shape of Sxx', Sxx.shape) #[2, 1, 1024]
#        print('shape of Syy', Syy.shape) #[2, 1, 1024]   
#        print('shape of Szz', Szz.shape) #[2, 1, 1024]       
#        print('shape of Sxy', Sxy.shape) #[2, 1, 1024]
#        print('shape of Sxz', Sxz.shape) #[2, 1, 1024]
#        print('shape of Syz', Syz.shape) #[2, 1, 1024]              

        Sxx = torch.reshape(Sxx,  (-1,1))
        Syy = torch.reshape(Syy,  (-1,1))
        Szz = torch.reshape(Szz,  (-1,1))
        Sxy = torch.reshape(Sxy,  (-1,1))
        Sxz = torch.reshape(Sxz,  (-1,1))
        Syz = torch.reshape(Syz,  (-1,1))
        
#        print('shape of Sxx', Sxx.shape) #[2048, 1]
#        print('shape of Syy', Syy.shape) #[2048, 1]
#        print('shape of Szz', Szz.shape) #[2048, 1]
#        print('shape of Sxy', Sxy.shape) #[2048, 1]
#        print('shape of Sxz', Sxz.shape) #[2048, 1]
#        print('shape of Syz', Syz.shape) #[2048, 1]         
        
                     
#        Exx = Ux
#        Eyy = Ux
#        Ezz = Ux
#        Exy = Ux
#        Exz = Ux
#        Eyz = Ux
#        momentum_balance1 = Ux
#        momentum_balance2 = Ux
#        momentum_balance3 = Ux
        
        grad_outputs=torch.ones_like(Ux) 
        print('which machine does grad_outputs belong to?', grad_outputs.get_device())
        
        
        Exx_der = torch.autograd.grad(Ux, x, grad_outputs=torch.ones_like(Ux), create_graph=True, allow_unused=True)[0]       #Ux, x
        Exx = check_derivative(Exx_der, x)
        Eyy_der = torch.autograd.grad(Uy, y, grad_outputs=torch.ones_like(Uy), create_graph=True, allow_unused=True)[0]       #Uy, y
        Eyy = check_derivative(Eyy_der, y)
        Ezz_der = torch.autograd.grad(Uz, z, grad_outputs=torch.ones_like(Uz), create_graph=True, allow_unused=True)[0]       #Uz, z
        Ezz = check_derivative(Ezz_der, z)

        Exy_der1 = torch.autograd.grad(Ux,y, grad_outputs=torch.ones_like(Ux), create_graph=True, allow_unused=True)[0] 
        Exy_der2 = torch.autograd.grad(Uy,x, grad_outputs=torch.ones_like(Uy), create_graph=True, allow_unused=True)[0] 
        Exy      = 0.5*(check_derivative(Exy_der1,y) + check_derivative(Exy_der2,x))
                
        Exz_der1 = torch.autograd.grad(Ux,z, grad_outputs=torch.ones_like(Ux), create_graph=True, allow_unused=True)[0] 
        Exz_der2 = torch.autograd.grad(Uz,x, grad_outputs=torch.ones_like(Uz), create_graph=True, allow_unused=True)[0] 
        Exz      = 0.5*(check_derivative(Exz_der1,z)  + check_derivative(Exz_der2,x))
        
        
        Eyz_der1 = torch.autograd.grad(Uy,z, grad_outputs=torch.ones_like(Uy), create_graph=True, allow_unused=True)[0]
        Eyz_der2 = torch.autograd.grad(Uz,y, grad_outputs=torch.ones_like(Uz), create_graph=True, allow_unused=True)[0]   
        Eyz      = 0.5*(check_derivative(Eyz_der1,z) +check_derivative(Eyz_der2,y))
        

        momentum_balance1_der1 = torch.autograd.grad(Sxx, x, grad_outputs=torch.ones_like(Sxx), create_graph=True, allow_unused=True)[0] 
        momentum_balance1_der2 = torch.autograd.grad(Sxy, y, grad_outputs=torch.ones_like(Sxy), create_graph=True, allow_unused=True)[0] 
        momentum_balance1_der3 = torch.autograd.grad(Sxz, z, grad_outputs=torch.ones_like(Sxz), create_graph=True, allow_unused=True)[0] 
        momentum_balance1 = check_derivative(momentum_balance1_der1,x) + check_derivative(momentum_balance1_der2, y) + check_derivative(momentum_balance1_der3,z)
       
             
        momentum_balance2_der1 = torch.autograd.grad(Sxy, x, grad_outputs=torch.ones_like(Sxy), create_graph=True, allow_unused=True)[0] 
        momentum_balance2_der2 = torch.autograd.grad(Syy, y, grad_outputs=torch.ones_like(Syy), create_graph=True, allow_unused=True)[0] 
        momentum_balance2_der3 = torch.autograd.grad(Syz, z, grad_outputs=torch.ones_like(Syz), create_graph=True, allow_unused=True)[0]
        momentum_balance2      = check_derivative(momentum_balance2_der1, x) + check_derivative(momentum_balance2_der2, y) + check_derivative(momentum_balance2_der3,z)
        
        
        momentum_balance3_der1 = torch.autograd.grad(Sxz, x, grad_outputs=torch.ones_like(Sxz), create_graph=True, allow_unused=True)[0] 
        momentum_balance3_der2 = torch.autograd.grad(Syz, y, grad_outputs=torch.ones_like(Syz), create_graph=True, allow_unused=True)[0] 
        momentum_balance3_der3 = torch.autograd.grad(Szz, z, grad_outputs=torch.ones_like(Szz), create_graph=True, allow_unused=True)[0]
        momentum_balance3      = check_derivative(momentum_balance3_der1, x) + check_derivative(momentum_balance3_der2,y) + check_derivative(momentum_balance3_der3,z)

        
#        print('shape of Exx', Exx.shape)    #[2048, 1]    
#        print('shape of Eyy', Eyy.shape)    #[2048, 1]      
#        print('shape of Ezz', Ezz.shape)    #[2048, 1]      

        
#        print('shape of Exy', Exy.shape)    #[2048, 1]    
#        print('shape of Exz', Exz.shape)    #[2048, 1]      
#        print('shape of Eyz', Eyz.shape)    #[2048, 1]        
                  
#        print('shape of momentum_balance1', momentum_balance1.shape)      #[2048, 1]          
#        print('shape of momentum_balance2', momentum_balance2.shape)      #[2048, 1]
#        print('shape of momentum_balance3', momentum_balance3.shape)      #[2048, 1]
        
        return deformed_source, Ux, Uy, Uz, Sxx, Syy, Szz, Sxy, Sxz, Syz, Exx, Eyy, Ezz, Exy, Exz, Eyz, momentum_balance1, momentum_balance2, momentum_balance3 # ux_target, uy_target, uz_target are not outputted if one network is utilised.

def train_with_pinns(model, trainloader, optimizer, device):
    model.train()
    vloss         = 0.0
    # would like to check individual loss values, i.e., chamfer and pinns loss values
    vloss_chamfer = 0.0
    vloss_pinns   = 0.0
     
     
    # -----further split the individual loss terms in the pinns loss---
    vloss_governing_equation1 = 0.0
    vloss_governing_equation2 = 0.0   
    vloss_governing_equation3 = 0.0    
    vloss_governing_equation4 = 0.0
    vloss_governing_equation5 = 0.0
    vloss_governing_equation6 = 0.0
    vloss_elastic_energy      = 0.0
    vloss_momentum_balance    = 0.0   

     
    count = 0
    for i, data in enumerate(trainloader):
        
#        print(type(data))
        data = data.float()
#        print(i)
#        print('the data used in training', data.shape)
        optimizer.zero_grad()
        
#        t = time.time()                
#        print('the device of data', data.get_device()) #RuntimeError: get_device is not implemented for type torch.FloatTensor
        # convert the data and model to gpu
#        print('the device of data', device)
        
        data          =  data.to(device)  
        model.to(device)
#        model.to(torch.device('cuda:0'))

        deformed, Ux, Uy, Uz, Sxx, Syy, Szz, Sxy, Sxz, Syz, Exx, Eyy, Ezz, Exy, Exz, Eyz, momentum_balance1, momentum_balance2, momentum_balance3   =  model(data)
#        elapsed = time.time() - t
#        print('elapsed time for conducting the model', elapsed)
        
        deformed      =  deformed.to(device)
#        ux_target     =  ux_target.to(device)
#        uy_target     =  uy_target.to(device)
#        uz_target     =  uz_target.to(device)
        Ux            =  Ux.to(device)
        Uy            =  Uy.to(device)
        Uz            =  Uz.to(device)
        Sxx           =  Sxx.to(device)
        Syy           =  Syy.to(device)
        Szz           =  Szz.to(device)
        Sxy           =  Sxy.to(device)
        Sxz           =  Sxz.to(device)
        Syz           =  Syz.to(device)
        Exx           =  Exx.to(device)
        Eyy           =  Eyy.to(device)
        Ezz           =  Ezz.to(device)
        Exy           =  Exy.to(device) 
        Exz           =  Exz.to(device)
        Eyz           =  Eyz.to(device)
        momentum_balance1 =momentum_balance1.to(device)
        momentum_balance2 =momentum_balance2.to(device)        
        momentum_balance3 =momentum_balance3.to(device)         
        
        data          =  data.to(device)
#        print('the device of data', data.get_device()) 
                
                
        #loss_cham     =  chamfer_loss(deformed, data[:,1,:,:], data.shape[2]) # use all points (i.e., both the boundary and internal points)
#        print('data.shape[2]/2', data.shape[2]/2)
#        print('type of data.shape[2]/2', type(data.shape[2]/2))
        num_points_used_in_loss_cham = int(data.shape[2]/2)
        
#        t = time.time()
        loss_cham     =  chamfer_loss(deformed[:,0:num_points_used_in_loss_cham,:],    data[:,1,0:num_points_used_in_loss_cham,:],    num_points_used_in_loss_cham)     # deformed is of (batch, num_points_source, 3)
#        elapsed = time.time() - t
#        print('elapsed time for computing chamfer loss', elapsed)
        
#        t = time.time()
        # the pinns loss using the same values of lames at different points
#        PINNs_Loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance=pinn_loss(Ux, Uy, Uz,  Sxx, Syy, Szz, Sxy, Sxz, Syz, Exx, Eyy, Ezz, Exy, Exz, Eyz, momentum_balance1, momentum_balance2, momentum_balance3)
        
        
        # get the lames vectors for both source point sets
        patient_1_source = data[0,0,:,:]
#        patient_2_source = data[1,0,:,:]
        lame1_vector_patient_1, lame2_vector_patient_1 =get_lame_vectors_torch(patient_1_source, lame1_version1=torch.tensor(82.21), lame2_version1=torch.tensor(1.68), lame1_version2=torch.tensor(8221.47), lame2_version2=torch.tensor(167.78))           
#        lame1_vector_patient_2, lame2_vector_patient_2 =get_lame_vectors_torch(patient_2_source, lame1_version1=torch.tensor(82.21), lame2_version1=torch.tensor(1.68), lame1_version2=torch.tensor(197.32), lame2_version2=torch.tensor(4.03))           
        '''
        In the case where the data are the same in one batch, we do not 
        need to compute the values of lame1(or lame2) twice. 
        We can directly copy the lame1 and lame2 computed for 
        '''
        lame1_vector_patient_2 = lame1_vector_patient_1
        lame2_vector_patient_2 = lame2_vector_patient_1
#        print('shape of lame1_vector_patient_1', lame1_vector_patient_1.size())
#        print('shape of lame2_vector_patient_1', lame2_vector_patient_1.size())
#        print('shape of lame1_vector_patient_2', lame1_vector_patient_2.size())
#        print('shape of lame2_vector_patient_2', lame2_vector_patient_2.size())
        
        lame1_vector = torch.cat( (lame1_vector_patient_1,lame1_vector_patient_2), 0)
        lame2_vector = torch.cat( (lame2_vector_patient_1,lame2_vector_patient_2), 0)
#        print('shape of lame1_vector', lame1_vector.size())
#        print('shape of lame2_vector', lame2_vector.size()) 
        
                
        # the pinns loss considering different values of lames at different points
        PINNs_Loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance=pinn_loss_lame_vector(Ux, Uy, Uz, Sxx,Syy,Szz,Sxy,Sxz,Syz,Exx,Eyy,Ezz,Exy,Exz,Eyz, momentum_balance1,momentum_balance2, momentum_balance3,lame1_vector, lame2_vector)        
        
        # do stuff
#        elapsed = time.time() - t
#        print('elapsed time for computing pinn_loss', elapsed)
#        print('training loss_cham',  loss_cham)
#        print('training PINNs_Loss', PINNs_Loss)
        total_loss = 100000*loss_cham + PINNs_Loss
#        print('total_loss',          total_loss)
        # forward + backward + optimize
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()    
        
        vloss += total_loss
        # would like to check individual loss values, i.e., chamfer and pinns loss values
        vloss_chamfer+= loss_cham
        vloss_pinns  +=PINNs_Loss
        
        # would like to further check the individual loss terms in the pinns loss       
        vloss_governing_equation1 += governing_equation1
        vloss_governing_equation2 += governing_equation2  
        vloss_governing_equation3 += governing_equation3    
        vloss_governing_equation4 += governing_equation4
        vloss_governing_equation5 += governing_equation5
        vloss_governing_equation6 += governing_equation6
        vloss_elastic_energy      += elastic_energy
        vloss_momentum_balance    += momentum_balance 
        
        count += 1
        
    ave_vloss         = float(vloss)/count
    ave_vloss_chamfer = float(vloss_chamfer)/count
    ave_vloss_pinns   = float(vloss_pinns)/count
    
    ave_vloss_governing_equation1 = float(vloss_governing_equation1)/count
    ave_vloss_governing_equation2 = float(vloss_governing_equation2)/count 
    ave_vloss_governing_equation3 = float(vloss_governing_equation3)/count 
    ave_vloss_governing_equation4 = float(vloss_governing_equation4)/count
    ave_vloss_governing_equation5 = float(vloss_governing_equation5)/count
    ave_vloss_governing_equation6 = float(vloss_governing_equation6)/count
    ave_vloss_elastic_energy      = float(vloss_elastic_energy)/count     
    ave_vloss_momentum_balance    = float(vloss_momentum_balance)/count    
       
    return ave_vloss, ave_vloss_chamfer, ave_vloss_pinns, ave_vloss_governing_equation1, ave_vloss_governing_equation2, ave_vloss_governing_equation3, ave_vloss_governing_equation4, ave_vloss_governing_equation5, ave_vloss_governing_equation6,ave_vloss_elastic_energy, ave_vloss_momentum_balance

def eval_with_pinns(model, testloader, device):
    model.eval()
    vloss = 0.0
    # would like to check individual loss values, i.e., chamfer and pinns loss values
    vloss_chamfer = 0.0
    vloss_pinns   = 0.0
    
    # -----further split the individual loss terms in the pinns loss---
    vloss_governing_equation1 = 0.0
    vloss_governing_equation2 = 0.0   
    vloss_governing_equation3 = 0.0    
    vloss_governing_equation4 = 0.0
    vloss_governing_equation5 = 0.0
    vloss_governing_equation6 = 0.0
    vloss_elastic_energy      = 0.0
    vloss_momentum_balance    = 0.0     
      
    count = 0

   # with torch.no_grad():
    for i, data in enumerate(testloader):
       data = data.float()
       
       # convert the data and model to the gpu device
       data =  data.to(device)  
       model.to(device)
       
       
       # get the lame vectors for both source point sets
       patient_1_source = data[0,0,:,:]
#       patient_2_source = data[1,0,:,:]       
       
       lame1_vector_patient_1, lame2_vector_patient_1 =get_lame_vectors_torch(patient_1_source, lame1_version1=torch.tensor(82.21), lame2_version1=torch.tensor(1.68), lame1_version2=torch.tensor(8221.47), lame2_version2=torch.tensor(167.78))           
#       lame1_vector_patient_2, lame2_vector_patient_2 =get_lame_vectors_torch(patient_2_source, lame1_version1=torch.tensor(82.21), lame2_version1=torch.tensor(1.68), lame1_version2=torch.tensor(197.32), lame2_version2=torch.tensor(4.03))        
      
       '''
       In the case where the data are the same in one batch, we do not 
       need to compute the values of lame1(or lame2) twice. 
       We can directly copy the lame1 and lame2 computed for 
       '''
       lame1_vector_patient_2 = lame1_vector_patient_1
       lame2_vector_patient_2 = lame2_vector_patient_1
       
       lame1_vector = torch.cat( (lame1_vector_patient_1,lame1_vector_patient_2), 0)
       lame2_vector = torch.cat( (lame2_vector_patient_1,lame2_vector_patient_2), 0)       
       
       
       deformed, Ux, Uy, Uz, Sxx, Syy, Szz, Sxy, Sxz, Syz, Exx, Eyy, Ezz, Exy, Exz, Eyz, momentum_balance1, momentum_balance2, momentum_balance3  =  model(data) # ux_target,uy_target,uz_target, 
       deformed      =  deformed.to(device)
    #             ux_target     =  ux_target.to(device)
    #             uy_target     =  uy_target.to(device)
    #             uz_target     =  uz_target.to(device)
       
       Ux            =  Ux.to(device)
       Uy            =  Uy.to(device)
       Uz            =  Uz.to(device)
       Sxx           =  Sxx.to(device)
       Syy           =  Syy.to(device)
       Szz           =  Szz.to(device)
       Sxy           =  Sxy.to(device)
       Sxz           =  Sxz.to(device)
       Syz           =  Syz.to(device)
       Exx           =  Exx.to(device)
       Eyy           =  Eyy.to(device)
       Ezz           =  Ezz.to(device)
       Exy           =  Exy.to(device)
       Exz           =  Exz.to(device)
       Eyz           =  Eyz.to(device)
       momentum_balance1 = momentum_balance1.to(device)
       momentum_balance2 = momentum_balance2.to(device)        
       momentum_balance3 = momentum_balance3.to(device) 
    
       data          =  data.to(device)             
       
       #loss_cham     =  chamfer_loss(deformed, data[:,1,:,:], data.shape[2])   # use all points (i.e., both the boundary and internal points)
       num_points_used_in_loss_cham = int(data.shape[2]/2)
       loss_cham     =  chamfer_loss(deformed[:,0:num_points_used_in_loss_cham,:],    data[:,1,0:num_points_used_in_loss_cham,:],    num_points_used_in_loss_cham)     # deformed is of (batch, num_points_source, 3) 
#       PINNs_Loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance      =  pinn_loss(Ux, Uy, Uz,  Sxx, Syy, Szz, Sxy, Sxz, Syz, Exx, Eyy, Ezz, Exy, Exz, Eyz, momentum_balance1, momentum_balance2, momentum_balance3)   # ux_target, uy_target, uz_target,   
       
       # the pinns loss considering different values of lames at different points
       PINNs_Loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance      =pinn_loss_lame_vector(Ux, Uy, Uz, Sxx,Syy,Szz,Sxy,Sxz,Syz,Exx,Eyy,Ezz,Exy,Exz,Eyz, momentum_balance1,momentum_balance2, momentum_balance3,lame1_vector, lame2_vector)
       
       
#       print('test loss_cham',  loss_cham) 
#       print('test PINNs_Loss', PINNs_Loss)
       total_loss = loss_cham + PINNs_Loss
#       print('test total_loss', total_loss)   
       vloss += total_loss
       # would like to check individual loss values, i.e., chamfer and pinns loss values
       vloss_chamfer+= loss_cham
       vloss_pinns  +=PINNs_Loss
       
       # would like to further check the individual loss terms in the pinns loss       
       vloss_governing_equation1 += governing_equation1
       vloss_governing_equation2 += governing_equation2  
       vloss_governing_equation3 += governing_equation3    
       vloss_governing_equation4 += governing_equation4
       vloss_governing_equation5 += governing_equation5
       vloss_governing_equation6 += governing_equation6
       vloss_elastic_energy      += elastic_energy
       vloss_momentum_balance    += momentum_balance       
              
       count += 1    
    
    ave_vloss         = float(vloss)/count
    ave_vloss_chamfer = float(vloss_chamfer)/count
    ave_vloss_pinns   = float(vloss_pinns)/count 
    
    
    ave_vloss_governing_equation1 = float(vloss_governing_equation1)/count
    ave_vloss_governing_equation2 = float(vloss_governing_equation2)/count 
    ave_vloss_governing_equation3 = float(vloss_governing_equation3)/count 
    ave_vloss_governing_equation4 = float(vloss_governing_equation4)/count
    ave_vloss_governing_equation5 = float(vloss_governing_equation5)/count
    ave_vloss_governing_equation6 = float(vloss_governing_equation6)/count
    ave_vloss_elastic_energy      = float(vloss_elastic_energy)/count     
    ave_vloss_momentum_balance    = float(vloss_momentum_balance)/count  
   
    return ave_vloss, ave_vloss_chamfer, ave_vloss_pinns, ave_vloss_governing_equation1, ave_vloss_governing_equation2, ave_vloss_governing_equation3, ave_vloss_governing_equation4, ave_vloss_governing_equation5, ave_vloss_governing_equation6,ave_vloss_elastic_energy, ave_vloss_momentum_balance


class prostateset_v2(torch.utils.data.Dataset):
    '''
    This class implements that the data is read 
    
    '''
    def __init__(self, ALL_SAMPLES, transform = None, downsampleornot =False ):

        self.samples    = ALL_SAMPLES
        self.transform  = transform 
        self.downsampleornot = downsampleornot # whether we downsample the data again 
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        sample = self.samples[index]
        if self.transform is not None:
           sample= self.transform(sample)              
        sample_return =[]
        sample_return.append(sample[0])
        sample_return.append(sample[1])  
        if self.downsampleornot:
          sample_return_downsampled = downsample_mri_us(sample_return)
        else:                             
          sample_return_downsampled = convert_list_into_array(sample_return)
        return sample_return_downsampled



def main(args): 
    loader = np.load
    
    # read the data before the dataset class is used
    samples_path = get_sample(args.traindata_path)
    print('samples_path',           samples_path)
    print('type of samples_path',   type(samples_path))
    print('length of samples_path', len(samples_path))
    
    ALL_SAMPLES = []
    for index in range(0, len(samples_path)):
        current_sample_path = samples_path[index] 
        current_sample      = np.load(current_sample_path, allow_pickle=True)
        ALL_SAMPLES.append(current_sample)
    print('type of ALL_SAMPLES',        type(ALL_SAMPLES))
    print('type of ALL_SAMPLES[0]',     type(ALL_SAMPLES[0]))
    print('shape of ALL_SAMPLES[0]',    ALL_SAMPLES[0].shape)  #(4, 1024, 3)  
    print('shape of ALL_SAMPLES[1]',    ALL_SAMPLES[1].shape)  #(4, 1024, 3)   
    print('shape of ALL_SAMPLES[0][0]', ALL_SAMPLES[0][0].shape)
    
    trainset = prostateset_v2(ALL_SAMPLES)
    testset  = prostateset_v2(ALL_SAMPLES)
    
    
    # dataset
#    trainset = prostateset(args.traindata_path, loader)
#    testset  = prostateset(args.testdata_path, loader)

    # training
    run(args, trainset, testset)


def print_model_parameters(model):
    ''' 
    This function prints the model's parameters
    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
           print (name)
           #print (name, param.data)

def run(args, trainset, testset):
    if not torch.cuda.is_available():
       args.device = 'cpu'
    args.device = torch.device(args.device)   
    
    # the model 
    model = PhysicsInformedRegistration()
    model.cuda()
#    print('have passed the model to GPU')
    # optimizer
    min_loss = float('inf')
    print(min_loss)
    
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    
#    print('learnable_params in the constructed Model', learnable_params)
#    print('learnable_params in the constructed Model', print_model_parameters(model))    
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=0.001)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)    
    
    
    loss_values_training_save    = []
    loss_values_training_chamfer = []
    loss_values_training_pinns   = []
    # the individual terms in the pinns loss, training
    loss_values_training_governing_equation1 = []
    loss_values_training_governing_equation2 = []
    loss_values_training_governing_equation3 = []  
    loss_values_training_governing_equation4 = []
    loss_values_training_governing_equation5 = []
    loss_values_training_governing_equation6 = []
    loss_values_training_elastic_energy      = []
    loss_values_training_momentum_balance    = []
   
    loss_values_validation_save    = []  
    loss_values_validation_chamfer = []    
    loss_values_validation_pinns   = []     
    # the individual terms in the pinns loss, validation
    loss_values_validation_governing_equation1 = []
    loss_values_validation_governing_equation2 = []
    loss_values_validation_governing_equation3 = []  
    loss_values_validation_governing_equation4 = []
    loss_values_validation_governing_equation5 = []
    loss_values_validation_governing_equation6 = []
    loss_values_validation_elastic_energy      = []
    loss_values_validation_momentum_balance    = []
 
    # move the train and test DataLoader outside the epoch loop 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,  shuffle=False,  num_workers=args.workers,  drop_last =True)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size,  shuffle=False,  num_workers=args.workers,  drop_last =True)
       
    for epoch in range(args.start_epoch, args.epochs):    
        
        time_start_of_one_epoch = time.time()  # the starting time of one epoch
        
        '''
        First way, train the registration net together.   
        '''
        running_loss, running_loss_chamfer, running_loss_pinns,running_loss_governing_equation1, running_loss_governing_equation2, running_loss_governing_equation3, running_loss_governing_equation4, running_loss_governing_equation5, running_loss_governing_equation6, running_loss_elastic_energy, running_loss_momentum_balance = train_with_pinns(model, trainloader, optimizer, args.device)
       
       # val_loss, val_loss_chamfer, val_loss_pinns, val_loss_governing_equation1, val_loss_governing_equation2, val_loss_governing_equation3, val_loss_governing_equation4, val_loss_governing_equation5, val_loss_governing_equation6, val_loss_elastic_energy, val_loss_momentum_balance = eval_with_pinns(model,  testloader, args.device)        
        
        # if we use the validation during training
#        is_best  = val_loss < min_loss 
#        min_loss = min(val_loss, min_loss)    
        # if we do not use the validation during training
        is_best = running_loss<min_loss
        min_loss = min(running_loss, min_loss)    
        
        
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'optimizer': optimizer.state_dict(),}    
        
        if is_best:
            save_checkpoint(snap, args.outfile, 'snap_best')
            save_checkpoint(model.state_dict(), args.outfile, 'model_best')                                
        save_checkpoint(snap, args.outfile, 'snap_last')
        save_checkpoint(model.state_dict(),  args.outfile, 'model_last')        
        
        print('training loss all',    running_loss)
        print('running_loss_chamfer', running_loss_chamfer)        
        print('running_loss_pinns',   running_loss_pinns)
        
        # append and save the training loss 
        loss_values_training_save.append(running_loss)
        loss_values_training_chamfer.append(running_loss_chamfer)
        loss_values_training_pinns.append(running_loss_pinns)
        
        loss_values_training_governing_equation1.append(running_loss_governing_equation1)
        loss_values_training_governing_equation2.append(running_loss_governing_equation2)
        loss_values_training_governing_equation3.append(running_loss_governing_equation3)
        loss_values_training_governing_equation4.append(running_loss_governing_equation4)
        loss_values_training_governing_equation5.append(running_loss_governing_equation5)
        loss_values_training_governing_equation6.append(running_loss_governing_equation6)
        loss_values_training_elastic_energy.append(running_loss_elastic_energy)
        loss_values_training_momentum_balance.append(running_loss_momentum_balance)
        
        # append and save the validation loss
#        loss_values_validation_save.append(val_loss)
#        loss_values_validation_chamfer.append(val_loss_chamfer)
#        loss_values_validation_pinns.append(val_loss_pinns)
#        
#        loss_values_validation_governing_equation1.append(val_loss_governing_equation1)
#        loss_values_validation_governing_equation2.append(val_loss_governing_equation2)
#        loss_values_validation_governing_equation3.append(val_loss_governing_equation3)
#        loss_values_validation_governing_equation4.append(val_loss_governing_equation4)
#        loss_values_validation_governing_equation5.append(val_loss_governing_equation5)
#        loss_values_validation_governing_equation6.append(val_loss_governing_equation6)
#        loss_values_validation_elastic_energy.append(val_loss_elastic_energy)
#        loss_values_validation_momentum_balance.append(val_loss_momentum_balance) 
        
        # save the training loss during training 
        base_path = 'modified_codes_v10'       
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training.txt', 'w') as fp:
            for item in loss_values_training_save:
                # write each item on a new line
                fp.write("%s\n" % item)       
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_chamfer.txt', 'w') as fp:
            for item in loss_values_training_chamfer:
                # write each item on a new line
                fp.write("%s\n" % item)
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_pinns.txt', 'w') as fp:
            for item in loss_values_training_pinns:
                # write each item on a new line
                fp.write("%s\n" % item)     
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_governing_equation1.txt', 'w') as fp:
            for item in loss_values_training_governing_equation1:
                # write each item on a new line
                fp.write("%s\n" % item)                 
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_governing_equation2.txt', 'w') as fp:
            for item in loss_values_training_governing_equation2:
                # write each item on a new line
                fp.write("%s\n" % item)                                
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_governing_equation3.txt', 'w') as fp:
            for item in loss_values_training_governing_equation3:
                # write each item on a new line
                fp.write("%s\n" % item)                               
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_governing_equation4.txt', 'w') as fp:
            for item in loss_values_training_governing_equation4:
                # write each item on a new line
                fp.write("%s\n" % item)                
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_governing_equation5.txt', 'w') as fp:
            for item in loss_values_training_governing_equation5:
                # write each item on a new line
                fp.write("%s\n" % item)                 
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_governing_equation6.txt', 'w') as fp:
            for item in loss_values_training_governing_equation6:
                # write each item on a new line
                fp.write("%s\n" % item)                                 
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_elastic_energy.txt', 'w') as fp:
            for item in loss_values_training_elastic_energy:
                # write each item on a new line
                fp.write("%s\n" % item)                
        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/training_momentum_balance.txt', 'w') as fp:
            for item in loss_values_training_momentum_balance:
                # write each item on a new line
                fp.write("%s\n" % item)                 
                
                                        
        # save the validation loss during training             
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation.txt', 'w') as fp:
#            for item in loss_values_validation_save:
#                # write each item on a new line
#                fp.write("%s\n" % item)        
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_chamfer.txt', 'w') as fp:
#            for item in loss_values_validation_chamfer:
#                # write each item on a new line
#                fp.write("%s\n" % item)
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_pinns.txt', 'w') as fp:
#            for item in loss_values_validation_pinns:
#                # write each item on a new line
#                fp.write("%s\n" % item)                       
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_governing_equation1.txt', 'w') as fp:
#            for item in loss_values_validation_governing_equation1:
#                # write each item on a new line
#                fp.write("%s\n" % item) 
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_governing_equation2.txt', 'w') as fp:
#            for item in loss_values_validation_governing_equation2:
#                # write each item on a new line
#                fp.write("%s\n" % item)            
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_governing_equation3.txt', 'w') as fp:
#            for item in loss_values_validation_governing_equation3:
#                # write each item on a new line
#                fp.write("%s\n" % item)
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_governing_equation4.txt', 'w') as fp:
#            for item in loss_values_validation_governing_equation4:
#                # write each item on a new line
#                fp.write("%s\n" % item)
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_governing_equation5.txt', 'w') as fp:
#            for item in loss_values_validation_governing_equation5:
#                # write each item on a new line
#                fp.write("%s\n" % item)
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_governing_equation6.txt', 'w') as fp:
#            for item in loss_values_validation_governing_equation6:
#                # write each item on a new line
#                fp.write("%s\n" % item)
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_elastic_energy.txt', 'w') as fp:
#            for item in loss_values_validation_elastic_energy:
#                # write each item on a new line
#                fp.write("%s\n" % item)            
#        with open(r'/home/zmin/registration_mz/'+base_path+'/out_put_folder/validation_momentum_balance.txt', 'w') as fp:
#            for item in loss_values_validation_momentum_balance:
#                # write each item on a new line
#                fp.write("%s\n" % item)               
        
        # How much time does ONE EPOCH COST? How much time does ONE EPOCH COST?
        elapsed_time_of_one_epoch = time.time() - time_start_of_one_epoch
        print('How much time does ONE EPOCH COST? How much time does ONE EPOCH COST? How much time does ONE EPOCH COST?', elapsed_time_of_one_epoch)
        
        
        
        
        
if __name__ == '__main__': 
   print('The Main Function is Running')
   
   ARGS = options()
   print(ARGS.dim_k)
   print(ARGS.device) 
   main(ARGS)    
    
    
    
    
 
