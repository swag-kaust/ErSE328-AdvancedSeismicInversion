import time
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import deepwave
from  scipy.ndimage import gaussian_filter, gaussian_filter1d
import math
import numbers
from torch import nn
from torch.nn import functional as F



class FWI():
    def __init__(self,par,coordinate=True):
       self.nx=par['nx']
       self.nz=par['nz']
       self.dx=par['dx']
       self.nt=par['nt']
       self.dt=par['dt']
       self.num_dims=par['num_dims']
       self.num_shots=par['ns']
       self.num_batches=par['num_batches']
       self.num_sources_per_shot=1
       self.num_receivers_per_shot = par['nr']
       self.ds= par['ds']
       self.dr= par['dr']
       self.sz = par['sz']
       self.rz = par['rz']
       self.os = par['osou']
       self.orec = par['orec']
        
       if coordinate: self.s_cor, self.r_cor =self.get_coordinate()


    def get_coordinate(self):
       """
       Create arrays containing the source and receiver locations
       
       This assume same receivers for all the shots 
       
        x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
        x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions]
        Note: the depth is set to zero , to change it change the first element in the last dimensino .
       """
    
    
       x_s = torch.zeros(self.num_shots, self.num_sources_per_shot, self.num_dims)
       # x direction 
       x_s[:, 0, 1] = torch.arange(0,self.num_shots).float() * self.ds  + self.os  
       # z direction  
       x_s[:, 0, 0] = self.sz
       x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
       # x direction 
       x_r[0, :, 1] = torch.arange(0,self.num_receivers_per_shot).float() * self.dr + self.orec
       x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
       # z direction 
       x_r[:, :, 0] = self.rz

 
       return x_s,x_r


    def Ricker(self,freq):
        wavelet = (deepwave.wavelets.ricker(freq, self.nt, self.dt, 1/freq)
                                 .reshape(-1, 1, 1))                        
        return wavelet

    def forward_modelling(self,model,wavelet,device):
       prop = deepwave.scalar.Propagator({'vp': model.to(device)}, self.dx,pml_width=[0,20,20,20,20,20])
#                                           survey_pad=[None, None, 200, 200])
       data = prop(wavelet.to(device), self.s_cor.to(device), self.r_cor.to(device), self.dt).cpu()
       return data
    
    


    def run_inversion(self,model,data_t,wavelet,msk,niter,device,**kwargs): 
       """ 
      Run the FWI inversion,  
      ===================================
      Arguments: 
         model: torch.Tensor [nz.nx]: 
            Initial model for FWI 
         data_t: torch.Tensor [nt,ns,nr]: 
            Observed data
         wavelet: torch.Tensor [nt,1,1] or [nt,ns,1]
            wavelet 
         msk: torch.Tensor [nz,nx]:
            Mask for water layer
         niter: int: 
            Number of iteration 
         device: gpu or cpu  
       ==================================
      Optional: 
         vmin: int:
            upper bound for the update 
         vmax: int: 
            lower bound for the update 
         smth_flag: bool: 
            smoothin the gradient flag 
         smth: sequence of tuble or list: 
            each element define the amount of smoothing for different axes
         plot_flag: Bool
              whether to plot the update or not 
         save_freq: int 
              Save the update every 'save_freq' iteration
         plot_freq: int 
              if plot_flag is True. The plotting will be every 'plot_freq' iteratiom
         patient: int 
              patient for stopping criteria in case of incease in the objective
       """
        
       m_max = kwargs.pop('vmax', 5.0)
       m_min = kwargs.pop('vmin', 1.5)
       smth_flag = kwargs.pop('smth_flag', False)
       if smth_flag:
          smth = kwargs.pop('smth', None)
          assert smth != None, " 'smth' is not specified "

       plot_flag = kwargs.pop('plot_flag', False) 
       plot_freq = kwargs.pop('plot_freq', 5) 
       patient = kwargs.pop('patient', 5)
       save_freq = kwargs.pop('save_freq', 1)
        
       model = model.to(device)
       wavelet = wavelet.to(device)
       msk = torch.from_numpy(msk).float().to(device)
       model.requires_grad=True 
       criterion = torch.nn.MSELoss()
       LR = 0.01
       optimizer = torch.optim.Adam([{'params':[model],'lr':LR}])
       num_batches = self.num_batches
       num_shots_per_batch = int(self.num_shots / num_batches)
       prop = deepwave.scalar.Propagator({'vp': model}, self.dx,pml_width=[0,20,20,20,20,20])
       t_start = time.time()
       loss_iter=[]
       increase = 0
       tol = 1e-4
   
       if smth_flag: smoothing = GaussianSmoothing(1,kernel_size=30,sigma=[smth[0],smth[1]],dim=2).to(device)


       # updates is the output file
       updates=[]
       min_loss=0
       # main inversion loop 
       for itr in range(niter):
           running_loss = 0 
           optimizer.zero_grad()
           for it in range(num_batches): # loop over shots 
               batch_wavl = wavelet.repeat(1, num_shots_per_batch, 1)
               batch_data_t = data_t[:,it::num_batches].to(device)
               batch_x_s = self.s_cor[it::num_batches].to(device)
               batch_x_r = self.r_cor[it::num_batches].to(device)
               batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, self.dt) 
               loss1 = criterion(batch_data_pred, batch_data_t)
               loss1.backward()
               running_loss += loss1.item() 
           model.grad =  model.grad * msk #  mask 
           if smth_flag: model.grad = smoothing(model.grad) * msk
           if itr == 0 : gmax0 = (torch.abs(model.grad)).max() # get max of first itr 
           model.grad = model.grad / gmax0   # normalize by max of first iteration  
           optimizer.step()   
           model.data[model.data < m_min] = m_min
           model.data[model.data > m_max] = m_max
           loss_iter.append(running_loss)	
           ####plot the gradient
           if plot_flag and itr%plot_freq ==0 :
            gmin, gmax = np.percentile(model.grad.cpu().numpy(), [2,98])
            plt.figure(figsize=(10,3))
            plt.imshow(model.grad.cpu().numpy(),cmap='bwr',vmin=gmin,vmax=gmax)
            plt.title('gradient')
            plt.colorbar()
            plt.show()
            
            mmin, mmax = np.percentile(model.detach().clone().cpu().numpy(), [2,98])
            plt.figure(figsize=(10,3))
            plt.imshow(model.detach().clone().cpu().numpy(),cmap='jet',vmin=mmin,vmax=mmax)
            plt.title('model update')
            plt.colorbar()
            plt.show()
           print('Iteration: ', itr, 'Objective: ', running_loss)
           if itr > 0 and itr%save_freq==0 :
                 updates.append(model.detach().clone().cpu().numpy()) 
           # stopping criteria 
           if np.abs(loss_iter[itr] - loss_iter[itr-1])/max(loss_iter[itr],loss_iter[itr-1]) < tol and itr>5: 
               t_end = time.time()
               print('Runtime in min :',(t_end-t_start)/60)  
               updates.append(model.detach().clone().cpu().numpy()) 
               return np.array(updates),loss_iter
           elif min_loss < loss_iter[itr] and itr > 5: 
              increase +=1
           else: 
              increase = 0
              min_loss = loss_iter[itr]           
           if  increase == patient: 
               t_end = time.time()
               print('Runtime in min :',(t_end-t_start)/60)  
               updates.append(model.detach().clone().cpu().numpy()) 
               return np.array(updates),loss_iter
       t_end = time.time()
       print('Runtime in min :',(t_end-t_start)/60)         
       updates.append(model.detach().clone().cpu().numpy()) 
       return np.array(updates),loss_iter




            




class GaussianSmoothing(nn.Module):
    """
    Borrowed from: 
     
     https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
     
    Currently only working on square kernel. 
    change sigma for rectangular smoothing but I think bounded by the size of the filtere
    =============================
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.kernel_size = kernel_size  
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        
        ## ================ Added by AA
        k = self.kernel_size-1
        if len(input.size())==1:
            x = input.reshape(1,1,input.shape[0])
            x = F.pad(x,(math.floor(k/2),math.ceil(k/2)),mode='replicate')
        elif len(input.size())==2:
            x = input.reshape(1,1,input.shape[0],input.shape[1])
            x = F.pad(x,(math.floor(k/2),math.ceil(k/2),math.floor(k/2),math.ceil(k/2)),mode='replicate')
        # ===============================
        x = self.conv(x, weight=self.weight, groups=self.groups)
        # =============================== Added by AA 
        if len(input.size())==1:   x = x.reshape(x.shape[2])
        elif len(input.size())==2: x = x.reshape(x.shape[2],x.shape[3])

        return  x