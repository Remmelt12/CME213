# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:00:48 2018

@author: Remmelt
"""

import numpy as np
import matplotlib.pyplot as plt

size=np.array([256,512,1024,2048,4096])
data1=np.array([[78.0242,134.535,164.204,196.817,205.351],
          [82.0856,149.318,187.521,226.836,238.991],
          [82.2955,183.134,277.259,336.6,361.28]])

data2=np.array([[52.0075,99.5728,141.091,165.072,172.595],
                [82.0856,149.318,187.521,226.836,238.991],
                [117.917,183.181,204.842,270.594,293.595]])

data3=np.array([[46.5974,105.313,156.341,185.915,198.226],
                [82.2955,183.134,277.259,336.6,361.28],
                [165.183,383.552,637.333,821.412,903.237]])

plt.plot((size**2)*10**(-6),data1[0,:],label='global')
plt.plot((size**2)*10**(-6),data1[1,:],label='block')
plt.plot((size**2)*10**(-6),data1[2,:],label='shared')
plt.xlabel('problem size (Megapoints)')
plt.title('Bandwith for the different methods.')
plt.ylabel('Bandwidth (GB/s)')
plt.legend(loc=4)
plt.savefig('3-1.png')
plt.show()

plt.plot((size**2)*10**(-6),data2[0,:],label='order = 2')
plt.plot((size**2)*10**(-6),data2[1,:],label='order = 4')
plt.plot((size**2)*10**(-6),data2[2,:],label='order = 8')
plt.xlabel('problem size (Megapoints)')
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwithd of the block method for different stencils.')
plt.legend(loc=4)
plt.savefig('3-2.png')
plt.show()

plt.title('Bandwithd of the shared method for different stencils.')
plt.plot((size**2)*10**(-6),data3[0,:],label='order = 2')
plt.plot((size**2)*10**(-6),data3[1,:],label='order = 4')
plt.plot((size**2)*10**(-6),data3[2,:],label='order = 8')
plt.xlabel('problem size (Megapoints)')
plt.ylabel('Bandwidth (GB/s)')
plt.legend()
plt.savefig('3-3.png')

