import numpy as np



#   double[][] normalizedMedianTest(double[][] paramArrayOfdouble, double paramDouble1, double paramDouble2) {
#     byte b1 = 15;
#     for (byte b2 = 0; b2 < paramArrayOfdouble.length; b2++) {
#       for (byte b = 2; b < 4; b++) {
#         double[] arrayOfDouble = getNeighbours(paramArrayOfdouble, b2, b, b1);
#         if (arrayOfDouble != null) {
#           double d = Math.abs(paramArrayOfdouble[b2][b] - getMedian(arrayOfDouble)) / (getMedian(getResidualsOfMedian(arrayOfDouble)) + paramDouble1);
#           if (d > paramDouble2)
#             paramArrayOfdouble[b2][b1] = -1.0D; 
#         } else {
#           paramArrayOfdouble[b2][b1] = -1.0D;
#         } 
#       } 
#     } 
#     return paramArrayOfdouble;
#   }
  


#   double[] getNeighbours(double[][] paramArrayOfdouble, int paramInt1, int paramInt2, int paramInt3) {
#     double[] arrayOfDouble = new double[9];
#     int i = paramInt1 / this.nx;
#     int j = paramInt1 - i * this.nx;
#     byte b = 0;
#     int k = 0;
#     k = paramInt1 - this.nx - 1;
#     if (i - 1 >= 0 && j - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -1.0D) {
#       b++;
#       arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
#     } 
#     k = paramInt1 - this.nx;
#     if (i - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -1.0D) {
#       b++;
#       arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
#     } 
#     k = paramInt1 - this.nx + 1;
#     if (i - 1 >= 0 && j + 1 < this.nx && paramArrayOfdouble[k][paramInt3] != -1.0D) {
#       b++;
#       arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
#     } 
#     k = paramInt1 - 1;
#     if (j - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -1.0D) {
#       b++;
#       arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
#     } 
#     k = paramInt1 + 1;
#     if (j + 1 < this.nx && paramArrayOfdouble[k][paramInt3] != -1.0D) {
#       b++;
#       arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
#     } 
#     k = paramInt1 + this.nx - 1;
#     if (i + 1 < this.ny && j - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -1.0D) {
#       b++;
#       arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
#     } 
#     k = paramInt1 + this.nx;
#     if (i + 1 < this.ny && paramArrayOfdouble[k][paramInt3] != -1.0D) {
#       b++;
#       arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
#     } 
#     k = paramInt1 + this.nx + 1;
#     if (i + 1 < this.ny && j + 1 < this.nx && paramArrayOfdouble[k][paramInt3] != -1.0D) {
#       b++;
#       arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
#     } 
#     if (b > 0) {
#       double[] arrayOfDouble1 = new double[b];
#       System.arraycopy(arrayOfDouble, 0, arrayOfDouble1, 0, b);
#       return arrayOfDouble1;
#     } 
#     return null;
#   }
    


#   double[][] replaceByMedian(double[][] paramArrayOfdouble) {
#     byte b1 = 15;
#     double[][] arrayOfDouble = new double[paramArrayOfdouble.length][(paramArrayOfdouble[0]).length];
#     for (byte b2 = 0; b2 < arrayOfDouble.length; b2++)
#       System.arraycopy(paramArrayOfdouble[b2], 0, arrayOfDouble[b2], 0, (paramArrayOfdouble[b2]).length); 
#     for (byte b3 = 0; b3 < arrayOfDouble.length; b3++) {
#       if (arrayOfDouble[b3][b1] == -1.0D) {
#         for (byte b = 2; b <= 3; b++) {
#           double[] arrayOfDouble1 = getNeighbours(paramArrayOfdouble, b3, b, b1);
#           if (arrayOfDouble1 != null) {
#             arrayOfDouble[b3][b] = getMedian(arrayOfDouble1);
#             arrayOfDouble[b3][b1] = 999.0D;
#           } else {
#             arrayOfDouble[b3][b] = 0.0D;
#             arrayOfDouble[b3][b1] = -2.0D;
#           } 
#         } 
#         if (arrayOfDouble[b3][b1] != -2.0D)
#           arrayOfDouble[b3][4] = Math.sqrt(arrayOfDouble[b3][2] * arrayOfDouble[b3][2] + arrayOfDouble[b3][3] * arrayOfDouble[b3][3]); 
#       } 
#     } 
#     return arrayOfDouble;
#   }


# -*- coding: utf-8 -*-
"""
Project: Particle Image Velocimetry (PIV) code!
@author: A. F. Forughi (Aug. 2020, Last update: Jan. 2021)
"""

# %% Libraries:
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm # pip install tqdm
from numba import jit # pip install numba

# %% Functions:
@jit(nopython=True)
def corr2(c1,c2): # Cross-correlation
    c1-=c1.mean()
    c2-=c2.mean()
    c12=(c1*c1).sum()*(c2*c2).sum()
    if c12>0.0:
        return (c1*c2).sum()/np.sqrt(c12)
    return -1.0

def fixer(vecx,vecy,vec,rij,r_limit,i_fix): # Fixing the irregular vectors (Normalized Median Test and low Correlation coeff.)
    fluc=np.zeros(vec.shape)
    for j in range(1,vec.shape[1]-1):
        for i in range(1,vec.shape[0]-1):
            neigh_x=np.array([])
            neigh_y=np.array([])
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if ii==0 and jj==0: continue
                    neigh_x=np.append(neigh_x,vecx[i+ii,j+jj]) # Neighbourhood components
                    neigh_y=np.append(neigh_y,vecy[i+ii,j+jj])
            res_x=neigh_x-np.median(neigh_x) # Residual
            res_y=neigh_y-np.median(neigh_y)
            
            res_s_x=np.abs(vecx[i,j]-np.median(neigh_x))/(np.median(np.abs(res_x))+0.1) # Normalized Residual (Epsilon=0.1)
            res_s_y=np.abs(vecy[i,j]-np.median(neigh_y))/(np.median(np.abs(res_y))+0.1)
            
            fluc[i,j]=np.sqrt(res_s_x*res_s_x+res_s_y*res_s_y) # Normalized Fluctuations
    # plt.contourf(fluc,levels=np.arange(2,200,0.1))#,vmin=0.0,vmax=2 # To see the outliers
    # plt.colorbar(label='Normalized Fluctuation')
    
    i_disorder=0
    for ii in range(i_fix): # Correction Cycle for patches of bad data
        i_disorder=0
        vec_diff=0.0
        for j in range(1,vec.shape[1]-1):
            for i in range(1,vec.shape[0]-1):
                if fluc[i,j]>2.0 or (rij[i,j]<r_limit): # Fluctuation threshold = 2.0
                    i_disorder+=1
                    vecx[i,j]=0.25*(vecx[i+1,j]+vecx[i-1,j]+vecx[i,j+1]+vecx[i,j-1]) # Bilinear Fix
                    vecy[i,j]=0.25*(vecy[i+1,j]+vecy[i-1,j]+vecy[i,j+1]+vecy[i,j-1])
                    vec_diff+=(vec[i,j]-np.sqrt(vecx[i,j]*vecx[i,j]+vecy[i,j]*vecy[i,j]))**2.0
                    vec[i,j]=np.sqrt(vecx[i,j]*vecx[i,j]+vecy[i,j]*vecy[i,j])
                    
        if i_disorder==0 or vec.mean()==0.0: break # No need for correction
        correction_residual=vec_diff/(i_disorder*np.abs(vec.mean()))
        if correction_residual<1.0e-20: break # Converged!
    if ii==i_fix-1: print("Maximum correction iteration was reached!")
    return vecx,vecy,vec,i_disorder,ii


def subpix(R,axis): # Subpixle resolution (parabolic-Gaussian fit)
    dum=np.floor(np.argmax(R)/R.shape[0])    
    R_x=int(dum) #vecy
    R_y=int(np.argmax(R)-dum*R.shape[0])  #vecx
    r=R[R_x,R_y]
    if np.abs(r-1.0)<0.01: return 0.0
    try: # Out of bound at the edges:
        if axis == 'y': #For vecy
            r_e=R[R_x+1,R_y]
            r_w=R[R_x-1,R_y]
        else:          #For Vecx
            r_e=R[R_x,R_y+1]
            r_w=R[R_x,R_y-1]
        if r_e>0.0 and r_w>0.0 and r>0.0: # Gaussian if possible (resolves pick locking)
            r_e=np.log(r_e)
            r_w=np.log(r_w)
            r=np.log(r)
        if (r_e+r_w-2*r)!=0.0:
            if np.abs((r_w-r_e)/(2.0*(r_e+r_w-2*r)))<1.0 and np.abs(r_e+1)>0.01 and np.abs(r_w+1)>0.01:
                return (r_w-r_e)/(2.0*(r_e+r_w-2*r))
        return 0.0
    except:
        return 0.0

# %% loading images and setting the parameters:


import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

import sys
sys.path.insert(0, '/Users/ryan/Documents/GitHub/')

from pytraction.utils import allign_slice
from pytraction.piv import PIV
from pytraction.fourier import fourier_xu

from openpiv import tools, pyprocess, validation, filters, scaling 


frame = 0
channel = 0

meshsize = 10 # grid spacing in pix
pix_durch_mu = 1.3

E = 100 # Young's modulus in Pa
s = 0.3 # Poisson's ratio


# load data from ryan
rg_file = glob.glob('data/*')


img1 = '/Users/ryan/Documents/GitHub/pytraction/data/e01_pos1_axon1.tif'
ref1 = '/Users/ryan/Documents/GitHub/pytraction/data/e01_pos1_axon1_ref.tif'

img1 = io.imread(img1)
ref1 = io.imread(ref1)

img_2 = np.array(img1[frame, channel, :, :], dtype='float32')
img_1 = np.array(ref1[channel,:,:], dtype='float32')



# img_1 = (np.flip(cv2.imread('Python Examples/a1.png', 0),0)).astype('float32') # Read Grayscale
# img_2 = (np.flip(cv2.imread('Python Examples/a2.png', 0),0)).astype('float32')

# plt.imshow(img_1)
# plt.show()

i_fix=500     # Number of maximum correction cycles
r_limit=0.1   # minimum acceptable correlation coefficient
l_scale=1.3e-6   # spatial scale [m/pixel]
t_scale=1.0   # time step = 1/frame_rate [s/frame]

# iw=32 # Interrodation Windows Sizes (pixel)
# sw=64 # Search Windows Sizes (sw > iw) (pixel)

# iw = 128
# sw = 256

iw = 48
sw = 128


# %% Search Algorithm:
ia,ja = img_1.shape
iw=int(2*np.floor((iw+1)/2)-1) # Even->Odd
sw=int(2*np.floor((sw+1)/2)-1)
margin=int((sw-iw)/2)
im=int(2*np.floor((ia-1-iw)/(iw-1))) # Number of I.W.s in x direction
jm=int(2*np.floor((ja-1-iw)/(iw-1))) # Number of I.W.s in y direction

vecx=np.zeros((im,jm)) # x-Displacement
vecy=np.zeros((im,jm)) # y-Displacement
vec=np.zeros((im,jm)) # Magnitude
rij=np.zeros((im,jm)) # Correlation coeff.

for j in tqdm(range(jm)):
    j_d=int(j*(iw-1)/2) # Bottom bound
    j_u=j_d+iw          # Top bound
    sw_d=max(0,j_d-margin) # First Row
    sw_d_diff=max(0,j_d-margin)-(j_d-margin)
    sw_u=min(ja-1,j_u+margin) # Last Row
    
    for i in range(im):
        i_l=int(i*(iw-1)/2) # Left bound
        i_r=i_l+iw          # Right bound
        sw_l=max(0,i_l-margin) # First column
        sw_l_diff=max(0,i_l-margin)-(i_l-margin)
        sw_r=min(ia-1,i_r+margin) # Last column
        
        R=np.zeros((sw-iw+1,sw-iw+1))-1 # Correlation Matrix
        c1=np.array(img_1[i_l:i_l+iw,j_d:j_d+iw]) # IW from 1st image
        for jj in range(sw_d,sw_u+1-iw):
            for ii in range(sw_l,sw_r+1-iw):
                c2=np.array(img_2[ii:ii+iw,jj:jj+iw]) # IW from 2nd image
                R[ii-sw_l,jj-sw_d]=corr2(c1,c2)
        rij[i,j]=R.max()
        if rij[i,j]>=r_limit:
            dum=np.floor(np.argmax(R)/R.shape[0])
            vecy[i,j]=dum-(margin-sw_l_diff)+subpix(R,'y')
            vecx[i,j]=np.argmax(R)-dum*R.shape[0]-(margin-sw_d_diff)+subpix(R,'x')
            vec[i,j]=np.sqrt(vecx[i,j]*vecx[i,j]+vecy[i,j]*vecy[i,j])
        else:
            vecx[i,j]=0.0;vecy[i,j]=0.0;vec[i,j]=0.0
        
# %% Corrections:
# vecx,vecy,vec,i_disorder,i_cor_done=fixer(vecx,vecy,vec,rij,r_limit,i_fix)

# %% Applying the scales:
X, Y = np.meshgrid(np.arange(0.5*iw, 0.5*iw*(jm+1), 0.5*iw), 
                   np.arange(0.5*iw, 0.5*iw*(im+1), 0.5*iw))
X*=l_scale
Y*=l_scale

vecx*=(l_scale/t_scale);vecy*=(l_scale/t_scale);vec*=(l_scale/t_scale)


# %% Exporting Data:

# np.savez('results.npz', vecx=vecx, vecy=vecy, vec=vec, rij=rij)
# res=np.load('results.npz'); vecx=res['vecx']; vecy=res['vecy']; vec=res['vec']; rij=res['rij']; # Load saved data

# fig, ax = plt.subplots(figsize=(8,8), dpi=300)
plt.quiver(X, Y, vecx, vecy,units='width')
plt.show()




package PIV;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.GenericDialog;
import ij.gui.WaitForUserDialog;
import ij.io.FileSaver;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.measure.ResultsTable;
import ij.plugin.filter.MaximumFinder;
import ij.plugin.filter.PlugInFilter;
import ij.process.ColorProcessor;
import ij.process.FHT;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Locale;
import org.opensourcephysics.display2d.GridPointData;

public class iterative_PIV implements PlugInFilter {
  String arg;
  
  ImagePlus imp;
  
  int width;
  
  int height;
  
  String title;
  
  String file;
  
  String piv0Path;
  
  private static int winS1;
  
  private static int vecS1;
  
  private static int sW1;
  
  private static int winS2;
  
  private static int vecS2;
  
  private static int sW2;
  
  private static int winS3;
  
  private static int vecS3;
  
  private static int sW3;
  
  int nPass = 3;
  
  private static double cThr;
  
  boolean db = false;
  
  boolean batch;
  
  boolean pp = false;
  
  boolean dCanceled = false;
  
  boolean xc = true;
  
  boolean chkPeakA = false;
  
  boolean noChkPeak = false;
  
  String ppMethod;
  
  double pp1;
  
  double pp2;
  
  int dbX = -1;
  
  int dbY = -1;
  
  private static String dir = "";
  
  int nx;
  
  int ny;
  
  double[][] PIVdata1;
  
  double[][] PIVdata;
  
  double[][] PIVdata0;
  
  double[][][] PIV0;
  
  int action = 3;
  
  double noiseNMT1 = 0.2D, thrNMT1 = 5.0D, c1DMT = 3.0D, c2DMT = 1.0D;
  
  double sdR;
  
  double meanR;
  
  double max0 = 0.0D;
  
  public int setup(String paramString, ImagePlus paramImagePlus) {
    this.arg = paramString;
    this.imp = paramImagePlus;
    return 2053;
  }
  
  public void run(ImageProcessor paramImageProcessor) {
    int i = this.imp.getImageStackSize();
    this.title = this.imp.getTitle();
    this.width = this.imp.getWidth();
    this.height = this.imp.getHeight();
    String str = "_PIV1";
    if (i != 2)
      IJ.error("2 slices stack is required"); 
    if (this.arg.equals("Cross-correlation")) {
      if (!getParamsC()) {
        this.imp.changes = false;
        return;
      } 
    } else if (this.arg.equals("Basic")) {
      if (!getParamsB()) {
        this.imp.changes = false;
        return;
      } 
    } else if (this.arg.equals("Debug")) {
      if (!getParamsD()) {
        this.imp.changes = false;
        return;
      } 
    } else if (!getParamsA()) {
      this.imp.changes = false;
      return;
    } 
    IJ.log("PIV paramters: ");
    IJ.log("pass1: Interrogation window=" + winS1 + " search window=" + sW1 + " vector spacing=" + vecS1);
    IJ.log("pass2: Interrogation window=" + winS2 + " search window=" + sW2 + " vector spacing=" + vecS2);
    IJ.log("pass3: Interrogation window=" + winS3 + " search window=" + sW3 + " vector spacing=" + vecS3);
    if (this.noChkPeak) {
      IJ.log("Peak check disabled");
    } else if (this.chkPeakA) {
      IJ.log("Using emperical parameters for peak check");
    } 
    for (byte b = 1; b <= this.nPass; b++) {
      int j = winS1;
      int k = vecS1;
      int m = sW1;
      if (b == 1) {
        if (this.piv0Path != null) {
          try {
            this.PIVdata0 = plot_.load2DArrayFromFile(this.piv0Path);
          } catch (Exception exception) {
            IJ.error(exception.getMessage());
          } 
          plotPIV(this.PIVdata0, this.title + "_PIV0", false);
        } 
      } else if (b == 2) {
        this.PIVdata0 = this.PIVdata;
        j = winS2;
        k = vecS2;
        m = sW2;
        str = "_PIV2";
      } else if (b == 3) {
        this.PIVdata0 = this.PIVdata;
        j = winS3;
        k = vecS3;
        m = sW3;
        str = "_PIV3";
      } 
      if (this.PIVdata0 == null) {
        this.PIV0 = new double[1][1][1];
      } else {
        int[] arrayOfInt = plot_.getDimensions(this.PIVdata0);
        this.PIV0 = plot_.convert2DPivTo3D(this.PIVdata0, arrayOfInt[0], arrayOfInt[1]);
      } 
      this.PIVdata = doPIV(this.imp, j, k, m, this.PIV0);
      if (!this.pp) {
        this.PIVdata = replaceByMedian(this.PIVdata);
        this.PIVdata = replaceByMedian2(this.PIVdata);
      } 
      plotPIV(this.PIVdata, this.title + str, false);
      if (this.db) {
        IJ.log("" + this.title + str + ":");
        logPIV(this.PIVdata);
      } 
      if (this.batch) {
        StringBuffer stringBuffer = generatePIVToPrint(this.PIVdata);
        write2File(dir, this.title + str + "_disp.txt", stringBuffer.toString());
      } 
    } 
    if (!this.batch) {
      this.PIVdata1 = pivPostProcess(this.PIVdata);
      ImagePlus imagePlus = WindowManager.getImage(this.title + str);
      if (imagePlus != null)
        imagePlus.close(); 
      plotPIV(this.PIVdata1, this.title + str, true);
      if (this.dCanceled) {
        IJ.log("" + this.title + str + ":");
        logPIV(this.PIVdata1);
      } 
    } else if (this.ppMethod != "None") {
      this.PIVdata1 = pivPostProcess_batch(this.PIVdata);
      StringBuffer stringBuffer = generatePIVToPrint(this.PIVdata1);
      write2File(dir, this.title + str + "_" + this.ppMethod + "_disp.txt", stringBuffer.toString());
    } 
    this.imp.changes = false;
    IJ.freeMemory();
  }
  
  private double[][] pivPostProcess(double[][] paramArrayOfdouble) {
    double[][] arrayOfDouble = new double[paramArrayOfdouble.length][(paramArrayOfdouble[0]).length];
    byte b;
    for (b = 0; b < arrayOfDouble.length; b++)
      System.arraycopy(paramArrayOfdouble[b], 0, arrayOfDouble[b], 0, (paramArrayOfdouble[b]).length); 
    if (this.db) {
      WaitForUserDialog waitForUserDialog = new WaitForUserDialog("pause");
      waitForUserDialog.show();
    } 
    b = 0;
    do {
      byte b1;
      if (!getParamsP()) {
        this.dCanceled = true;
        return paramArrayOfdouble;
      } 
      ImagePlus imagePlus1 = WindowManager.getImage(this.title + "_temp");
      switch (this.action) {
        case 0:
          arrayOfDouble = normalizedMedianTest(arrayOfDouble, this.noiseNMT1, this.thrNMT1);
          arrayOfDouble = replaceByMedian(arrayOfDouble);
          break;
        case 1:
          arrayOfDouble = dynamicMeanTest(arrayOfDouble, this.c1DMT, this.c2DMT);
          arrayOfDouble = replaceByMedian(arrayOfDouble);
          break;
        case 2:
          for (b1 = 0; b1 < arrayOfDouble.length; b1++)
            System.arraycopy(paramArrayOfdouble[b1], 0, arrayOfDouble[b1], 0, (paramArrayOfdouble[b1]).length); 
          break;
        case 3:
          b = 1;
          break;
      } 
      if (imagePlus1 != null)
        imagePlus1.close(); 
      plotPIV(arrayOfDouble, this.title + "_temp", false);
      if (!this.db)
        continue; 
      logPIV(arrayOfDouble);
    } while (b == 0);
    ImagePlus imagePlus = WindowManager.getImage(this.title + "_temp");
    if (imagePlus != null)
      imagePlus.close(); 
    if (this.db) {
      IJ.log("PIV post process:");
      logPIV(arrayOfDouble);
    } 
    if (this.action == 3) {
      StringBuffer stringBuffer = generatePIVToPrint(arrayOfDouble);
      write2File(dir, this.file, stringBuffer.toString());
    } 
    this.dCanceled = false;
    return arrayOfDouble;
  }
  
  private double[][] pivPostProcess_batch(double[][] paramArrayOfdouble) {
    double[][] arrayOfDouble = new double[paramArrayOfdouble.length][(paramArrayOfdouble[0]).length];
    for (byte b = 0; b < arrayOfDouble.length; b++)
      System.arraycopy(paramArrayOfdouble[b], 0, arrayOfDouble[b], 0, (paramArrayOfdouble[b]).length); 
    if (this.ppMethod == "NMT") {
      arrayOfDouble = normalizedMedianTest(arrayOfDouble, this.noiseNMT1, this.thrNMT1);
      arrayOfDouble = replaceByMedian(arrayOfDouble);
    } else {
      arrayOfDouble = dynamicMeanTest(arrayOfDouble, this.c1DMT, this.c2DMT);
      arrayOfDouble = replaceByMedian(arrayOfDouble);
    } 
    return arrayOfDouble;
  }
  
  private void plotPIV(double[][] paramArrayOfdouble, String paramString, boolean paramBoolean) {
    double d2;
    ColorProcessor colorProcessor = new ColorProcessor(this.width, this.height);
    int[] arrayOfInt = plot_.getDimensions(paramArrayOfdouble);
    double[][] arrayOfDouble = plot_.get2DElement(paramArrayOfdouble, arrayOfInt[0], arrayOfInt[1], 4);
    double d1 = plot_.findMax2DArray(arrayOfDouble);
    plot_.colorMax = d1;
    if (this.max0 == 0.0D) {
      d2 = 24.0D / d1;
      this.max0 = d1;
    } else {
      d2 = 24.0D / this.max0;
      plot_.colorMax = this.max0;
    } 
    plot_.loadLut("S_Pet");
    plot_.drawVectors((ImageProcessor)colorProcessor, arrayOfInt, paramArrayOfdouble, arrayOfDouble, d2, plot_.colors);
    ImagePlus imagePlus = new ImagePlus(paramString, (ImageProcessor)colorProcessor);
    imagePlus.show();
    if (paramBoolean)
      plot_.makeScaleGraph(d2); 
    if (this.batch) {
      FileSaver fileSaver = new FileSaver(imagePlus);
      fileSaver.saveAsTiff(dir + paramString + "_vPlot.tif");
    } 
  }
  
  private boolean getParamsA() {
    GenericDialog genericDialog = new GenericDialog("Iterative PIV (Advanced Mode)");
    genericDialog.addCheckbox("Load file as 0th pass PIV data?", false);
    genericDialog.addMessage("(All sizes are in pixels)");
    genericDialog.addMessage("1st pass PIV parameters:");
    if (winS1 == 0)
      winS1 = 128; 
    genericDialog.addNumericField("PIV1 interrogation window size", winS1, 0);
    if (sW1 == 0)
      sW1 = 256; 
    genericDialog.addMessage("(If search window size=window size, conventional xcorr will be used)");
    genericDialog.addNumericField("SW1 :search window size", sW1, 0);
    if (vecS1 == 0)
      vecS1 = 64; 
    genericDialog.addNumericField("VS1 :Vector spacing", vecS1, 0);
    genericDialog.addMessage("-----------------------");
    genericDialog.addMessage("2nd pass PIV parameters: (set window size to zero to do only 1pass PIV)");
    if (winS2 == 0)
      winS2 = 64; 
    genericDialog.addNumericField("PIV2 interrogation window size", winS2, 0);
    if (sW2 == 0)
      sW2 = 128; 
    genericDialog.addNumericField("SW2 :search window size", sW2, 0);
    if (vecS2 == 0)
      vecS2 = 32; 
    genericDialog.addNumericField("VS2 :Vector spacing", vecS2, 0);
    genericDialog.addMessage("-----------------------");
    genericDialog.addMessage("3rd pass PIV parameters: (set window size to zero to do only 2pass PIV)");
    if (winS3 == 0)
      winS3 = 48; 
    genericDialog.addNumericField("PIV3 interrogation window size", winS3, 0);
    if (sW3 == 0)
      sW3 = 128; 
    genericDialog.addNumericField("SW3 :search window size", sW3, 0);
    if (vecS3 == 0)
      vecS3 = 16; 
    genericDialog.addNumericField("VS3 :Vector spacing", vecS3, 0);
    genericDialog.addMessage("-----------------------");
    if (cThr == 0.0D)
      cThr = 0.6D; 
    genericDialog.addNumericField("correlation threshold", cThr, 2);
    genericDialog.addCheckbox("Use advanced peak check? (empirical parameters)", false);
    genericDialog.addCheckbox("Disable all peak checking?", false);
    genericDialog.addCheckbox("Don't replace invalid vector by median?", false);
    genericDialog.addCheckbox("batch mode?", false);
    genericDialog.addChoice("Postprocessing", new String[] { "None", "NMT", "DMT" }, "None");
    genericDialog.addNumericField("Postprocessing parameter1", 0.2D, 2);
    genericDialog.addNumericField("Postprocessing parameter1", 5.0D, 2);
    if (dir.equals(""))
      dir = "/"; 
    genericDialog.addStringField("Path to save outputs", dir, 30);
    genericDialog.showDialog();
    boolean bool = genericDialog.getNextBoolean();
    winS1 = (int)genericDialog.getNextNumber();
    sW1 = (int)genericDialog.getNextNumber();
    vecS1 = (int)genericDialog.getNextNumber();
    winS2 = (int)genericDialog.getNextNumber();
    sW2 = (int)genericDialog.getNextNumber();
    vecS2 = (int)genericDialog.getNextNumber();
    winS3 = (int)genericDialog.getNextNumber();
    sW3 = (int)genericDialog.getNextNumber();
    vecS3 = (int)genericDialog.getNextNumber();
    cThr = genericDialog.getNextNumber();
    this.chkPeakA = genericDialog.getNextBoolean();
    this.noChkPeak = genericDialog.getNextBoolean();
    this.pp = genericDialog.getNextBoolean();
    this.batch = genericDialog.getNextBoolean();
    this.ppMethod = genericDialog.getNextChoice();
    this.pp1 = genericDialog.getNextNumber();
    this.pp2 = genericDialog.getNextNumber();
    if (this.ppMethod == "NMT") {
      this.noiseNMT1 = this.pp1;
      this.thrNMT1 = this.pp2;
    } else {
      this.c1DMT = this.pp1;
      this.c2DMT = this.pp2;
    } 
    dir = genericDialog.getNextString();
    if (vecS3 == 0 || sW3 == 0 || winS3 == 0)
      this.nPass = 2; 
    if (vecS2 == 0 || sW2 == 0 || winS2 == 0)
      this.nPass = 1; 
    if (!genericDialog.wasCanceled()) {
      if (!checkParams()) {
        IJ.error("Incompatible PIV parameters");
        return false;
      } 
      if (bool) {
        OpenDialog openDialog = new OpenDialog("Select the PIV data", "");
        if (openDialog.getDirectory() == null || openDialog.getFileName() == null)
          return false; 
        this.piv0Path = openDialog.getDirectory();
        this.piv0Path += openDialog.getFileName();
      } 
    } else {
      return false;
    } 
    return true;
  }
  
  private boolean getParamsB() {
    GenericDialog genericDialog = new GenericDialog("Iterative PIV (Basic)");
    genericDialog.addMessage("(All sizes are in pixels)");
    genericDialog.addMessage("1st pass PIV parameters:");
    if (winS1 == 0)
      winS1 = 128; 
    genericDialog.addNumericField("PIV1 interrogation window size", winS1, 0);
    if (sW1 == 0)
      sW1 = 256; 
    genericDialog.addMessage("(If search window size=window size, conventional xcorr will be used)");
    genericDialog.addNumericField("SW1 :search window size", sW1, 0);
    genericDialog.addMessage("-----------------------");
    genericDialog.addMessage("2nd pass PIV parameters: (set window size to zero to do only 1pass PIV)");
    if (winS2 == 0)
      winS2 = 64; 
    genericDialog.addNumericField("PIV2 interrogation window size", winS2, 0);
    if (sW2 == 0)
      sW2 = 128; 
    genericDialog.addNumericField("SW2 :search window size", sW2, 0);
    genericDialog.addMessage("-----------------------");
    genericDialog.addMessage("3rd pass PIV parameters: (set window size to zero to do only 2pass PIV)");
    if (winS3 == 0)
      winS3 = 32; 
    genericDialog.addNumericField("PIV3 interrogation window size", winS3, 0);
    if (sW3 == 0)
      sW3 = 96; 
    genericDialog.addNumericField("SW3 :search window size", sW3, 0);
    genericDialog.addMessage("-----------------------");
    if (cThr == 0.0D)
      cThr = 0.6D; 
    genericDialog.addNumericField("correlation threshold", cThr, 2);
    genericDialog.showDialog();
    winS1 = (int)genericDialog.getNextNumber();
    sW1 = (int)genericDialog.getNextNumber();
    vecS1 = winS1 / 2;
    winS2 = (int)genericDialog.getNextNumber();
    sW2 = (int)genericDialog.getNextNumber();
    vecS2 = winS2 / 2;
    winS3 = (int)genericDialog.getNextNumber();
    sW3 = (int)genericDialog.getNextNumber();
    vecS3 = winS3 / 2;
    cThr = genericDialog.getNextNumber();
    if (vecS3 == 0 || sW3 == 0 || winS3 == 0)
      this.nPass = 2; 
    if (vecS2 == 0 || sW2 == 0 || winS2 == 0)
      this.nPass = 1; 
    if (!checkParams()) {
      IJ.error("Incompatible PIV parameters");
      return false;
    } 
    if (genericDialog.wasCanceled())
      return false; 
    return true;
  }
  
  private boolean getParamsC() {
    GenericDialog genericDialog = new GenericDialog("Iterative PIV (Cross-Correlation)");
    genericDialog.addMessage("(All sizes are in pixels)");
    if (winS1 == 0)
      winS1 = 128; 
    genericDialog.addNumericField("PIV1 interrogation window size", winS1, 0);
    if (winS2 == 0)
      winS2 = 64; 
    genericDialog.addMessage("(set PIV2 window size to zero to do only 1 pass PIV)");
    genericDialog.addNumericField("PIV2 interrogation window size", winS2, 0);
    if (winS3 == 0)
      winS3 = 32; 
    genericDialog.addMessage("(set PIV3 window size to zero to do only 2 pass PIV)");
    genericDialog.addNumericField("PIV3 interrogation window size", winS3, 0);
    genericDialog.showDialog();
    winS1 = (int)genericDialog.getNextNumber();
    sW1 = winS1;
    vecS1 = winS1 / 2;
    winS2 = (int)genericDialog.getNextNumber();
    sW2 = winS2;
    vecS2 = winS2 / 2;
    winS3 = (int)genericDialog.getNextNumber();
    sW3 = winS3;
    vecS3 = winS3 / 2;
    if (vecS3 == 0 || sW3 == 0 || winS3 == 0)
      this.nPass = 2; 
    if (vecS2 == 0 || sW2 == 0 || winS2 == 0)
      this.nPass = 1; 
    if (!checkParams()) {
      IJ.error("Incompatible PIV parameters");
      return false;
    } 
    if (genericDialog.wasCanceled())
      return false; 
    return true;
  }
  
  private boolean getParamsD() {
    GenericDialog genericDialog = new GenericDialog("Iterative PIV (Debug mode)");
    genericDialog.addMessage("(All sizes are in pixels)");
    genericDialog.addMessage("1st pass PIV parameters:");
    if (winS1 == 0)
      winS1 = 128; 
    genericDialog.addNumericField("PIV1 interrogation window size", winS1, 0);
    if (sW1 == 0)
      sW1 = 256; 
    genericDialog.addMessage("(If search window size=window size, conventional xcorr will be used)");
    genericDialog.addNumericField("SW1 :search window size", sW1, 0);
    if (vecS1 == 0)
      vecS1 = 64; 
    genericDialog.addNumericField("VS1 :Vector spacing", vecS1, 0);
    genericDialog.addMessage("-----------------------");
    genericDialog.addMessage("2nd pass PIV parameters: (set window size to zero to do only 1pass PIV)");
    if (winS2 == 0)
      winS2 = 64; 
    genericDialog.addNumericField("PIV2 interrogation window size", winS2, 0);
    if (sW2 == 0)
      sW2 = 128; 
    genericDialog.addNumericField("SW2 :search window size", sW2, 0);
    if (vecS2 == 0)
      vecS2 = 32; 
    genericDialog.addNumericField("VS2 :Vector spacing", vecS2, 0);
    genericDialog.addMessage("-----------------------");
    genericDialog.addMessage("3rd pass PIV parameters: (set window size to zero to do only 2pass PIV)");
    if (winS3 == 0)
      winS3 = 48; 
    genericDialog.addNumericField("PIV3 interrogation window size", winS3, 0);
    if (sW3 == 0)
      sW3 = 128; 
    genericDialog.addNumericField("SW3 :search window size", sW3, 0);
    if (vecS3 == 0)
      vecS3 = 16; 
    genericDialog.addNumericField("VS3 :Vector spacing", vecS3, 0);
    genericDialog.addMessage("-----------------------");
    if (cThr == 0.0D)
      cThr = 0.6D; 
    genericDialog.addNumericField("correlation threshold", cThr, 2);
    genericDialog.addCheckbox("Use advanced peak check? (empirical parameters)", false);
    genericDialog.addCheckbox("Disable all peak checking?", true);
    genericDialog.addCheckbox("Don't replace invalid vector by median?", true);
    genericDialog.addMessage("-----------------------");
    genericDialog.addNumericField("debug_X", -1.0D, 0);
    genericDialog.addNumericField("debug_Y", -1.0D, 0);
    genericDialog.showDialog();
    winS1 = (int)genericDialog.getNextNumber();
    sW1 = (int)genericDialog.getNextNumber();
    vecS1 = (int)genericDialog.getNextNumber();
    winS2 = (int)genericDialog.getNextNumber();
    sW2 = (int)genericDialog.getNextNumber();
    vecS2 = (int)genericDialog.getNextNumber();
    winS3 = (int)genericDialog.getNextNumber();
    sW3 = (int)genericDialog.getNextNumber();
    vecS3 = (int)genericDialog.getNextNumber();
    cThr = genericDialog.getNextNumber();
    this.chkPeakA = genericDialog.getNextBoolean();
    this.noChkPeak = genericDialog.getNextBoolean();
    this.pp = genericDialog.getNextBoolean();
    this.db = true;
    this.dbX = (int)genericDialog.getNextNumber();
    this.dbY = (int)genericDialog.getNextNumber();
    this.batch = false;
    if (vecS3 == 0 || sW3 == 0 || winS3 == 0)
      this.nPass = 2; 
    if (vecS2 == 0 || sW2 == 0 || winS2 == 0)
      this.nPass = 1; 
    if (!genericDialog.wasCanceled()) {
      if (!checkParams()) {
        IJ.error("Incompatible PIV parameters");
        return false;
      } 
    } else {
      return false;
    } 
    return true;
  }
  
  private boolean getParamsP() {
    String[] arrayOfString = { "Normalized median test and replace invalid by median", "Dynamic mean test and replace invalid by median", "Restore unprocessed PIV", "Accept this PIV and output" };
    GenericDialog genericDialog = new GenericDialog("PIV post-processing");
    genericDialog.addChoice("What to do?", arrayOfString, arrayOfString[this.action]);
    genericDialog.addMessage("-----------------------");
    genericDialog.addMessage("Normalized median test parameters:");
    genericDialog.addNumericField("noise for NMT", this.noiseNMT1, 2);
    genericDialog.addNumericField("Threshold for NMT", this.thrNMT1, 2);
    genericDialog.addMessage("-----------------------");
    genericDialog.addMessage("Dynamic mean test parameters:");
    genericDialog.addNumericField("C1 for DMT", this.c1DMT, 2);
    genericDialog.addNumericField("C2 for DMT", this.c2DMT, 2);
    genericDialog.addMessage("Dynamic threshold = C1+C2*(StdDev within the surrounding 3x3 vectors)");
    genericDialog.showDialog();
    this.action = genericDialog.getNextChoiceIndex();
    this.noiseNMT1 = genericDialog.getNextNumber();
    this.thrNMT1 = genericDialog.getNextNumber();
    this.c1DMT = genericDialog.getNextNumber();
    this.c2DMT = genericDialog.getNextNumber();
    if (genericDialog.wasCanceled())
      return false; 
    if (this.action == 3) {
      SaveDialog saveDialog = new SaveDialog("Save PIVdata", IJ.getDirectory("home"), "PIV_" + this.imp.getTitle(), ".txt");
      if (saveDialog.getDirectory() == null || saveDialog.getFileName() == null)
        return false; 
      dir = saveDialog.getDirectory();
      this.file = saveDialog.getFileName();
    } 
    return true;
  }
  
  private boolean checkParams() {
    if (winS1 == sW1) {
      this.xc = true;
    } else {
      this.xc = false;
    } 
    if (this.xc == true && !powerOf2Size(winS1)) {
      IJ.error("PIV using conventional cross-correlation need the window size to be power of 2");
      return false;
    } 
    if (winS1 > sW1) {
      IJ.error("Search window must be larger than interrogation window");
      return false;
    } 
    if (vecS1 > winS1) {
      IJ.error("PIV vector spacing must be smaller or equal to interrogation window size");
      return false;
    } 
    if (this.nPass != 1) {
      if (winS2 >= winS1) {
        IJ.error("Interrogation window of second pass should be smaller than that of first pass");
        return false;
      } 
      if (this.xc == true && !powerOf2Size(winS2)) {
        IJ.error("PIV using conventional cross-correlation need the window size to be power of 2");
        return false;
      } 
      if (winS2 > sW2) {
        IJ.error("Search window must be larger than interrogation window");
        return false;
      } 
      if (vecS2 > winS2) {
        IJ.error("PIV vector spacing must be smaller or equal to interrogation window size");
        return false;
      } 
    } 
    if (this.nPass == 3) {
      if (winS3 >= winS2) {
        IJ.error("Interrogation window of third pass should be smaller than that of second pass");
        return false;
      } 
      if (this.xc == true && !powerOf2Size(winS3)) {
        IJ.error("PIV using conventional cross-correlation need the window size to be power of 2");
        return false;
      } 
      if (winS3 > sW3) {
        IJ.error("Search window must be larger than interrogation window");
        return false;
      } 
      if (vecS3 > winS3) {
        IJ.error("PIV vector spacing must be smaller or equal to interrogation window size");
        return false;
      } 
    } 
    return true;
  }
  
  private boolean powerOf2Size(int paramInt) {
    int i = 2;
    for (; i < paramInt; i *= 2);
    return (i == paramInt);
  }
  
  double[][] doPIV(ImagePlus paramImagePlus, int paramInt1, int paramInt2, int paramInt3, double[][][] paramArrayOfdouble) {
    ImageStack imageStack = paramImagePlus.getStack();
    ImageProcessor imageProcessor1 = imageStack.getProcessor(1);
    ImageProcessor imageProcessor2 = imageStack.getProcessor(2);
    int[] arrayOfInt = new int[6];
    double[] arrayOfDouble1 = new double[2];
    double[] arrayOfDouble2 = new double[2];
    double d1 = 0.0D, d2 = 0.0D, d3 = 0.0D, d4 = 0.0D, d5 = 0.0D, d6 = 0.0D;
    double d7 = 0.0D, d8 = 0.0D, d9 = 0.0D;
    boolean bool1 = true;
    boolean bool2 = (paramArrayOfdouble.length == 1) ? true : false;
    int i = 0, j = 0;
    int k = paramInt1 / 4;
    byte b1 = 0;
    byte b2 = 0;
    boolean bool3 = false;
    this.nx = (int)Math.floor(((this.width - k * 2 - paramInt1) / paramInt2)) + 1;
    this.ny = (int)Math.floor(((this.height - k * 2 - paramInt1) / paramInt2)) + 1;
    double[][] arrayOfDouble = new double[this.nx * this.ny][16];
    if (this.db) {
      IJ.log("nx=" + this.nx);
      IJ.log("ny=" + this.ny);
    } 
    for (byte b3 = 0; b3 < this.ny; b3++) {
      for (byte b = 0; b < this.nx; b++) {
        FloatProcessor floatProcessor;
        int m, n;
        IJ.showProgress(b3 * this.nx + b, this.nx * this.ny);
        arrayOfDouble[b3 * this.nx + b][0] = (k + paramInt1 / 2 + paramInt2 * b);
        arrayOfDouble[b3 * this.nx + b][1] = (k + paramInt1 / 2 + paramInt2 * b3);
        if (!bool2) {
          double[] arrayOfDouble3 = lerpData(arrayOfDouble[b3 * this.nx + b][0], arrayOfDouble[b3 * this.nx + b][1], paramArrayOfdouble);
          d1 = arrayOfDouble3[0];
          d2 = arrayOfDouble3[1];
          d7 = Math.sqrt(arrayOfDouble3[0] * arrayOfDouble3[0] + arrayOfDouble3[1] * arrayOfDouble3[1]);
          arrayOfDouble[b3 * this.nx + b][12] = d1;
          arrayOfDouble[b3 * this.nx + b][13] = d2;
          arrayOfDouble[b3 * this.nx + b][14] = d7;
        } 
        if (this.xc) {
          i = (int)d1;
          j = (int)d2;
        } else {
          i = 0;
          j = 0;
        } 
        int i4 = k + b3 * paramInt2 - (paramInt3 - paramInt1) / 2 + j;
        int i3 = k + b * paramInt2 - (paramInt3 - paramInt1) / 2 + i;
        if (i4 < 0)
          i4 = 0; 
        if (i3 < 0)
          i3 = 0; 
        if (i4 + paramInt3 > this.height)
          i4 = this.height - paramInt3; 
        if (i3 + paramInt3 > this.width)
          i3 = this.width - paramInt3; 
        imageProcessor2.setRoi(i3, i4, paramInt3, paramInt3);
        int i1 = k + b * paramInt2;
        int i2 = k + b3 * paramInt2;
        if (i2 + paramInt1 > this.height - k)
          i2 = this.height - k - paramInt1; 
        if (i1 + paramInt1 > this.width - k)
          i1 = this.width - k - paramInt1; 
        imageProcessor1.setRoi(i1, i2, paramInt1, paramInt1);
        if (i2 == 0 || i1 == 0 || i2 == this.height - paramInt1 || i1 == this.width - paramInt1) {
          bool1 = false;
        } else if (paramInt3 - paramInt1 < 20) {
          bool1 = false;
        } else {
          bool1 = true;
        } 
        if (this.xc) {
          FHT fHT2 = new FHT(imageProcessor1.crop());
          FHT fHT3 = new FHT(imageProcessor2.crop());
          fHT2.transform();
          fHT3.transform();
          FHT fHT4 = fHT3.conjugateMultiply(fHT2);
          fHT4.inverseTransform();
          fHT4.swapQuadrants();
          FHT fHT1 = fHT4;
          m = paramInt1 / 2;
          n = paramInt1 / 2;
        } else {
          floatProcessor = cvMatchTemplate.doMatch(imageProcessor2.crop(), imageProcessor1.crop(), 5, false);
          m = k + b * paramInt2 - i3;
          n = k + b3 * paramInt2 - i4;
        } 
        if (k + paramInt1 / 2 + paramInt2 * b == this.dbX && k + paramInt1 / 2 + paramInt2 * b3 == this.dbY) {
          IJ.log("position: " + arrayOfDouble[b3 * this.nx + b][0] + "," + arrayOfDouble[b3 * this.nx + b][1]);
          IJ.log("edge:" + bool1);
          (new ImagePlus("Match result", (ImageProcessor)floatProcessor)).show();
          (new ImagePlus("tar_" + this.dbX + "," + this.dbY, imageProcessor2.crop())).show();
          (new ImagePlus("ref_" + this.dbX + "," + this.dbY, imageProcessor1.crop())).show();
        } 
        if ((floatProcessor.getStatistics()).stdDev == 0.0D) {
          arrayOfDouble[b3 * this.nx + b][2] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][3] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][4] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][5] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][6] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][7] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][8] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][9] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][10] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][11] = 0.0D;
          arrayOfDouble[b3 * this.nx + b][15] = 0.0D;
        } else {
          int[] arrayOfInt1 = { m, n };
          int[] arrayOfInt2 = { (int)arrayOfDouble[b3 * this.nx + b][0], (int)arrayOfDouble[b3 * this.nx + b][1] };
          ImageProcessor imageProcessor = floatProcessor.convertToShort(true);
          if (!bool2) {
            double d10, d11, arrayOfDouble3[] = new double[2];
            double[] arrayOfDouble4 = new double[2];
            if (this.db) {
              IJ.log("position: " + arrayOfInt2[0] + "," + arrayOfInt2[1]);
              IJ.log("dx0, dy0: " + d1 + "," + d2);
              IJ.log("xyOri: " + arrayOfInt1[0] + "," + arrayOfInt1[1]);
            } 
            if (this.xc) {
              arrayOfInt = findMaxA(imageProcessor, bool1);
              if (arrayOfInt[0] == -999 && arrayOfInt[1] == -999 && arrayOfInt[2] == -999 && arrayOfInt[3] == -999) {
                IJ.log("no maximum found at:");
                IJ.log("position: " + arrayOfInt2[0] + "," + arrayOfInt2[1]);
              } 
            } else {
              double[] arrayOfDouble5 = { d1, d2 };
              arrayOfInt = findMaxC((ImageProcessor)floatProcessor, bool1, arrayOfDouble5, arrayOfInt1, arrayOfInt2);
            } 
            int i5 = 1;
            if (arrayOfInt[0] == -999) {
              arrayOfDouble1[0] = -999.0D;
              arrayOfDouble1[1] = -999.0D;
              arrayOfDouble2[0] = -999.0D;
              arrayOfDouble2[1] = -999.0D;
              i5 = 2;
              d10 = 0.0D;
              d11 = 0.0D;
            } else {
              if (arrayOfInt[0] == arrayOfInt[2] && arrayOfInt[1] == arrayOfInt[3]) {
                arrayOfDouble1 = gaussianPeakFit(imageProcessor, arrayOfInt[0], arrayOfInt[1]);
                if (this.db) {
                  IJ.log("dxdyG[0]:" + arrayOfDouble1[0]);
                  IJ.log("dxdyG[1]:" + arrayOfDouble1[1]);
                } 
                arrayOfDouble2 = arrayOfDouble1;
                d3 = arrayOfDouble1[0] - m;
                d4 = arrayOfDouble1[1] - n;
                if (this.xc) {
                  d3 += i;
                  d4 += j;
                } 
                d5 = d3;
                d6 = d4;
                d8 = Math.sqrt(d3 * d3 + d4 * d4);
                d9 = d8;
                arrayOfDouble3 = checkVector(d3, d4, d8, d1, d2, d7);
                arrayOfDouble4 = arrayOfDouble3;
                d10 = floatProcessor.getf(arrayOfInt[0], arrayOfInt[1]);
                d11 = d10;
                if (this.noChkPeak) {
                  i5 = checkThr(d10);
                } else {
                  i5 = checkPeakB1(d10, d8, arrayOfDouble3);
                } 
              } else {
                arrayOfDouble1 = gaussianPeakFit(imageProcessor, arrayOfInt[0], arrayOfInt[1]);
                arrayOfDouble2 = gaussianPeakFit(imageProcessor, arrayOfInt[2], arrayOfInt[3]);
                if (this.db) {
                  IJ.log("dxdyG[0]:" + arrayOfDouble1[0]);
                  IJ.log("dxdyG[1]:" + arrayOfDouble1[1]);
                  IJ.log("dxdyG2[0]:" + arrayOfDouble2[0]);
                  IJ.log("dxdyG2[1]:" + arrayOfDouble2[1]);
                } 
                d3 = arrayOfDouble1[0] - m;
                d4 = arrayOfDouble1[1] - n;
                d5 = arrayOfDouble2[0] - m;
                d6 = arrayOfDouble2[1] - n;
                if (this.xc) {
                  d3 += i;
                  d4 += j;
                  d5 += i;
                  d6 += j;
                } 
                d8 = Math.sqrt(d3 * d3 + d4 * d4);
                d9 = Math.sqrt(d5 * d5 + d6 * d6);
                arrayOfDouble3 = checkVector(d3, d4, d8, d1, d2, d7);
                arrayOfDouble4 = checkVector(d5, d6, d9, d1, d2, d7);
                d10 = floatProcessor.getf(arrayOfInt[0], arrayOfInt[1]);
                d11 = floatProcessor.getf(arrayOfInt[2], arrayOfInt[3]);
                if (!this.xc)
                  if (this.noChkPeak) {
                    i5 = checkThr(d10);
                  } else if (this.chkPeakA) {
                    i5 = checkPeakA(d10, d11, d8, d9, arrayOfDouble3, arrayOfDouble4);
                  } else {
                    i5 = checkPeakB(d10, d11, d8, d9, arrayOfDouble3, arrayOfDouble4);
                  }  
              } 
              if (this.db) {
                IJ.log("dx1: " + d3);
                IJ.log("dy1: " + d4);
                IJ.log("dx2: " + d5);
                IJ.log("dy2: " + d6);
                IJ.log("dx0: " + d1);
                IJ.log("dy0: " + d2);
                IJ.log("mag0: " + d7);
              } 
              if (d10 < cThr)
                b2++; 
            } 
            if (this.db) {
              IJ.log("ang1: " + arrayOfDouble3[0]);
              IJ.log("ang2: " + arrayOfDouble4[0]);
              IJ.log("p1: " + d10);
              IJ.log("p2: " + d11);
              IJ.log("dL1: " + arrayOfDouble3[1]);
              IJ.log("dL2: " + arrayOfDouble4[1]);
              IJ.log("dL2-dL1: " + (Math.abs(arrayOfDouble4[1]) - Math.abs(arrayOfDouble3[1])));
              IJ.log("Choice:" + i5);
            } 
            switch (i5) {
              case 0:
                arrayOfDouble[b3 * this.nx + b][2] = d5;
                arrayOfDouble[b3 * this.nx + b][3] = d6;
                arrayOfDouble[b3 * this.nx + b][4] = d9;
                arrayOfDouble[b3 * this.nx + b][5] = arrayOfDouble4[0];
                arrayOfDouble[b3 * this.nx + b][6] = d11;
                arrayOfDouble[b3 * this.nx + b][7] = d3;
                arrayOfDouble[b3 * this.nx + b][8] = d4;
                arrayOfDouble[b3 * this.nx + b][9] = d8;
                arrayOfDouble[b3 * this.nx + b][10] = arrayOfDouble3[0];
                arrayOfDouble[b3 * this.nx + b][11] = d10;
                arrayOfDouble[b3 * this.nx + b][15] = 21.0D;
                break;
              case 1:
                arrayOfDouble[b3 * this.nx + b][2] = d3;
                arrayOfDouble[b3 * this.nx + b][3] = d4;
                arrayOfDouble[b3 * this.nx + b][4] = d8;
                arrayOfDouble[b3 * this.nx + b][5] = arrayOfDouble3[0];
                arrayOfDouble[b3 * this.nx + b][6] = d10;
                arrayOfDouble[b3 * this.nx + b][7] = d5;
                arrayOfDouble[b3 * this.nx + b][8] = d6;
                arrayOfDouble[b3 * this.nx + b][9] = d9;
                arrayOfDouble[b3 * this.nx + b][10] = arrayOfDouble4[0];
                arrayOfDouble[b3 * this.nx + b][11] = d11;
                break;
              case 2:
                arrayOfDouble[b3 * this.nx + b][2] = d3;
                arrayOfDouble[b3 * this.nx + b][3] = d4;
                arrayOfDouble[b3 * this.nx + b][4] = d8;
                arrayOfDouble[b3 * this.nx + b][5] = arrayOfDouble3[0];
                arrayOfDouble[b3 * this.nx + b][6] = d10;
                arrayOfDouble[b3 * this.nx + b][7] = d5;
                arrayOfDouble[b3 * this.nx + b][8] = d6;
                arrayOfDouble[b3 * this.nx + b][9] = d9;
                arrayOfDouble[b3 * this.nx + b][10] = arrayOfDouble4[0];
                arrayOfDouble[b3 * this.nx + b][11] = d11;
                arrayOfDouble[b3 * this.nx + b][15] = -1.0D;
                b1++;
                break;
            } 
          } else {
            if (this.db) {
              IJ.log("position: " + arrayOfInt2[0] + "," + arrayOfInt2[1]);
              IJ.log("dx0, dy0: " + d1 + "," + d2);
              IJ.log("xyOri: " + arrayOfInt1[0] + "," + arrayOfInt1[1]);
            } 
            arrayOfInt = findMaxA((ImageProcessor)floatProcessor, bool1);
            if ((((arrayOfInt[0] == -999) ? 1 : 0) & ((arrayOfInt[1] == -999) ? 1 : 0)) != 0) {
              arrayOfDouble1 = new double[] { -999.0D, -999.0D };
              arrayOfDouble2 = new double[] { -999.0D, -999.0D };
              d3 = 0.0D;
              d4 = 0.0D;
              d5 = 0.0D;
              d6 = 0.0D;
              d8 = 0.0D;
              d9 = 0.0D;
            } else {
              arrayOfDouble1 = gaussianPeakFit(imageProcessor, arrayOfInt[0], arrayOfInt[1]);
              arrayOfDouble2 = gaussianPeakFit(imageProcessor, arrayOfInt[2], arrayOfInt[3]);
              d3 = arrayOfDouble1[0] - m;
              d4 = arrayOfDouble1[1] - n;
              d5 = arrayOfDouble2[0] - m;
              d6 = arrayOfDouble2[1] - n;
              d8 = Math.sqrt(d3 * d3 + d4 * d4);
              d9 = Math.sqrt(d5 * d5 + d6 * d6);
            } 
            int i5 = floatProcessor.getWidth();
            int i6 = floatProcessor.getHeight();
            if (arrayOfInt[0] == -999 && arrayOfInt[1] == -999) {
              arrayOfDouble[b3 * this.nx + b][2] = d3;
              arrayOfDouble[b3 * this.nx + b][3] = d4;
              arrayOfDouble[b3 * this.nx + b][4] = d8;
              arrayOfDouble[b3 * this.nx + b][5] = 0.0D;
              arrayOfDouble[b3 * this.nx + b][6] = -1.0D;
              arrayOfDouble[b3 * this.nx + b][7] = d5;
              arrayOfDouble[b3 * this.nx + b][8] = d6;
              arrayOfDouble[b3 * this.nx + b][9] = d9;
              arrayOfDouble[b3 * this.nx + b][10] = 0.0D;
              arrayOfDouble[b3 * this.nx + b][11] = -1.0D;
              b1++;
            } else if (Math.abs(arrayOfDouble1[0] - arrayOfInt[0]) < (i5 / 2) && Math.abs(arrayOfDouble1[1] - arrayOfInt[1]) < (i6 / 2)) {
              arrayOfDouble[b3 * this.nx + b][2] = d3;
              arrayOfDouble[b3 * this.nx + b][3] = d4;
              arrayOfDouble[b3 * this.nx + b][4] = d8;
              arrayOfDouble[b3 * this.nx + b][5] = 0.0D;
              arrayOfDouble[b3 * this.nx + b][6] = floatProcessor.getf(arrayOfInt[0], arrayOfInt[1]);
              arrayOfDouble[b3 * this.nx + b][7] = d5;
              arrayOfDouble[b3 * this.nx + b][8] = d6;
              arrayOfDouble[b3 * this.nx + b][9] = d9;
              arrayOfDouble[b3 * this.nx + b][10] = 0.0D;
              arrayOfDouble[b3 * this.nx + b][11] = floatProcessor.getf(arrayOfInt[2], arrayOfInt[3]);
            } else {
              arrayOfDouble[b3 * this.nx + b][2] = d3;
              arrayOfDouble[b3 * this.nx + b][3] = d4;
              arrayOfDouble[b3 * this.nx + b][4] = d8;
              arrayOfDouble[b3 * this.nx + b][5] = 0.0D;
              arrayOfDouble[b3 * this.nx + b][6] = floatProcessor.getf(arrayOfInt[0], arrayOfInt[1]);
              arrayOfDouble[b3 * this.nx + b][7] = d5;
              arrayOfDouble[b3 * this.nx + b][8] = d6;
              arrayOfDouble[b3 * this.nx + b][9] = d9;
              arrayOfDouble[b3 * this.nx + b][10] = 0.0D;
              arrayOfDouble[b3 * this.nx + b][11] = floatProcessor.getf(arrayOfInt[2], arrayOfInt[3]);
              arrayOfDouble[b3 * this.nx + b][15] = -1.0D;
              b1++;
            } 
          } 
        } 
      } 
    } 
    IJ.log("#interpolated vector / #total vector = " + b1 + "/" + (this.nx * this.ny));
    IJ.log("#vector with corr. value lower than threshold / #total vector = " + b2 + "/" + (this.nx * this.ny));
    return arrayOfDouble;
  }
  
  private int checkPeakA(double paramDouble1, double paramDouble2, double paramDouble3, double paramDouble4, double[] paramArrayOfdouble1, double[] paramArrayOfdouble2) {
    byte b = 1;
    double d1 = paramDouble1 / paramDouble2;
    double d2 = paramDouble3 - paramArrayOfdouble1[1];
    if (d1 > 1.5D && paramDouble1 > this.meanR + 2.0D * this.sdR) {
      b = 1;
    } else if (paramDouble1 > this.meanR + 3.0D * this.sdR && paramArrayOfdouble1[0] < 20.0D && paramArrayOfdouble2[0] < 20.0D) {
      b = 1;
    } else if (paramDouble1 > this.meanR + 2.0D * this.sdR && paramArrayOfdouble1[0] < 20.0D && paramArrayOfdouble2[0] < 20.0D && Math.abs(paramArrayOfdouble1[0] - paramArrayOfdouble2[0]) < 5.0D) {
      if (paramDouble3 >= d2 * 0.8D && paramDouble3 / d2 < 3.0D) {
        b = 1;
      } else if (paramDouble4 >= d2 * 0.8D && paramDouble4 / d2 < 3.0D) {
        b = 0;
      } else {
        b = 2;
      } 
    } else if (paramDouble1 < cThr) {
      b = 2;
    } else if (paramDouble1 - paramDouble2 < 0.1D || paramDouble1 / paramDouble2 < 1.2D) {
      if (paramArrayOfdouble1[0] - paramArrayOfdouble2[0] > 90.0D && paramDouble4 / d2 < 3.0D && paramDouble4 / d2 > 0.33D) {
        b = 0;
      } else if (paramArrayOfdouble1[0] - paramArrayOfdouble2[0] > 30.0D && paramDouble4 / d2 < 1.5D && paramDouble4 / d2 > 0.67D) {
        b = 0;
      } 
    } else if (paramArrayOfdouble1[0] - paramArrayOfdouble2[0] > 50.0D && paramArrayOfdouble2[0] < 6.0D && paramDouble4 / d2 < 3.0D && paramDouble4 / d2 > 0.33D) {
      b = 0;
    } 
    return b;
  }
  
  private int checkPeakB(double paramDouble1, double paramDouble2, double paramDouble3, double paramDouble4, double[] paramArrayOfdouble1, double[] paramArrayOfdouble2) {
    byte b = 1;
    double d = paramDouble3 - paramArrayOfdouble1[1];
    if (paramDouble1 < cThr) {
      b = 2;
    } else if (paramArrayOfdouble1[0] > 30.0D && d > 1.0D) {
      if (paramArrayOfdouble2[0] < 5.0D && paramDouble4 / d < 2.0D && paramDouble4 / d > 0.5D) {
        b = 0;
      } else {
        b = 2;
      } 
    } else if ((paramDouble3 / d > 2.0D || paramDouble3 / d < 0.5D) && d > 1.0D) {
      if (paramArrayOfdouble2[0] < 5.0D && paramDouble4 / d < 2.0D && paramDouble4 / d > 0.5D) {
        b = 0;
      } else {
        b = 2;
      } 
    } 
    return b;
  }
  
  private int checkPeakB1(double paramDouble1, double paramDouble2, double[] paramArrayOfdouble) {
    byte b = 1;
    double d = paramDouble2 - paramArrayOfdouble[1];
    if (paramDouble2 / d > 5.0D && paramDouble2 > 1.0D && paramDouble1 < cThr)
      b = 2; 
    if (paramArrayOfdouble[0] > 60.0D && paramDouble2 > 1.0D && paramDouble1 < cThr)
      b = 2; 
    return b;
  }
  
  private int checkThr(double paramDouble) {
    byte b = 1;
    if (paramDouble < cThr) {
      b = 2;
    } else {
      b = 1;
    } 
    return b;
  }
  
  private void logPIV(double[][] paramArrayOfdouble) {
    for (byte b = 0; b < paramArrayOfdouble.length; b++)
      IJ.log("\t" + paramArrayOfdouble[b][0] + "\t" + paramArrayOfdouble[b][1] + "\t" + paramArrayOfdouble[b][2] + "\t" + paramArrayOfdouble[b][3] + "\t" + paramArrayOfdouble[b][4] + "\t" + paramArrayOfdouble[b][5] + "\t" + paramArrayOfdouble[b][6] + "\t" + paramArrayOfdouble[b][7] + "\t" + paramArrayOfdouble[b][8] + "\t" + paramArrayOfdouble[b][9] + "\t" + paramArrayOfdouble[b][10] + "\t" + paramArrayOfdouble[b][11] + "\t" + paramArrayOfdouble[b][12] + "\t" + paramArrayOfdouble[b][13] + "\t" + paramArrayOfdouble[b][14] + "\t" + paramArrayOfdouble[b][15]); 
  }
  
  private int[] findMaxA(ImageProcessor paramImageProcessor, boolean paramBoolean) {
    ResultsTable resultsTable = ResultsTable.getResultsTable();
    resultsTable.reset();
    MaximumFinder maximumFinder = new MaximumFinder();
    double d1 = (paramImageProcessor.getStatistics()).stdDev;
    double d2 = (paramImageProcessor.getStatistics()).mean;
    double d3 = d1;
    byte b = 0;
    while ((resultsTable.getCounter() < 2 && b < 5) || (resultsTable.getCounter() == 0 && b < 10)) {
      resultsTable.reset();
      maximumFinder.findMaxima(paramImageProcessor, d3, -808080.0D, 4, paramBoolean, false);
      d3 /= 2.0D;
      b++;
    } 
    int[] arrayOfInt = new int[4];
    if (resultsTable.getCounter() == 1) {
      arrayOfInt[0] = (int)resultsTable.getValue("X", 0);
      arrayOfInt[1] = (int)resultsTable.getValue("Y", 0);
      arrayOfInt[2] = (int)resultsTable.getValue("X", 0);
      arrayOfInt[3] = (int)resultsTable.getValue("Y", 0);
    } else if (resultsTable.getCounter() > 1) {
      arrayOfInt[0] = (int)resultsTable.getValue("X", 0);
      arrayOfInt[1] = (int)resultsTable.getValue("Y", 0);
      arrayOfInt[2] = (int)resultsTable.getValue("X", 1);
      arrayOfInt[3] = (int)resultsTable.getValue("Y", 1);
    } else {
      arrayOfInt[0] = -999;
      arrayOfInt[1] = -999;
      arrayOfInt[2] = -999;
      arrayOfInt[3] = -999;
    } 
    return arrayOfInt;
  }
  
  private int[] findMaxC(ImageProcessor paramImageProcessor, boolean paramBoolean, double[] paramArrayOfdouble, int[] paramArrayOfint1, int[] paramArrayOfint2) {
    double d7, d8;
    ResultsTable resultsTable = ResultsTable.getResultsTable();
    resultsTable.reset();
    MaximumFinder maximumFinder = new MaximumFinder();
    double d1 = (paramImageProcessor.getStatistics()).stdDev;
    double d2 = (paramImageProcessor.getStatistics()).mean;
    this.sdR = d1;
    this.meanR = d2;
    double d3 = (paramImageProcessor.getStatistics()).max;
    double d4 = (paramImageProcessor.getStatistics()).kurtosis;
    double d5 = (paramImageProcessor.getStatistics()).skewness;
    double d6 = d1;
    int[] arrayOfInt = new int[4];
    int k = paramImageProcessor.getWidth() / 2;
    if (d4 > 6.0D || (d4 > 3.0D && d5 > -0.05D) || (d3 - d2) / d1 > 4.0D) {
      maximumFinder.findMaxima(paramImageProcessor, d1 / 2.0D, d2 + 2.0D * d1, 4, paramBoolean, false);
      if (resultsTable.getCounter() > 1) {
        arrayOfInt[0] = (int)resultsTable.getValue("X", 0);
        arrayOfInt[1] = (int)resultsTable.getValue("Y", 0);
        arrayOfInt[2] = (int)resultsTable.getValue("X", 1);
        arrayOfInt[3] = (int)resultsTable.getValue("Y", 1);
        d7 = paramImageProcessor.getf(arrayOfInt[0], arrayOfInt[1]);
        d8 = paramImageProcessor.getf(arrayOfInt[2], arrayOfInt[3]);
        double d10 = d7 - d8;
        double d11 = d7 / d8;
        double d12 = d10 / d1;
        if (d12 > 2.0D || d11 > 2.0D) {
          if (this.db) {
            IJ.log("Z1: significant peak found. curPos = " + paramArrayOfint2[0] + "," + paramArrayOfint2[1]);
            IJ.log("peakH: " + d12);
            IJ.log("p/p: " + d11);
            IJ.log("X: " + arrayOfInt[0]);
            IJ.log("Y: " + arrayOfInt[1]);
          } 
          arrayOfInt[2] = arrayOfInt[0];
          arrayOfInt[3] = arrayOfInt[1];
          return arrayOfInt;
        } 
      } else if (resultsTable.getCounter() == 1) {
        if (this.db) {
          IJ.log("Z3: significant peak found. curPos = " + paramArrayOfint2[0] + "," + paramArrayOfint2[1]);
          IJ.log("kurt= " + d4);
          IJ.log("skew= " + d5);
          IJ.log("X: " + resultsTable.getValue("X", 0));
          IJ.log("Y: " + resultsTable.getValue("Y", 0));
        } 
        arrayOfInt[0] = (int)resultsTable.getValue("X", 0);
        arrayOfInt[1] = (int)resultsTable.getValue("Y", 0);
        arrayOfInt[2] = (int)resultsTable.getValue("X", 0);
        arrayOfInt[3] = (int)resultsTable.getValue("Y", 0);
        return arrayOfInt;
      } 
    } 
    int i = (int)((paramArrayOfint1[0] - Math.round((k / 2))) + Math.round(paramArrayOfdouble[0]));
    int j = (int)((paramArrayOfint1[1] - Math.round((k / 2))) + Math.round(paramArrayOfdouble[1]));
    byte b = 4;
    if (paramArrayOfdouble[0] < 0.0D) {
      d7 = (paramArrayOfdouble[0] * -1.0D < (k / b)) ? paramArrayOfdouble[0] : (-k / b);
    } else {
      d7 = (paramArrayOfdouble[0] < (k / b)) ? paramArrayOfdouble[0] : (k / b);
    } 
    if (paramArrayOfdouble[1] < 0.0D) {
      d8 = (paramArrayOfdouble[1] * -1.0D < (k / b)) ? paramArrayOfdouble[1] : (-k / b);
    } else {
      d8 = (paramArrayOfdouble[1] < (k / b)) ? paramArrayOfdouble[1] : (k / b);
    } 
    i = (int)(i + d7);
    j = (int)(j + d8);
    paramImageProcessor.setRoi(i, j, k, k);
    double d9 = (paramImageProcessor.getStatistics()).stdDev;
    resultsTable.reset();
    maximumFinder.findMaxima(paramImageProcessor, d1 / 20.0D, d2, 4, paramBoolean, false);
    if (resultsTable.getCounter() == 0) {
      paramImageProcessor.resetRoi();
      resultsTable.reset();
      maximumFinder.findMaxima(paramImageProcessor, d1 / 10.0D, d2 + 2.0D * d1, 4, paramBoolean, false);
      if (resultsTable.getCounter() == 0) {
        if (this.db) {
          IJ.log("no significant peak found. curPos = " + paramArrayOfint2[0] + "," + paramArrayOfint2[1]);
          IJ.log("limitX: " + i);
          IJ.log("limitY: " + j);
          IJ.log("tolerance: " + (d1 / 10.0D));
          IJ.log("threshold: " + (d2 + 2.0D * d1));
        } 
        arrayOfInt[0] = -999;
        arrayOfInt[1] = -999;
        arrayOfInt[2] = -999;
        arrayOfInt[3] = -999;
      } else {
        if (this.db) {
          IJ.log("A1: one peak found in the whole map. curPos = " + paramArrayOfint2[0] + "," + paramArrayOfint2[1]);
          IJ.log("limitX: " + i);
          IJ.log("limitY: " + j);
          IJ.log("tolerance: " + (d1 / 10.0D));
        } 
        arrayOfInt[0] = (int)resultsTable.getValue("X", 0);
        arrayOfInt[1] = (int)resultsTable.getValue("Y", 0);
        arrayOfInt[2] = (int)resultsTable.getValue("X", 0);
        arrayOfInt[3] = (int)resultsTable.getValue("Y", 0);
      } 
    } else {
      if (this.db) {
        IJ.log("B1: peaks found in the preshift window. curPos = " + paramArrayOfint2[0] + "," + paramArrayOfint2[1]);
        IJ.log("limitX: " + i);
        IJ.log("limitY: " + j);
        IJ.log("tolerance: " + (d1 / 20.0D));
        for (byte b1 = 0; b1 < resultsTable.getCounter() && 
          b1 < 2; b1++) {
          IJ.log("X: " + resultsTable.getValue("X", b1));
          IJ.log("Y: " + resultsTable.getValue("Y", b1));
        } 
      } 
      arrayOfInt[0] = (int)resultsTable.getValue("X", 0);
      arrayOfInt[1] = (int)resultsTable.getValue("Y", 0);
      if (resultsTable.getCounter() > 1) {
        arrayOfInt[2] = (int)resultsTable.getValue("X", 1);
        arrayOfInt[3] = (int)resultsTable.getValue("Y", 1);
      } else {
        arrayOfInt[2] = arrayOfInt[0];
        arrayOfInt[3] = arrayOfInt[1];
      } 
      double d10 = paramImageProcessor.getf(arrayOfInt[0], arrayOfInt[1]);
      double d11 = paramImageProcessor.getf(arrayOfInt[2], arrayOfInt[3]);
      double d12 = d10 - d11;
      double d13 = d10 / d11;
      double d14 = d12 / d1;
      double d15 = d12 / d9;
      if (this.db) {
        IJ.log("peakH= " + d14);
        IJ.log("peakH2= " + d15);
        IJ.log("pRatio= " + d13);
      } 
      if (d10 > 0.5D && ((d14 + d15 > 2.0D && d13 > 2.0D) || d14 + d15 > 3.0D)) {
        if (this.db)
          IJ.log("peak1 muck significant than peak2"); 
        arrayOfInt[2] = arrayOfInt[0];
        arrayOfInt[3] = arrayOfInt[1];
      } else if ((d10 <= 0.5D && d13 > 2.0D) || d14 + d15 > 4.0D) {
        if (this.db) {
          IJ.log("peak1 2 times higher than peak2");
          IJ.log("peak1: " + d10);
        } 
        arrayOfInt[2] = arrayOfInt[0];
        arrayOfInt[3] = arrayOfInt[1];
      } 
    } 
    return arrayOfInt;
  }
  
  private double[] gaussianPeakFit(ImageProcessor paramImageProcessor, int paramInt1, int paramInt2) {
    double[] arrayOfDouble = new double[2];
    double d1 = 0.0D, d2 = 0.0D, d3 = 0.0D;
    if (paramInt1 == 0 || paramInt1 == paramImageProcessor
      .getWidth() - 1 || paramInt2 == 0 || paramInt2 == paramImageProcessor
      
      .getHeight() - 1) {
      arrayOfDouble[0] = paramInt1;
      arrayOfDouble[1] = paramInt2;
    } else {
      if (paramImageProcessor.getPixel(paramInt1 - 1, paramInt2) != 0)
        d1 = Math.log(paramImageProcessor.getPixel(paramInt1 - 1, paramInt2)); 
      if (paramImageProcessor.getPixel(paramInt1, paramInt2) != 0)
        d2 = Math.log(paramImageProcessor.getPixel(paramInt1, paramInt2)); 
      if (paramImageProcessor.getPixel(paramInt1 + 1, paramInt2) != 0)
        d3 = Math.log(paramImageProcessor.getPixel(paramInt1 + 1, paramInt2)); 
      arrayOfDouble[0] = paramInt1 + (d1 - d3) / (2.0D * d1 - 4.0D * d2 + 2.0D * d3);
      if (Double.isNaN(arrayOfDouble[0]) || Double.isInfinite(arrayOfDouble[0]))
        arrayOfDouble[0] = paramInt1; 
      if (paramImageProcessor.getPixel(paramInt1, paramInt2 - 1) != 0)
        d1 = Math.log(paramImageProcessor.getPixel(paramInt1, paramInt2 - 1)); 
      if (paramImageProcessor.getPixel(paramInt1, paramInt2 + 1) != 0)
        d3 = Math.log(paramImageProcessor.getPixel(paramInt1, paramInt2 + 1)); 
      arrayOfDouble[1] = paramInt2 + (d1 - d3) / (2.0D * d1 - 4.0D * d2 + 2.0D * d3);
      if (Double.isNaN(arrayOfDouble[1]) || Double.isInfinite(arrayOfDouble[1]))
        arrayOfDouble[1] = paramInt2; 
    } 
    return arrayOfDouble;
  }
  
  double[][] normalizedMedianTest(double[][] paramArrayOfdouble, double paramDouble1, double paramDouble2) {
    byte b1 = 15;
    for (byte b2 = 0; b2 < paramArrayOfdouble.length; b2++) {
      for (byte b = 2; b < 4; b++) {
        double[] arrayOfDouble = getNeighbours(paramArrayOfdouble, b2, b, b1);
        if (arrayOfDouble != null) {
          double d = Math.abs(paramArrayOfdouble[b2][b] - getMedian(arrayOfDouble)) / (getMedian(getResidualsOfMedian(arrayOfDouble)) + paramDouble1);
          if (d > paramDouble2)
            paramArrayOfdouble[b2][b1] = -1.0D; 
        } else {
          paramArrayOfdouble[b2][b1] = -1.0D;
        } 
      } 
    } 
    return paramArrayOfdouble;
  }
  
  double[][] dynamicMeanTest(double[][] paramArrayOfdouble, double paramDouble1, double paramDouble2) {
    byte b1 = 15;
    for (byte b2 = 0; b2 < paramArrayOfdouble.length; b2++) {
      for (byte b = 2; b < 4; b++) {
        double[] arrayOfDouble = getNeighbours(paramArrayOfdouble, b2, b, b1);
        if (arrayOfDouble != null) {
          double d2 = getMean(arrayOfDouble);
          double d3 = calcStd(arrayOfDouble, d2);
          double d1 = paramDouble1 + paramDouble2 * d3;
          if (Math.abs(paramArrayOfdouble[b2][b] - d2) > d1)
            paramArrayOfdouble[b2][b1] = -1.0D; 
        } else {
          paramArrayOfdouble[b2][b1] = -1.0D;
        } 
      } 
    } 
    return paramArrayOfdouble;
  }
  
  double[] getNeighbours(double[][] paramArrayOfdouble, int paramInt1, int paramInt2, int paramInt3) {
    double[] arrayOfDouble = new double[9];
    int i = paramInt1 / this.nx;
    int j = paramInt1 - i * this.nx;
    byte b = 0;
    int k = 0;
    k = paramInt1 - this.nx - 1;
    if (i - 1 >= 0 && j - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -1.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 - this.nx;
    if (i - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -1.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 - this.nx + 1;
    if (i - 1 >= 0 && j + 1 < this.nx && paramArrayOfdouble[k][paramInt3] != -1.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 - 1;
    if (j - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -1.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 + 1;
    if (j + 1 < this.nx && paramArrayOfdouble[k][paramInt3] != -1.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 + this.nx - 1;
    if (i + 1 < this.ny && j - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -1.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 + this.nx;
    if (i + 1 < this.ny && paramArrayOfdouble[k][paramInt3] != -1.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 + this.nx + 1;
    if (i + 1 < this.ny && j + 1 < this.nx && paramArrayOfdouble[k][paramInt3] != -1.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    if (b > 0) {
      double[] arrayOfDouble1 = new double[b];
      System.arraycopy(arrayOfDouble, 0, arrayOfDouble1, 0, b);
      return arrayOfDouble1;
    } 
    return null;
  }
  
  double[] getNeighbours2(double[][] paramArrayOfdouble, int paramInt1, int paramInt2, int paramInt3) {
    double[] arrayOfDouble = new double[9];
    int i = paramInt1 / this.nx;
    int j = paramInt1 - i * this.nx;
    byte b = 0;
    int k = 0;
    k = paramInt1 - this.nx - 1;
    if (i - 1 >= 0 && j - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -2.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 - this.nx;
    if (i - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -2.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 - this.nx + 1;
    if (i - 1 >= 0 && j + 1 < this.nx && paramArrayOfdouble[k][paramInt3] != -2.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 - 1;
    if (j - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -2.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 + 1;
    if (j + 1 < this.nx && paramArrayOfdouble[k][paramInt3] != -2.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 + this.nx - 1;
    if (i + 1 < this.ny && j - 1 >= 0 && paramArrayOfdouble[k][paramInt3] != -2.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 + this.nx;
    if (i + 1 < this.ny && paramArrayOfdouble[k][paramInt3] != -2.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    k = paramInt1 + this.nx + 1;
    if (i + 1 < this.ny && j + 1 < this.nx && paramArrayOfdouble[k][paramInt3] != -2.0D) {
      b++;
      arrayOfDouble[b - 1] = paramArrayOfdouble[k][paramInt2];
    } 
    if (b > 0) {
      double[] arrayOfDouble1 = new double[b];
      System.arraycopy(arrayOfDouble, 0, arrayOfDouble1, 0, b);
      return arrayOfDouble1;
    } 
    return null;
  }
  
  double getMedian(double[] paramArrayOfdouble) {
    Arrays.sort(paramArrayOfdouble);
    int i = paramArrayOfdouble.length / 2;
    if (paramArrayOfdouble.length % 2 > 0)
      return paramArrayOfdouble[i]; 
    return (paramArrayOfdouble[i] + paramArrayOfdouble[i - 1]) / 2.0D;
  }
  
  double getMean(double[] paramArrayOfdouble) {
    double d = 0.0D;
    for (byte b = 0; b < paramArrayOfdouble.length; b++)
      d += paramArrayOfdouble[b]; 
    return d / paramArrayOfdouble.length;
  }
  
  double calcStd(double[] paramArrayOfdouble, double paramDouble) {
    double d = 0.0D;
    for (byte b = 0; b < paramArrayOfdouble.length; b++)
      d += (paramArrayOfdouble[b] - paramDouble) * (paramArrayOfdouble[b] - paramDouble); 
    return d / paramArrayOfdouble.length;
  }
  
  double[] getResidualsOfMedian(double[] paramArrayOfdouble) {
    double d = getMedian(paramArrayOfdouble);
    for (byte b = 0; b < paramArrayOfdouble.length; b++)
      paramArrayOfdouble[b] = Math.abs(paramArrayOfdouble[b] - d); 
    return paramArrayOfdouble;
  }
  
  double[][] replaceByMedian(double[][] paramArrayOfdouble) {
    byte b1 = 15;
    double[][] arrayOfDouble = new double[paramArrayOfdouble.length][(paramArrayOfdouble[0]).length];
    for (byte b2 = 0; b2 < arrayOfDouble.length; b2++)
      System.arraycopy(paramArrayOfdouble[b2], 0, arrayOfDouble[b2], 0, (paramArrayOfdouble[b2]).length); 
    for (byte b3 = 0; b3 < arrayOfDouble.length; b3++) {
      if (arrayOfDouble[b3][b1] == -1.0D) {
        for (byte b = 2; b <= 3; b++) {
          double[] arrayOfDouble1 = getNeighbours(paramArrayOfdouble, b3, b, b1);
          if (arrayOfDouble1 != null) {
            arrayOfDouble[b3][b] = getMedian(arrayOfDouble1);
            arrayOfDouble[b3][b1] = 999.0D;
          } else {
            arrayOfDouble[b3][b] = 0.0D;
            arrayOfDouble[b3][b1] = -2.0D;
          } 
        } 
        if (arrayOfDouble[b3][b1] != -2.0D)
          arrayOfDouble[b3][4] = Math.sqrt(arrayOfDouble[b3][2] * arrayOfDouble[b3][2] + arrayOfDouble[b3][3] * arrayOfDouble[b3][3]); 
      } 
    } 
    return arrayOfDouble;
  }
  
  double[][] replaceByMedian2(double[][] paramArrayOfdouble) {
    byte b1 = 15;
    double[][] arrayOfDouble = new double[paramArrayOfdouble.length][(paramArrayOfdouble[0]).length];
    for (byte b2 = 0; b2 < arrayOfDouble.length; b2++)
      System.arraycopy(paramArrayOfdouble[b2], 0, arrayOfDouble[b2], 0, (paramArrayOfdouble[b2]).length); 
    for (byte b3 = 0; b3 < arrayOfDouble.length; b3++) {
      if (arrayOfDouble[b3][b1] == -2.0D) {
        for (byte b = 2; b <= 3; b++) {
          double[] arrayOfDouble1 = getNeighbours2(paramArrayOfdouble, b3, b, b1);
          if (arrayOfDouble1 != null) {
            arrayOfDouble[b3][b] = getMedian(arrayOfDouble1);
            arrayOfDouble[b3][b1] = 9999.0D;
          } else {
            arrayOfDouble[b3][b] = 0.0D;
            arrayOfDouble[b3][b1] = -22.0D;
          } 
        } 
        if (arrayOfDouble[b3][b1] != -22.0D)
          arrayOfDouble[b3][4] = Math.sqrt(arrayOfDouble[b3][2] * arrayOfDouble[b3][2] + arrayOfDouble[b3][3] * arrayOfDouble[b3][3]); 
      } 
    } 
    return arrayOfDouble;
  }
  
  public static double[] checkVector(double paramDouble1, double paramDouble2, double paramDouble3, double paramDouble4, double paramDouble5, double paramDouble6) {
    double[] arrayOfDouble = new double[2];
    double d1 = paramDouble1 * paramDouble4 + paramDouble2 * paramDouble5;
    double d2 = paramDouble3;
    double d3 = paramDouble6;
    double d4 = Math.acos(d1 / d2 * d3);
    if (Double.isNaN(d4) || Double.isInfinite(d4)) {
      arrayOfDouble[0] = 0.0D;
    } else {
      arrayOfDouble[0] = d4 * 180.0D / Math.PI;
    } 
    arrayOfDouble[1] = d2 - d3;
    return arrayOfDouble;
  }
  
  private StringBuffer generatePIVToPrint(double[][] paramArrayOfdouble) {
    NumberFormat numberFormat = NumberFormat.getInstance(Locale.US);
    if (numberFormat instanceof DecimalFormat)
      ((DecimalFormat)numberFormat).applyPattern("###.##;-###.##"); 
    numberFormat.setMaximumFractionDigits(12);
    numberFormat.setMinimumFractionDigits(0);
    StringBuffer stringBuffer = new StringBuffer();
    for (byte b = 0; b < paramArrayOfdouble.length; b++) {
      for (byte b1 = 0; b1 < (paramArrayOfdouble[0]).length; b1++) {
        stringBuffer.append(numberFormat.format(paramArrayOfdouble[b][b1]));
        stringBuffer.append(" ");
      } 
      stringBuffer.append("\n");
    } 
    return stringBuffer;
  }
  
  public boolean write2File(String paramString1, String paramString2, String paramString3) {
    PrintWriter printWriter = null;
    try {
      FileOutputStream fileOutputStream = new FileOutputStream(paramString1 + paramString2);
      BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
      printWriter = new PrintWriter(bufferedOutputStream);
      printWriter.print(paramString3);
      printWriter.close();
      return true;
    } catch (IOException iOException) {
      IJ.error("" + iOException);
      return false;
    } 
  }
  
  double[] lerpData(double paramDouble1, double paramDouble2, double[][][] paramArrayOfdouble) {
    double[] arrayOfDouble = new double[(paramArrayOfdouble[0][0]).length - 2];
    GridPointData gridPointData = new GridPointData(paramArrayOfdouble.length, (paramArrayOfdouble[0]).length, (paramArrayOfdouble[0][0]).length);
    gridPointData.setData(paramArrayOfdouble);
    int[] arrayOfInt = { 0, 1, 2 };
    arrayOfDouble = gridPointData.interpolate(paramDouble1, paramDouble2, arrayOfInt, arrayOfDouble);
    return arrayOfDouble;
  }
}
