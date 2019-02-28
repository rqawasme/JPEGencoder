import cv2
import numpy as np
from math import pi,cos
from tkinter import *
import tkinter.messagebox




####
#Convert RGB to YUV
#This also does a 4:2:0 chroma(U and V)subsampling
####
def RGB2YUV(imgRGB,height,width,channels):

    matYUV = np.array([[0.299, 0.587, 0.114],
                       [-0.14713, -0.28886, 0.436],
                       [0.615,-0.51499,-0.10001]])

    YUVarr = np.empty(shape=(height,width,channels), dtype=np.float)

    for i in range(height):
        for j in range(width):
            YUVarr[i][j] = np.matmul(matYUV, imgRGB[i][j])
    
    return YUVarr
    
###
#Convert YUV to RGB
###
def YUV2RGB(imgYUV,height,width,channels):

    matYUV = np.array([[1, 0, 1.13983],
                       [1, -0.39465, -0.58060],
                       [1,2.03211,0]])

    RGBarr = np.empty(shape=(height,width,channels), dtype=np.float)

    for i in range(height):
        for j in range(width):
            RGBarr[i][j] = np.matmul(matYUV, imgYUV[i][j])
            
    return RGBarr

            
            
####
#4:2:0 chroma subsampling for U and V channels
#   for each 2x2 pixel, replace all 4 pixel with average of all 4 pixels
#
####
def subsampleUV(UVarr,height,width,channels):            

    Subsampled = np.empty(shape=(height,width,channels), dtype = np.float)

    #keep the Y channel the same
    Subsampled[:,:,0] = UVarr[:,:,0]

    for i in range(0,height,2):
        for j in range(0,width,2):
            if i%2 == 0 and j%2 ==0:
               
                #subsample U channel
                average_u = (UVarr[i][j][1] +UVarr[i+1][j][1] +UVarr[i][j+1][1] +UVarr[i+1][j+1][1])/4
                Subsampled[i][j][1] = average_u #UVarr[i][j][1]
                Subsampled[i+1][j][1] = average_u #UVarr[i+1][j][1]
                Subsampled[i][j+1][1] = average_u #UVarr[i][j+1][1]
                Subsampled[i+1][j+1][1] = average_u #UVarr[i+1][j+1][1]
                
                #subsample V channel
                average_v = (UVarr[i][j][2] +UVarr[i+1][j][2] +UVarr[i][j+1][2] +UVarr[i+1][j+1][2])/4
                Subsampled[i][j][2] = average_v #UVarr[i][j][2]
                Subsampled[i+1][j][2] = average_v #UVarr[i+1][j][2]
                Subsampled[i][j+1][2] = average_v #UVarr[i][j+1][2]
                Subsampled[i+1][j+1][2] = average_v #UVarr[i+1][j+1][2]
                
    
    #print("Equal", np.array_equal(UVarr, Subsampled))
    return Subsampled
    

###
#apply DCT to image I
# apply seperately for each 8x8 block B for each channels of I
# B' = T*B*transpose(T)
# T is matrix of DCT basis'
###
def DCT(YUVarr,height,width,channels,T): 
    #subtract 128 from YUVarr to comply to jpeg standard
    YUVarr -= 128

    #split YUVarr into 8x8 blocks then aply DCT
    for i in range(0,height,8):
        for j in range(0,width,8):
            for k in range(channels):
                #apply DCT to each 8x8 block for all channels
                if i%8 == 0 and j%8 ==0:

                    mul1 = np.matmul(YUVarr[i:i+8,j:j+8,k],np.transpose(T))
                    YUVarr[i:i+8,j:j+8,k] = np.matmul(T, mul1)
    return YUVarr
    
###
# Inverse DCT of image I
# apply seperately for each 8x8 block B for each channels of I
# B' = transpose(T)*B*T
# T is matrix of DCT basis'
###
def InverseDCT(QUANTIZEDarr,height,width,channels,T):
    
    #split QUANTIZEDarr into 8x8 blocks then aply DCT
    for i in range(0,height,8):
        for j in range(0,width,8):
            for k in range(channels):
                #apply DCT to each 8x8 block for all channels
                if i%8 == 0 and j%8 ==0:
                        mul1 = np.matmul(QUANTIZEDarr[i:i+8,j:j+8,k],T)
                        QUANTIZEDarr[i:i+8,j:j+8,k] = np.matmul(np.transpose(T), mul1)
     
    #add 128 from QUANTIZEDarr to comply to jpeg standard
    QUANTIZEDarr += 128
    
    
    return QUANTIZEDarr
    
    
###
#Quantization of DCT image D
# apply seperately for each 8x8 block B 
# B' = round_nearest_int(B/Q) , dividing element wise
# Q is LuminananceTable or ChrominanceTable
#
# after steps above, do inverse by multiplying element wise
# B'' = B'*Q
###    
def Quantization(DCTarr,height,width,channels, qf):
    
    LuminananceTable = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]])
    
    ChrominanceTable = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]])

    # 50 is the usual compression; scaling factor = 1
    # qf can only be between 0 and 100

    # getting scaling factor which will be multiplied to arrays 
    if qf >= 50:
        scaling_factor = (100-qf)/50
    else:
        scaling_factor = (50/qf)

    #scaling matrices
    if scaling_factor != 0:# qf is not 0
        scaled_matrixL = scaling_factor * np.array(LuminananceTable)
        scaled_matrixC = scaling_factor * np.array(ChrominanceTable)
    else:#no quantization
        scaled_matrixL = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]])
        scaled_matrixC = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]])

    scaled_matrixL.astype(np.uint8)
    scaled_matrixC.astype(np.uint8)
    
    #split YUVarr into 8x8 blocks then aply Quantization 
    for i in range(0,height,8):
        for j in range(0,width,8):
            for k in range(channels):
                if i%8 == 0 and j%8 ==0:
                    #use LuminananceTable for Y channel
                    if(k == 0):
                        DCTarr[i:i+8,j:j+8,k] = np.rint(np.divide(DCTarr[i:i+8,j:j+8,k],scaled_matrixL))
                        #apply inverse by multiplying element wise
                        DCTarr[i:i+8,j:j+8,k] = np.multiply(DCTarr[i:i+8,j:j+8,k],scaled_matrixL)
                    #use ChrominanceTable for U and V channel
                    else:
                        DCTarr[i:i+8,j:j+8,k] = np.rint(np.divide(DCTarr[i:i+8,j:j+8,k],scaled_matrixC))
                        #apply inverse  by multiplying element wise
                        DCTarr[i:i+8,j:j+8,k] = np.multiply(DCTarr[i:i+8,j:j+8,k],scaled_matrixC)
      
    return DCTarr
    
def encoder(imageName, qualityFactor):
    #read image
    imgBGR = cv2.imread(imageName, cv2.IMREAD_COLOR) 
    #convert to RGB since imread returns BGR
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    
    #print("RGB\n", imgRGB)
    
    
    # height, width, number of channels in image
    dimensions = imgRGB.shape
    height = dimensions[0]
    width = dimensions[1]
    channels = dimensions[2]
    
    #if dimensions are not multiple of 8, discard excess pixels
    height = (height//8)*8 #integer division
    width = (width//8)*8
    
    
    cv2.imshow('image1', imgBGR)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    
    #convert RGB to YUV 
    YUVarr = RGB2YUV(imgRGB,height,width,channels)
    #print("YUV\n", YUVarr)
    
    
    #do 4:2:0 chroma sumsampling on U and V channels
    YUVarr = subsampleUV(YUVarr,height,width,channels)
    #print("subsampled\n", YUVarr)

    
    #   construct T needed for DCT and inverse DCT
    T = np.empty(shape=(8,8))
    #first row of T is 1/(2*sqrt(2)) 
    T[0,:] = 1/(2*np.sqrt(2))
    #construct the rest of T 
    for i in range(7):
        for j in range(8):
            #print(((i + 1)*(j*2 + 1)))
            T[i+1][j] = 0.5*cos(((i + 1)*(j*2 + 1)*pi)/16)
    
    
    YUVarr = DCT(YUVarr, height,width,channels,T)
    #print("DCT\n",YUVarr) 
    
    
    #apply quantization
    YUVarr = Quantization(YUVarr, height, width,channels,qualityFactor)
    #print("QUANTIZEDarr\n",YUVarr) 
    
    
    #do inverse DCT
    YUVarr = InverseDCT(YUVarr, height,width,channels,T)
    #print("Inverse DCT\n",YUVarr)
    
    #YUV to RGB
    ResultingRGB = (YUV2RGB(YUVarr, height, width, channels))
    #set range to be between 0 and 1--to display image properly
    #image has Red,Blue, and Green artifacts if this is not set
    ResultingRGB = ResultingRGB/255
    #set datatype to float32 to tell imshow() the range is between 0 and 1
    ResultingRGB = ResultingRGB.astype(np.float32)
    #print("Resulting RGB \n", ResultingRGB)
    
    
    #RGB to BGR, for showing image using imshow()
    ResultingBGR  = cv2.cvtColor(ResultingRGB , cv2.COLOR_RGB2BGR)
    #print("Resulting BGR\n", ResultingBGR)

    
    cv2.imshow("image2", ResultingBGR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###########

#main()

#creating a blank window
root = Tk()

root.title("Programming Assignment 2") 

title = Label(root, text="Programming Assignment 2", font=40)

#strings to display
label1 = Label(root, text="Please enter the name of image with extension: ")
label2 = Label(root, text="Please enter the desired quality factor (1-100): ")

#holds what is written in entries
file = StringVar()
number = StringVar()

#entry boxes
entry1 = Entry(root,textvariable=file)
entry2 = Entry(root, textvariable=number)


#placing labels and entries on window
title.grid(row=0, column=1, columnspan=2, pady=10)
label1.grid(row=1, column=1, sticky=E,padx=10, pady=10)
label2.grid(row=2,column = 1,sticky=E,padx=10)
entry1.grid(row=1,column=2,padx=10, pady=5) 
entry2.grid(row=2, column=2,padx=10)


def doit():#what to do when button is pressed
    if( int(number.get()) <= 0 or int(number.get()) > 100):
        tkinter.messagebox.showinfo("Invalid Input!","Please enter an integer between 1 to 100.")
    else:
        encoder(file.get(), int(number.get()))

#creating a button to do encoding
button = Button(root, text="Encode JPEG",command=doit)
button.grid(row=3,column=1,columnspan=2,pady=10)

root.mainloop()#displays window



