# import libraries
import glob
import cv2
import numpy as np
import scipy.stats

# load subjective scores from dmos.mat
import scipy.io
mat = scipy.io.loadmat('dmos.mat')

# read reference images
# 28 images in this folder
img1=[]
cv_img=[]
for img in glob.glob("Live Database/databaserelease2/refimgs/*.bmp"):
		n=cv2.imread(img,0)
		img1.append(img[39:len(img)])
	    	cv_img.append(n)
# print(img1)

# read distorted images from jp2k
# 227 images in this folder
imgName1=[]
disImg1=[]
for img in glob.glob("Live Database/databaserelease2/jp2k/*.bmp"):
		if(img[36]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName1.append(img[36:len(img)])
			disImg1.append(disImgRead)
    


# read from jpeg
# 233 images in this folder
imgName2=[]
disImg2=[]
for img in glob.glob("Live Database/databaserelease2/jpeg/*.bmp"):
		if(img[36]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName2.append(img[36:len(img)])
			disImg2.append(disImgRead)



# read from wn
# 174 images in this folder
imgName3=[]
disImg3=[]
for img in glob.glob("Live Database/databaserelease2/wn/*.bmp"):
		if(img[34]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName3.append(img[34:len(img)])
			disImg3.append(disImgRead)
    


# read from gblur
# 174 images in this folder
imgName4=[]
disImg4=[]
for img in glob.glob("Live Database/databaserelease2/gblur/*.bmp"):
		if(img[37]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName4.append(img[37:len(img)])
			disImg4.append(disImgRead)


# read from fastfading
# 173 images in this folder 
# Be carefull one image is missing from folder
imgName5=[]
disImg5=[]
for img in glob.glob("Live Database/databaserelease2/fastfading/*.bmp"):
		if(img[42]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName5.append(img[42:len(img)])
			disImg5.append(disImgRead)

ysub=mat['dmos'][0]
## print subjective scores from dmos.mat 
# print(ysub[0:227])
# feature vector
X=np.zeros((982,768))
# since ysub given are not according to order of ysub
# yMod is modified y's
yMod=np.zeros((982,1))

# for handling jp2k dataset
index_X=0
f=open("Live Database/databaserelease2/jp2k/info.txt","r")
f1=f.readlines()
for x in f1:
    orgImg=x.split(" ")[0]
    dImg=x.split(" ")[1]
    if(dImg[5]=='b'):
        p=dImg[3]
        p=int(p,10)
#         print(p)
        yMod[index_X,:]=ysub[p-1]
    elif(dImg[6]=='b'):
        p=dImg[3:5]
        p=int(p,10)
#         print(p)
        yMod[index_X,:]=ysub[p-1]
    elif(dImg[7]=='b'):
        p=dImg[3:6]
        p=int(p,10)
#         print(p)
        yMod[index_X,:]=ysub[p-1]
#     print(temp)

# print(yMod[0:227])
# print(ysub[0:227])
    k=img1.index(orgImg)
#     print(orgImg)
    A=cv_img[k]
        # print(A)
    l=imgName1.index(dImg)
        # print(dImg)
    Ap=disImg1[l]
        # print(A.shape,Ap.shape)
        # padding
    U, sig,V =np.linalg.svd(A, full_matrices=True)
    Up,sigp,Vp=np.linalg.svd(Ap, full_matrices=True)
    Un=np.zeros((768,768))
    Upn=np.zeros((768,768))
    Vn=np.zeros((768,768))
    Vpn=np.zeros((768,768))
        # print(len(U),len(V),len(Up),len(Upn))
    Un[0:len(U),0:len(U)]=U
    Upn[0:len(Up),0:len(Up)]=Up
    Vn[0:len(V),0:len(V)]=V
    Vpn[0:len(V),0:len(V)]=Vp
    alpha=np.multiply(Un,Upn)
    alpha=np.sum(alpha,axis=0)/768
    beta=np.multiply(Vn,Vpn)
    beta=np.sum(beta,axis=0)/768
    # print(alpha)
    X[index_X,:]=np.add(alpha,beta)
    # print(index_X,X[0:226])
    index_X=index_X+1  
# print(yMod[0:226])





f=open("Live Database/databaserelease2/jpeg/info.txt","r")
f1=f.readlines()
for x in f1:
        orgImg=x.split(" ")[0]
        dImg=x.split(" ")[1]
        offset1=226
        if(dImg[5]=='b'):
            p=dImg[3]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+226]
        elif(dImg[6]=='b'):
            p=dImg[3:5]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+226]
        elif(dImg[7]=='b'):
            p=dImg[3:6]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+226]
        k=img1.index(orgImg)
        A=cv_img[k]
        l=imgName2.index(dImg)
        Ap=disImg2[l]
        # print(A.shape,Ap.shape)
        # padding
        U, sig,V =np.linalg.svd(A, full_matrices=True)
        Up,sigp,Vp=np.linalg.svd(Ap, full_matrices=True)
        Un=np.zeros((768,768))
        Upn=np.zeros((768,768))
        Vn=np.zeros((768,768))
        Vpn=np.zeros((768,768))
        # print(len(U),len(V),len(Up),len(Upn))
        Un[0:len(U),0:len(U)]=U
        Upn[0:len(Up),0:len(Up)]=Up
        Vn[0:len(V),0:len(V)]=V
        Vpn[0:len(V),0:len(V)]=Vp
        alpha=np.multiply(Un,Upn)
        alpha=np.sum(alpha,axis=0)/768
        beta=np.multiply(Vn,Vpn)
        beta=np.sum(beta,axis=0)/768
        X[index_X,:]=np.add(alpha,beta)
        # print(index_X,X[227:459])
        index_X=index_X+1

# print(yMod[227:459])


# for wind noise data
f=open("Live Database/databaserelease2/wn/info.txt","r")
f1=f.readlines()
for x in f1:
        orgImg=x.split(" ")[0]
        dImg=x.split(" ")[1]
        # print(dImg,orgImg)
        offset2=459
        if(dImg[5]=='b'):
            p=dImg[3]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+459]
        elif(dImg[6]=='b'):
            p=dImg[3:5]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+459]
        elif(dImg[7]=='b'):
            p=dImg[3:6]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+459]
        k=img1.index(orgImg)
        A=cv_img[k]
        l=imgName3.index(dImg)
        Ap=disImg3[l]
        # print(A.shape,Ap.shape)
        # padding
        U, sig,V =np.linalg.svd(A, full_matrices=True)
        Up,sigp,Vp=np.linalg.svd(Ap, full_matrices=True)
        Un=np.zeros((768,768))
        Upn=np.zeros((768,768))
        Vn=np.zeros((768,768))
        Vpn=np.zeros((768,768))
        # print(len(U),len(V),len(Up),len(Upn))
        Un[0:len(U),0:len(U)]=U
        Upn[0:len(Up),0:len(Up)]=Up
        Vn[0:len(V),0:len(V)]=V
        Vpn[0:len(Vp),0:len(Vp)]=Vp
        alpha=np.multiply(Un,Upn)
        alpha=np.sum(alpha,axis=0)/768
        beta=np.multiply(Vn,Vpn)
        beta=np.sum(beta,axis=0)/768
        X[index_X,:]=np.add(alpha,beta)
#         print(index_X,X[460:634])
        index_X=index_X+1
        
# print(yMod[460:634])



# for gblur data
# print(index_X)
f=open("Live Database/databaserelease2/gblur/info.txt","r")
f1=f.readlines()
# print(f1)
for x in f1:
    for i in range(0,174):
        temp=x.split("\r\r")[i]
        orgImg=temp.split(" ")[0]
        dImg=temp.split(" ")[1]
#         print(i,orgImg,dImg)
        if(dImg[5]=='b'):
            p=dImg[3]
            p=int(p,10)
            # print(p)
            yMod[index_X,:]=ysub[p+633]
        elif(dImg[6]=='b'):
            p=dImg[3:5]
            p=int(p,10)
            # print(p)
            yMod[index_X,:]=ysub[p+633]
        elif(dImg[7]=='b'):
            p=dImg[3:6]
            p=int(p,10)
            # print(p)
            yMod[index_X,:]=ysub[p+633]
        k=img1.index(orgImg)
        A=cv_img[k]
        l=imgName4.index(dImg)
        Ap=disImg4[l]
        U, sig,V =np.linalg.svd(A, full_matrices=True)
        Up,sigp,Vp=np.linalg.svd(Ap, full_matrices=True)
        Un=np.zeros((768,768))
        Upn=np.zeros((768,768))
        Vn=np.zeros((768,768))
        Vpn=np.zeros((768,768))
        Un[0:len(U),0:len(U)]=U
        Upn[0:len(Up),0:len(Up)]=Up
        Vn[0:len(V),0:len(V)]=V
        Vpn[0:len(Vp),0:len(Vp)]=Vp
        alpha=np.multiply(Un,Upn)
        alpha=np.sum(alpha,axis=0)/768
        beta=np.multiply(Vn,Vpn)
        beta=np.sum(beta,axis=0)/768
        X[index_X,:]=np.add(alpha,beta)
        # print(index_X,X[634:808])
        index_X=index_X+1
        # print(index_X)

# print(yMod[634:808])


# for fastfading data
f=open("Live Database/databaserelease2/fastfading/info.txt","r")
f1=f.readlines()
for x in f1:
        orgImg=x.split(" ")[0]
        dImg=x.split(" ")[1]
        offset1=807
        if(dImg[5]=='b'):
            p=dImg[3]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+807]
        elif(dImg[6]=='b'):
            p=dImg[3:5]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+807]
        elif(dImg[7]=='b'):
            p=dImg[3:6]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+807]
        if(dImg!="img1.bmp"):
            k=img1.index(orgImg)
            A=cv_img[k]
            l=imgName5.index(dImg)
            Ap=disImg5[l]
            U, sig,V =np.linalg.svd(A, full_matrices=True)
            Up,sigp,Vp=np.linalg.svd(Ap, full_matrices=True)
            Un=np.zeros((768,768))
            Upn=np.zeros((768,768))
            Vn=np.zeros((768,768))
            Vpn=np.zeros((768,768))
            # print(len(U),len(V),len(Up),len(Upn))
            Un[0:len(U),0:len(U)]=U
            Upn[0:len(Up),0:len(Up)]=Up
            Vn[0:len(V),0:len(V)]=V
            Vpn[0:len(Vp),0:len(Vp)]=Vp
            alpha=np.multiply(Un,Upn)
            alpha=np.sum(alpha,axis=0)/768
            beta=np.multiply(Vn,Vpn)
            beta=np.sum(beta,axis=0)/768
            X[index_X,:]=np.add(alpha,beta)
            # print(index_X,X[808:983])
            index_X=index_X+1

##Above code give you feature vector of 982*768 from given dataset
################# feature vector extraction completed################


############################ SVR TRAINING AND PREDICTION#############
ysub=yMod
import numpy as np

#dividing training and testing data
from sklearn.model_selection import train_test_split
dat1_train, dat1_test, dat2_train, dat2_test = train_test_split(X,ysub, test_size = 0.2, random_state = 0)

# for training svr
from sklearn import svm
svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(dat1_train, dat2_train).predict(dat1_test)

# printing value of corelation coefficient
b=dat2_test.ravel()
print(np.corrcoef(y_rbf,b))



##################NEURAL NETWORK TRAINING##############################
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# define a model
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(512, input_dim=512, kernel_initializer='normal', activation='relu')) # input layer; size 13
    model.add(Dense(450, kernel_initializer='normal', activation='relu')) # hidden layer; size 6
    model.add(Dense(1, kernel_initializer='normal')) # output layer; size 1 with no activation since it's a regression problem
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# dividing trainig and testing data
X_new=X[:,0:512]
x_train, x_test, y_train, y_test = train_test_split(X_new,ysub, test_size= 0.2, random_state=27)

# predicting value using neural network
seed=7
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
pipeline.fit(x_train,y_train)
y_neural=pipeline.predict(x_test)


# finding pearson co-relation coefficent 
new_y_test=y_test.ravel()
print(np.corrcoef(y_neural,new_y_test))


########################TRAINING USING LOGISTIC REGRESSION ########################

# dividing dmos values into 5 classes
# labelled 0,1,2,3,4
from sklearn.linear_model import LogisticRegression
y_train=(y_train/20).astype('int')
y_test=(y_test/20).astype('int')

# Training logistic classifier and predicting values
logisticRegr = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logisticRegr.fit(x_train, y_train)
y_logistic = logisticRegr.predict(x_test)

# finding pearson co-relation coefficient
y_test=y_test.ravel()
print(np.corrcoef(y_logistic,y_test))

#########################end##############################
