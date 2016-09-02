import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as mpl
from numpy import genfromtxt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def training(df):
	features=df.shape[1]-1
	
	df3=pd.concat([df2,df1])
	label=list(df3.label)
	matrix=StandardScaler().fit_transform(df3.ix[:,:(len(df3.columns)-1)])

	size=[]
	meanvec=[]
	for i in [3,5]:

		df1=df[df.label==i]
		size.append(df1.shape[0])
		meanvec1=df1.mean(axis=0)	
		meanvec1=meanvec1[:features]
		meanvec.append(np.asarray(meanvec1))

	scat_in = np.zeros((features,features))
	for cl,mv in zip([3,5],meanvec):
		sc_mat = np.zeros((features,features)) 
		df2=df[df.label==cl]
		df1=df2.drop('label',1)

		for i,dfrow in df1.iterrows():
			row=np.asarray(dfrow)
			mv=mv.reshape(1,features)        
			row, mv = row.reshape(features,1), mv.reshape(features,1)
			sc_mat += (row-mv).dot((row-mv).T)
		scat_in += sc_mat 

	mean_total = np.asarray(df.mean(axis=0))
	mean_total=mean_total[:features]
	scat_bet = np.zeros((features,features))
	for i,mvec in enumerate(meanvec):
		mvec=mvec.reshape(1,features)    
		mvec = mvec.reshape(features,1)
		mean_total = mean_total.reshape(features,1) # make column vector
		scat_bet += size[i] * (mvec - mean_total).dot((mvec - mean_total).T)
	eigen_val, eigen_vec = np.linalg.eig(np.linalg.pinv(scat_in).dot(scat_bet))  


	eigens=list()

	for i in range(len(eigen_val)):
		eigens.append([(np.abs(eigen_val[i])),eigen_vec[:,i]])
	eigens.sort
	eigens.reverse
	eigen_total=sum(eigen_val)
	lam=[]
	cum_sum=0
	for value in eigen_val:
		cum_sum+=value
		lam.append(cum_sum/eigen_total)

	plt.plot(lam,marker='o')
	plt.xlabel("# of Features")
	plt.ylabel("Cumulative sum of eigen values/eigen value total")
	plt.show()

	last=[]
	name=[]
	for i in range(features):
		last.append(eigens[i][1].reshape(features,1))
		name.append(str(i))
	name.append("label")
	reduced=np.hstack(last)
	final=matrix.dot(reduced)
	final=np.real(final)
	df_out=pd.DataFrame(columns=name)
	for i in range(features):
			df_out.ix[:,i]= final[:,i]
	df_out['label']=label
	print df_out.shape
	labels=np.asarray(label)
	labels=labels.reshape(1,df3.shape[0])
	labels=labels.reshape(df3.shape[0],1)

	return df_out,reduced,final,labels,name


def get_data():
	df=pd.read_csv("pca_out.csv",index_col=0)
	test=pd.read_csv("pca_test_out.csv",index_col=0)
	return df,test

def prepare(df,model):
	features=df.shape[1]-1

	df1=df[df.label==3]
	df2=df[df.label==5]
	df3=pd.concat([df2,df1])
	m=np.asarray(df3)
	label=m[:,features]
	mat=StandardScaler().fit_transform(df3.ix[:,:(len(df3.columns)-1)])
	matrix=mat.dot(model)
	matrix=np.real(matrix)
	return matrix,label

if __name__=="__main__":
	train,df=get_data()
	features=train.shape[1]-1
	df_out,model,t,label,name=training(train)
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(t,label) 
	df_out.to_csv("outfile", sep=",")
	test,labelt=prepare(df,model)
	df_out2=pd.DataFrame(columns=name)
	for i in range(features):
			df_out2.ix[:,i]= test[:,i]
	df_out2.label=labelt
	df_out2.to_csv("lda_test_out_35.csv", sep=",")
	ypred=[]
	hashed={'T':0,'F':0}
	for i in range(len(test)):
		p1=neigh.predict(test[i])
		ypred.append(p1)
		if((p1==labelt[i])):
			hashed['T']+=1
	
		else:
			hashed['F']+=1
			
	ypred=np.asarray(ypred)

	print confusion_matrix(labelt, ypred)
	print float(hashed['T'])/(hashed['T']+hashed['F'])*100


