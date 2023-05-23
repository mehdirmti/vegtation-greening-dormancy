#from http.client import _DataType
from pickle import TRUE
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt


# load data
data = np.load('#Outputs/8_PixelWiseDataForGMDHAnalysis.npz' , allow_pickle=True)
# Convert numpy.lib.npyio.NpzFile object to dict
data = dict(zip((data.files), (data[k] for k in data)))

ET = data['ET'].tolist()
SSM = data['SSM'].tolist()
RSM = data['RSM'].tolist()
T = data['T'].tolist()
P = data['P'].tolist()
VPD = data['VPD'].tolist()

OG = data['OG']
OD = data['OD']
PeakTime = data['PeakTime']
lat = data['lat']
lon = data['lon']
year = data['year']


def HydroMeteoDataAssignment(X, DATA, n_range):
    (a, b, c) = DATA['winter'].shape
    X[n_range[0],:] = np.nanmean(DATA['winter'].reshape(a, b*c), axis = 1) 
    X[n_range[1],:] = np.nanmean(DATA['earlyspring'].reshape(a, b*c), axis = 1)
    X[n_range[2],:] = np.nanmean(DATA['spring'].reshape(a, b*c), axis = 1)
    X[n_range[3],:] = np.nanmean(DATA['summer'].reshape(a, b*c), axis = 1)
    X[n_range[4],:] = np.nanmean(DATA['latesummer'].reshape(a, b*c), axis = 1)

    return X

# define a function to do analysis
def Do_PCA_Analysis(TARGET, Targ):
    
    y = np.nanmean(TARGET.reshape(TARGET.shape[0], TARGET.shape[1]*TARGET.shape[2]), axis = 1)

    # store variables name for future use
    Variables_name = [r'$SSM_{w}$', r'$SSM_{es}$', r'$SSM_{sp}$', r'$SSM_{su}$', r'$SSM_{ls}$',
                        r'$RSM_{w}$', r'$RSM_{es}$', r'$RSM_{sp}$', r'$RSM_{su}$', r'$RSM_{ls}$',
                        r'$T_{w}$', r'$T_{es}$', r'$T_{sp}$', r'$T_{su}$', r'$T_{ls}$',
                        r'$P_{w}$', r'$P_{es}$', r'$P_{sp}$', r'$P_{su}$', r'$P_{ls}$',
                        r'$VPD_{w}$', r'$VPD_{es}$', r'$VPD_{sp}$', r'$VPD_{su}$', r'$VPD_{ls}$'] 
    Variables_name = np.array(Variables_name) 

    # allocate free space to put data
    x = np.full((len(Variables_name), len(y)), np.nan)

    # Hydro-meteorological data assignment
    x = HydroMeteoDataAssignment(x, SSM, np.arange(0,5))
    x = HydroMeteoDataAssignment(x, RSM, np.arange(5,10))
    x = HydroMeteoDataAssignment(x, T, np.arange(10,15))
    x = HydroMeteoDataAssignment(x, P, np.arange(15,20))
    x = HydroMeteoDataAssignment(x, VPD, np.arange(20,25))

    data = np.concatenate((y.reshape((1, len(y))), x), axis=0).transpose()
    Variables_name = np.insert(Variables_name, 0, Targ)


    if Targ == 'OG':
        ind = []
        ind = np.ones(len(Variables_name), dtype=bool)
        ind[3:len(Variables_name):5] = 0
        ind[4:len(Variables_name):5] = 0
        ind[5:len(Variables_name):5] = 0
        data = data[:, ind]
        Variables_name = Variables_name[ind]
    elif Targ == 'OD':
        ind = []
        ind = np.ones(len(Variables_name), dtype=bool)
        ind[1:len(Variables_name):5] = 0
        ind[2:len(Variables_name):5] = 0
        data = data[:,ind]
        Variables_name = Variables_name[ind]  


    data = pd.DataFrame(data, columns=Variables_name)

    from sklearn.preprocessing import StandardScaler
    data_s = StandardScaler().fit_transform(data)
    data_s = pd.DataFrame(data_s, columns=Variables_name)

    from sklearn.decomposition import PCA
    pcamodel = PCA(n_components=10)
    pca = pcamodel.fit_transform(data_s)
    Explained_Variance = pcamodel.explained_variance_ 
    Acc_Explained_Variance = pcamodel.explained_variance_ratio_

    # define a figure 
    fig = plt.figure(figsize=(9, 6))
    plt.bar(range(1,len(pcamodel.explained_variance_ )+1),pcamodel.explained_variance_ )
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1,len(pcamodel.explained_variance_ )+1),
                np.cumsum(pcamodel.explained_variance_),
                c='red',
                label="Cumulative Explained Variance")
    plt.legend(loc='upper left')
    plt.show()

    # define a figure 
    fig = plt.figure(figsize=(9, 6))
    plt.plot(pcamodel.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    #PCA1 is at 0 in xscale
    # define a figure 
    fig = plt.figure(figsize=(9, 6))
    plt.plot(pcamodel.explained_variance_)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    def myplot(score,coeff,labels=None):
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        # define a figure 
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams.update({'font.size': 12})
        ax.scatter(xs * scalex,ys * scaley,s=5)
        for i in range(n):
            ax.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center',**{'fontname':'Times New Roman', 'size':'12'})
            else:
                ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center',**{'fontname':'Times New Roman', 'size':'12'})
    
        ax.set_xlabel("PC{}".format(1),**{'fontname':'Times New Roman', 'size':'14'})
        ax.set_ylabel("PC{}".format(2),**{'fontname':'Times New Roman', 'size':'14'})
        ax.grid()

        # Set the tick labels font
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Times New Roman')
            label.set_fontsize(14)        
        fig.savefig('#Outputs/Script 11/' + Targ + '-biplot.jpg', dpi=600, transparent=False)   # save the figure to file
  
        #ax.set_title(Targ)

    myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),list(data_s.columns))
    # Store laoding to excel
    loadings = pcamodel.components_.T * np.sqrt(pcamodel.explained_variance_)

    loading_matrix = pd.DataFrame(loadings[:,:2], columns=['PC1', 'PC2'], index=data_s.columns)
    pd.DataFrame(loading_matrix, columns=['PC1', 'PC2']).to_excel(r'#Outputs/Script 11/' + Targ + '.xlsx', index = True)
    plt.show()

    return 
# end of function

Do_PCA_Analysis(OG, 'OG')
Do_PCA_Analysis(OD, 'OD')
