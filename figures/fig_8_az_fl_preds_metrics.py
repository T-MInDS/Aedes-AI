import os, sys
import pandas as pd, numpy as np
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append("./utils")
from performance_metrics import *

def autolabel(rects,ax):
    for rect in rects:
        height=rect.get_height()
        if height>=0:
            ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        else:
            ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='top')

if __name__ == '__main__':
        
    ddir="./results/Test/"
<<<<<<< HEAD
    fil1=ddir+"Test_lstm_model_predictions.csv"
    fil2=ddir+"Test_lstm_model_dpo_predictions.csv"
    fil3=ddir+"Test_lstm_model_dpo_ta_predictions.csv"
    fil4=ddir+"Test_lstm_model_ta_predictions.csv"
=======
    fil1=ddir+"Test_gru_predictions.csv"
    fil2=ddir+"Test_gru_hi_predictions.csv"
    fil3=ddir+"Test_gru_lo_predictions.csv"
    fil4=ddir+"Test_gru_hi_lo_predictions.csv"
>>>>>>> revisions

    font={'size':18}
    mpl.rc('font',**font)
    c1=(0.85,0.11,0.38)
    c2=(0.12,0.53,0.90)
    c3=(1,0.76,0.03)
    c4=(0,0.30,0.25)
    colors=[c1,c2,c3,c4]
    labels=["Base","HI","LO","HI LO"]
    files=[fil1,fil2,fil3,fil4]
    styles=['--','-.','-',':']
    alphas=[1,1,0.75,1]
    cos=['Avondale,Arizona', 'Collier,Florida']
    index=0
    for co in cos:
        r2,rmse,auc,r=list(),list(),list(),list()
        fig=plt.figure(index)
        plt.subplots_adjust(wspace=0.25)
        for i in range(0,4):
            fil=files[i]
            data=pd.read_csv(fil)
            mols=data[(data.Year==2020) & (data.Location==co)].MoLS
            nn=data[(data.Year==2020) & (data.Location==co)]["Neural Network"]
            results=score(mols,nn)
            r2.append(round(results[0],3))
            rmse.append(round(results[1],3))
            auc.append(round(results[2],3))
            r.append(round(results[3],3))
            x=np.arange(len(mols))
            axs=plt.subplot(1,3,1)
            axs.plot(x,mols,color='black',linewidth=3,alpha=0.75,label='MoLS' if i==0 else "")
            axs.plot(x,nn,color=colors[i],label=labels[i],linestyle=styles[i],alpha=alphas[i])
        axs.legend(loc='upper left')
        axs.set_ylim([0,4500])
        if co=='Avondale,Arizona':
            axs.set_title(co.replace(',',', '))
        else:
            axs.set_title('Collier County, Florida')
        axs.set_ylabel('Abundance Predictions')
        axs.set_xlabel('Days')
        x_bar=np.arange(4)

        ax=plt.subplot(2,3,2)
        ax.set_ylim([0,1.1])
        rec=ax.bar(x_bar,r2,color=[c1,c2,c3,c4])
        autolabel(rec,ax)
        ax.set_xticklabels(["Base","Base","HI","LO","HI LO"])
        ax.set_ylabel('$R_+^2$')

        ax=plt.subplot(2,3,3)
        ax.set_ylim([0,1.1])
        rec=ax.bar(x_bar,r,color=[c1,c2,c3,c4])
        autolabel(rec,ax)
        ax.set_xticklabels(["Base","Base","HI","LO","HI LO"])
        ax.set_ylabel('$r$')

        ax=plt.subplot(2,3,5)
        ax.set_ylim([0,0.18])
        rec=ax.bar(x_bar,rmse,color=[c1,c2,c3,c4])
        autolabel(rec,ax)
        ax.set_xticklabels(["Base","Base","HI","LO","HI LO"])
        ax.set_ylabel('$NRMSE$')

        ax=plt.subplot(2,3,6)
        ax.set_ylim([-0.16,0.16])
        rec=ax.bar(x_bar,auc,color=[c1,c2,c3,c4])
        autolabel(rec,ax)
        ax.set_xticklabels(["Base","Base","HI","LO","HI LO"])
        ax.set_ylabel('$Rel.\; AUC\; Diff.$') 
        index+=1

    plt.show()
