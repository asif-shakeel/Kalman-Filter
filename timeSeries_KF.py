'''
Author: Asif Shakeel

Kalman filter parameter calculations, prediction and smoothing. 

'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sc
import sys
from datetime import timedelta, datetime
from scipy import stats
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a ")
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
import os
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import linprog
import pathlib
from pathlib import Path
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sc
from datetime import timedelta, datetime
import sys
import os
import pathlib
from pathlib import Path
import geopandas
import osmnx
from sklearn.metrics.pairwise import euclidean_distances
import time
from shapely.geometry import Point


city_name = "Paris" # "Lyon" # "Paris"
application=  "Netflix" # "Netflix" # "Instagram" #
updownsuf="DL"
mean_or_sum='mean' # 'mean' or 'sum' over tiles 



whichdays='All' # 'weekends' or 'weekdays' or 'All'
smooth_pred='prediction' # 'smoothing' or 'prediction'
trainval_fit= 'trainval' # 'trainval'  or 'fit'
model='periodic' # 'periodic' or 'aperiodic'
 #range(95) #[starttimeindex]
optim='Nelder-Mead' # # 'Monte-Carlo' or some scipy optimizer like 'Nelder-Mead'
maxiterations=1000 # iterations of optimizer (not used)
noise_bound_divisor=1


startdaynum=0 # the day offset from 03/16 (startdaynum=0) on which to run the calculation
numdays=55 # number of days to run startng with startdaynum
numdaystrain=40
starttimeindex=0 # initial time index (0..95, here 45 is 11:15 am) on which to begin the flow, ending on starttimeindex+1
numtimes=96 # number of time slices per day




gapactive=True
valgapactive=False
fitgapactive=gapactive or valgapactive
valintervalactive=False
valintervaleval=False


gapinitind=15 #*numtimes
gapstartind=8
gapsize=4
valgapinitind=56 #*numtimes # should be smaller than startdaynum+numdays
valgapstartind=numtimes-5
valgapsize=numtimes-5

valintervalinitind=45 #*numtimes + 
valintervalstartind=int(numtimes/2)
valintervalsize=48




# gapstart=(gapinitind-startdaynum*numtimes)+gapstartind if startdaynum*numtimes<=gapinitind else (startdaynum+numdays+10)*numtimes
# valgapstart=(valgapinitind-(startdaynum+numdaystrain)*numtimes)+valgapstartind if (startdaynum+numdays)*numtimes>=valgapinitind else (startdaynum+numdays+10)*numtimes
# valintervalstart=(valintervalinitind-(startdaynum+numdaystrain)*numtimes)+valintervalstartind if (startdaynum+numdays)*numtimes>=valintervalinitind else (startdaynum+numdays+10)*numtimes

gapstart=(gapinitind-startdaynum)*numtimes+gapstartind if startdaynum*numtimes<=gapinitind*numtimes else (startdaynum+numdays+10)*numtimes
valgapstart=(valgapinitind-(startdaynum+numdaystrain))*numtimes+valgapstartind if (startdaynum+numdays)*numtimes>=valgapinitind*numtimes else (startdaynum+numdays+10)*numtimes
valintervalstart=(valintervalinitind-(startdaynum+numdaystrain))*numtimes+valintervalstartind if (startdaynum+numdays)*numtimes>=valintervalinitind*numtimes else (startdaynum+numdays+10)*numtimes



gapind=list(range(gapstart,gapstart+gapsize))
valgapind=list(range(valgapstart,valgapstart+valgapsize))
valintervalind=list(range(valintervalstart,valintervalstart+valintervalsize))
valgapindoff=[s + (startdaynum+numdaystrain)*numtimes for s in valgapind]
valintervalindoff=[s + (startdaynum+numdaystrain)*numtimes for s in valintervalind]

if valintervalactive:
    fitgapind=gapind+valgapindoff +valintervalindoff
else:
    fitgapind=gapind+valgapindoff

path=Path().resolve()
readdir=path  # Specify directory, assumed here to be under the same directory as this file
writedir=path  # Specify directory, assumed here to be under the same directory as this file
workdir=path


readdirdata=Path('/home/asifshakeel/NetMob_Data/Cities') / city_name /  application # Specify city/app directory, assumed here to be under the same directory as this file
outputsdir=Path('/home/asifshakeel/NetMob_Data/Outputs') / "KF"  / city_name /  application / updownsuf / model  / trainval_fit / smooth_pred
plotsdir=Path('/home/asifshakeel/NetMob_Data/Plots') / "KF" / city_name / application / updownsuf /  model  / trainval_fit  / smooth_pred



if not os.path.exists(plotsdir):
    os.makedirs(plotsdir)

if not os.path.exists(outputsdir):
    os.makedirs(outputsdir)








city_dims = {
    'Bordeaux': (334, 342),
    'Clermont-Ferrand': (208, 268),
    'Dijon': (195, 234),
    'Grenoble': (409, 251),
    'Lille': (330, 342),
    'Lyon': (426, 287),
    'Mans': (228, 246),
    'Marseille': (211, 210),
    'Metz': (226, 269),
    'Montpellier': (334, 327),
    'Nancy': (151, 165),
    'Nantes': (277, 425),
    'Nice': (150, 214),
    'Orleans': (282, 256),
    'Paris': (409, 346),
    'Rennes': (423, 370),
    'Saint-Etienne': (305, 501),
    'Strasbourg': (296, 258),
    'Toulouse': (280, 347),
    'Tours': (251, 270)
    }

rows=city_dims[city_name][0]
cols=city_dims[city_name][1]



regions_file=city_name+".geojson"
regions_path=readdir / "regions" / regions_file
regions_df=geopandas.read_file(regions_path)

municipalities = geopandas.read_file(
    regions_path
)


from shapely.geometry import LineString




filelist=[]
for root, dirs, files in os.walk(readdirdata):
    for name in files:
        filelist.append(str(Path(root).resolve() / name)) #filelist.append(os.path.join(root,name))
filelist=[f for f in filelist if updownsuf in f] # choose only the upload files
filelist=sorted(filelist)

filelist=filelist[startdaynum:startdaynum+numdays]

startdate=filelist[0][-15:-7]
startdate=startdate[-4:-2]+"-"+startdate[-2:]+"-"+startdate[:4]
trainenddate=filelist[numdaystrain-1][-15:-7]
trainenddate=trainenddate[-4:-2]+"-"+trainenddate[-2:]+"-"+trainenddate[:4]
valstartdate=filelist[numdaystrain][-15:-7]
valstartdate=valstartdate[-4:-2]+"-"+valstartdate[-2:]+"-"+valstartdate[:4]
enddate=filelist[numdays-1][-15:-7]
enddate=enddate[-4:-2]+"-"+enddate[-2:]+"-"+enddate[:4]

firstdf=True
emd_df=pd.DataFrame()
y_df=pd.DataFrame()
colnames=['date']+[str(k) for k in range(numtimes)]
daycnt=startdaynum


for filename in filelist:
    date=filename[-15:-7]
    outputsdirdate = outputsdir / date
    if not os.path.exists(outputsdirdate):
        os.makedirs(outputsdirdate)
    print("\n")
    print("*************************************************")
    print(date)
    print("*************************************************")
    start=pd.to_datetime(date[-4:-2]+"-"+date[-2:]+"-"+date[:4])
    datetime_range=pd.date_range(start=start, periods=numtimes,freq='15min')

    time_range=[datetimeind.strftime('%H:%M') for  datetimeind in datetime_range]


    columns=['tile_id']+time_range
    df = pd.read_csv(filename,  sep=" ", header=None) # read each day as a dataframe
    if date=='20190331':
        for cnt in range(4):
            df.insert(9, time_range[11-cnt],  0) #  df[9]
    df.columns=columns



    emddict={}
    emdlist=[]
    datalist=[]
    timelist=[]

    goodcalc=0
    badcalc=0
    
    datetime_range_sel=range(starttimeindex,starttimeindex+numtimes)

    startt=time.time()
    print("\n")
 
    colnames=['date']+['daycnt']+time_range
    if mean_or_sum=='sum':
        y_df.loc[len(y_df),colnames]=[date]+[daycnt]+df[time_range].sum(axis=0).to_list()
    else:
        y_df.loc[len(y_df),colnames]=[date]+[daycnt]+df[time_range].mean(axis=0).to_list()
    daycnt=(daycnt+1)%7

if whichdays=='weekdays':
    y_df=y_df[y_df['daycnt']>=2]
elif whichdays=='weekends':
    y_df=y_df[y_df['daycnt']<2]
  
def init_param_eval(y_df,model):

    daymean=y_df[time_range].mean(axis=0) #.iloc[0:7]
    y_df_noise=y_df.copy() #.iloc[0:7]
    y_df_noise=y_df_noise[time_range].sub(daymean, axis=1)
    mu1=daymean.mean()
    sigma0=np.std(y_df_noise.to_numpy()) # np.std(y_df[time_range].to_numpy()) #
    daydelta=np.diff(daymean.to_numpy())

    if model == 'periodic':
        m1=4*24
        k=1
        n=m1
        m=2
        parameter_vec_0=np.array([1/np.sqrt(4)*sigma0,1/np.sqrt(4)*sigma0,1/np.sqrt(4)*sigma0])

    else:

        m1=4*24
        k=1
        n=1
        m=1

        parameter_vec_0=np.array([1/np.sqrt(4)*sigma0,1/np.sqrt(4)*sigma0])





    return parameter_vec_0,mu1,sigma0,k,n,m

parameter_vec_0,mu1,sigma0,k,n,m=init_param_eval(y_df,model)
sigma_alpha_0=1/np.sqrt(4)*sigma0
print("data std = {}".format(sigma0))

def init_covMat_eval(model,parameter_vec,mu1,sigma_alpha_0):

    if model == 'periodic':
        m1=4*24
        k=1
        n=m1
        m=2

        H_0=(parameter_vec[0]**2)*np.diag(np.ones(k))
        Q_0=np.diag(np.array([parameter_vec[1]**2, parameter_vec_0[2]**2] ))
        P_0=((sigma_alpha_0)**2)*np.diag(np.ones(n))

        a_0=np.zeros((n,1))
        a_0[0,0]=mu1
        # a_0[1:,0]=daydelta[::-1]

        Z=np.zeros((k,n))
        Z[0,0]=1
        Z[0,1]=1

        T=np.zeros((n,n))
        T[0,0]=1
        T[1,1:]=-1
        for i in range(2,n):
            T[i,i-1]=1

        R=np.zeros((n,m))
        R[0,0]=1
        R[1,1]=1

    else:

        m1=4*24
        k=1
        n=1
        m=1


        H_0=(parameter_vec[0]**2)*np.diag(np.ones(k))
        Q_0=np.diag(np.array([parameter_vec[1]**2] ))
        P_0=((sigma_alpha_0)**2)*np.diag(np.ones(n))


        a_0=np.zeros((n,1))
        a_0[0,0]=mu1


        Z=np.zeros((k,n))
        Z[0,0]=1


        T=np.zeros((n,n))
        T[0,0]=1


        R=np.zeros((n,m))
        R[0,0]=1



    return H_0,Q_0,P_0,a_0,Z,T,R
        



def v_t_eval(y_t,Z,a_t):
    return y_t-np.matmul(Z,a_t)[0,0]

def F_t_eval(Z,P_t,H_t):
    return (np.linalg.multi_dot([Z,P_t,Z.T]) + H_t)[0,0]

def a_tt_eval(a_t,y_t,Z,P_t,H_t):
    return a_t + np.matmul(P_t,Z.T)/F_t_eval(Z,P_t,H_t)*v_t_eval(y_t,Z,a_t)

def P_tt_eval(Z,P_t,H_t):
    return P_t - np.linalg.multi_dot([P_t,Z.T,Z,P_t])/F_t_eval(Z,P_t,H_t)

def a_t_eval(T,a_tt):
    return np.matmul(T,a_tt)


def P_t_eval(T,P_tt,Q_t,R):
    return np.linalg.multi_dot([T,P_tt,T.T]) + np.linalg.multi_dot([R,Q_t,R.T])




def loglikelihood(parameter_vec,y_df_flat,sigma_alpha_0): 
    
    ll=len(y_df_flat)/2.0*np.log(2*np.pi)

    H_t,Q_t,P_t,a_t,Z,T,R=init_covMat_eval(model,parameter_vec,mu1,sigma_alpha_0)


    y_ind=0
    for y_t in y_df_flat:
#        if (y_ind >=gapstart and y_ind<gapstart+gapsize):
        if (gapactive and (y_ind in gapind)) or (fitgapactive  and (y_ind in fitgapind)):
            Z_t=np.zeros_like(Z) #Z #
            v_t=v_t_eval(y_t,Z_t,a_t)
            F_t=F_t_eval(Z_t,P_t,H_t)
        else:
            Z_t=Z
            v_t=v_t_eval(y_t,Z_t,a_t)
            F_t=F_t_eval(Z_t,P_t,H_t)
            if F_t!=0:
                ll+=1/2.0*(np.log(abs(F_t))+((v_t)**2)/F_t)
        y_ind+=1
        # v_t=v_t_eval(y_t,Z_t,a_t)
        # F_t=F_t_eval(Z_t,P_t,H_t)
        if F_t!=0:
            # ll+=1/2.0*(np.log(abs(F_t))+((v_t)**2)/F_t)
            a_tt=a_tt_eval(a_t,y_t,Z_t,P_t,H_t)
            P_tt=P_tt_eval(Z_t,P_t,H_t)
        else:
            a_tt=a_t
            P_tt=P_t
            
        a_t=a_t_eval(T,a_tt)
        P_t=P_t_eval(T,P_tt,Q_t,R)
        
    return ll

# y_df_check=y_df[time_range].to_numpy().flatten()
# rangeind=list(range(valgapinitind*numtimes+valgapstartind,valgapinitind*numtimes+valgapstartind+valgapsize))
# print(rangeind)
# print(y_df_check[rangeind])
# indices = y_df_check < 1000
# print(np.where(indices)[0])
# print(y_df_check[indices])

if trainval_fit=='trainval':
    y_df_val=y_df.copy()
    y_df=y_df.iloc[:numdaystrain,:]
    y_df_val=y_df_val.iloc[numdaystrain:,:]

        

y_df_flat=y_df[time_range].to_numpy().flatten()

if optim != 'Monte-Carlo':
    options={'maxiter':maxiterations,'disp':True}

    #bounds=[(0,10*sigma0) for s in parameter_vec_0]
    bounds=[(0,sigma0/noise_bound_divisor) for s in parameter_vec_0]

    def callbackfun(x, f, context):
        print("{} at minimum {} accepted {}".format(f, x, int(context)))
    # print(bounds)

    # def callbackfun(intermediate_result):
    #     print("result: {}" % (intermediate_result))

    result = sc.optimize.minimize(loglikelihood, parameter_vec_0, method=optim,args=(y_df_flat,sigma_alpha_0), bounds=bounds,options=options) #, constraints=linear_constraint, bounds=bounds,options=options) #
    #result = sc.optimize.dual_annealing(loglikelihood,args=(y_df_flat,Z,T,R,a_0,sigma_alpha_0), bounds=bounds,callback=callbackfun) #, constraints=linear_constraint, bounds=bounds,disp=True) #, options=options,constraints=linear_constraint
    #result = sc.optimize.basinhopping(loglikelihood,parameter_vec_0, minimizer_kwargs={'args':(y_df_flat,Z,T,R,a_0,sigma_alpha_0)},disp=True) #, constraints=linear_constraint, bounds=bounds) #, options=options,constraints=linear_constraint

    parameter_vec_opt=list(result.x)

else:
    parameter_vec=parameter_vec_0
    parameter_vec_opt=parameter_vec
    ll_list=[]
    param_list=[]
    param_list.append(parameter_vec)
    ll_list.append(-loglikelihood(parameter_vec,y_df_flat,sigma_alpha_0))
    i=0
    max_ll=ll_list[0]
    max_i=i


    while i < maxiterations:
        multiplier=np.exp(np.random.normal(0,0.2,size=3))
        parameter_vec=np.multiply(multiplier,param_list[i])
        ll=-loglikelihood(parameter_vec,y_df_flat,sigma_alpha_0)
        if np.random.rand() < np.exp(ll-ll_list[i]):
            i+=1
            param_list.append(parameter_vec)
            ll_list.append(ll)
            print("{}, {}, {} ".format(i,param_list[i],ll_list[i]))
            if max_ll<ll:
                max_ll=ll
                max_i=i
                parameter_vec_opt=parameter_vec
if len(parameter_vec_opt)==3:
    print("optimized parameters: sigma_epsilon, sigma_zeta, sigma_eta = {}".format(parameter_vec_opt))
else:
    print("optimized parameters: sigma_epsilon, sigma_zeta = {}".format(parameter_vec_opt))



if trainval_fit=='trainval':
    stages=['train','val']
else:
    stages=['fit']
for stage in stages:
    datetime_arr=pd.date_range(start=startdate, periods=numtimes*numdays,freq='15min')
    firstdate=startdate
    lastdate=enddate
   
    if stage=='train':
        firstdate=startdate
        lastdate=trainenddate
        
        datetime_arr=pd.date_range(start=startdate, periods=numtimes*numdaystrain,freq='15min')
    elif stage=='val':
        firstdate=valstartdate
        lastdate=enddate
      
        datetime_arr=datetime_arr[numdaystrain*numtimes:]
        y_df_flat=y_df_val[time_range].to_numpy().flatten()

    if smooth_pred=='prediction':
        legenlbl='prediction'



        H_t,Q_t,P_t,a_t,Z,T,R=init_covMat_eval(model,parameter_vec_opt,mu1,sigma_alpha_0)


        at_arr=[]
        Pt_arr=[]
        vt_arr=[]
        Ft_arr=[]

        y_hat_arr=[]
        stdpos_arr=[]
        stdneg_arr=[]

        y_ind=0
        Z_t=Z
        for y_t in y_df_flat:
            y_hat_t=np.matmul(Z,a_t)[0,0]
            y_hat_arr.append(y_hat_t)
            if n>1:
                if (P_t[0,0]+P_t[1,1]+2*(P_t[0,1])) < 0:
                    stdpos_arr.append(y_hat_t)
                    stdneg_arr.append(y_hat_t)
                else:
                    stdpos_arr.append(y_hat_t+2*np.sqrt(P_t[0,0]+P_t[1,1]+2*(P_t[0,1])))
                    stdneg_arr.append(y_hat_t-2*np.sqrt(P_t[0,0]+P_t[1,1]+2*(P_t[0,1])))
            else:
                stdpos_arr.append(y_hat_t+2*np.sqrt(P_t[0,0]))
                stdneg_arr.append(y_hat_t-2*np.sqrt(P_t[0,0]))
            if (gapactive and (stage=='train') and (y_ind in gapind)) or (valgapactive and (stage=='val') and (y_ind in valgapind)) or  (fitgapactive and (stage=='fit') and (y_ind in fitgapind))or  (valintervalactive and (stage=='val') and (y_ind in valintervalind)):  
                Z_t=np.zeros_like(Z) #
            else:
                Z_t=Z
            y_ind+=1
            v_t=v_t_eval(y_t,Z_t,a_t)
            F_t=F_t_eval(Z_t,P_t,H_t)
            vt_arr.append(v_t)
            Ft_arr.append(F_t)
            at_arr.append(a_t)
            Pt_arr.append(P_t)
            if F_t!=0:
                a_tt=a_tt_eval(a_t,y_t,Z_t,P_t,H_t)
                P_tt=P_tt_eval(Z_t,P_t,H_t)
            else:
                a_tt=a_t
                P_tt=P_t
            a_t=a_t_eval(T,a_tt)
            P_t=P_t_eval(T,P_tt,Q_t,R)


    else:
        legenlbl='smoothing'


        H_t,Q_t,P_t,a_t,Z,T,R=init_covMat_eval(model,parameter_vec_opt,mu1,sigma_alpha_0)

        at_arr=[]
        Pt_arr=[]
        vt_arr=[]
        Ft_arr=[]


        y_ind=0
        for y_t in y_df_flat:
            if (gapactive and (stage=='train') and (y_ind in gapind)) or (valgapactive and (stage=='val') and (y_ind in valgapind)) or  (fitgapactive and (stage=='fit') and (y_ind in fitgapind))or  (valintervalactive and (stage=='val') and (y_ind in valintervalind)): 
                Z_t=np.zeros_like(Z) #Z #
            else:
                Z_t=Z
            y_ind+=1
            v_t=v_t_eval(y_t,Z_t,a_t)
            F_t=F_t_eval(Z_t,P_t,H_t)
            vt_arr.append(v_t)
            Ft_arr.append(F_t)
            at_arr.append(a_t)
            Pt_arr.append(P_t)
            if F_t!=0:
                a_tt=a_tt_eval(a_t,y_t,Z_t,P_t,H_t)
                P_tt=P_tt_eval(Z_t,P_t,H_t)
            else:
                a_tt=a_t
                P_tt=P_t
            a_t=a_t_eval(T,a_tt)
            P_t=P_t_eval(T,P_tt,Q_t,R)

        r_t=np.zeros((n,1))
        N_t=np.zeros((n,n))



        H_t,Q_t,P_t,a_t,Z,T,R=init_covMat_eval(model,parameter_vec_opt,mu1,sigma_alpha_0)

        y_hat_arr=[]
        stdpos_arr=[]
        stdneg_arr=[]


        for j in range(len(y_df_flat))[::-1]:
            if (gapactive and (stage=='train') and (y_ind in gapind)) or (valgapactive and (stage=='val') and (y_ind in valgapind)) or  (fitgapactive and (stage=='fit') and (y_ind in fitgapind))  or  (valintervalactive and (stage=='val') and (y_ind in valintervalind)): 
                Z_t=Z # np.zeros_like(Z) #
            else:
                Z_t=Z
            v_t=vt_arr[j]
            F_t=Ft_arr[j]
            a_t=at_arr[j]
            P_t=Pt_arr[j]
            if F_t!=0:
                L_t=T-np.linalg.multi_dot([T,P_t,Z_t.T,Z_t])/F_t
                r_t=(Z_t.T)*v_t/F_t + np.matmul(L_t.T,r_t)
                N_t=np.matmul(Z_t.T,Z_t)/F_t + np.linalg.multi_dot([L_t.T,N_t,L_t])
                V_t=P_t - np.linalg.multi_dot([P_t,N_t,P_t])
            else:
                L_t=T
                r_t=np.matmul(L_t.T,r_t)
                N_t=np.linalg.multi_dot([L_t.T,N_t,L_t])
                V_t=P_t - np.linalg.multi_dot([P_t,N_t,P_t])
            alpha_hat_t=a_t+np.matmul(P_t,r_t)
            y_hat_t=np.matmul(Z,alpha_hat_t)[0,0]
            y_hat_arr.append(y_hat_t)
            if n>1:
                if (V_t[0,0]+V_t[1,1]+2*(V_t[0,1])) < 0:
                    stdpos_arr.append(y_hat_t)
                    stdneg_arr.append(y_hat_t)
                else:
                    stdpos_arr.append(y_hat_t+2*np.sqrt(V_t[0,0]+V_t[1,1]+2*(V_t[0,1])))
                    stdneg_arr.append(y_hat_t-2*np.sqrt(V_t[0,0]+V_t[1,1]+2*(V_t[0,1])))
            else:
                stdpos_arr.append(y_hat_t+2*np.sqrt(V_t[0,0]))
                stdneg_arr.append(y_hat_t-2*np.sqrt(V_t[0,0]))
        y_hat_arr=y_hat_arr[::-1]
        stdpos_arr=stdpos_arr[::-1]
        stdneg_arr=stdneg_arr[::-1]

    stdpos_arr=np.array(stdpos_arr)
    stdneg_arr=np.array(stdneg_arr)
    y_hat_arr=np.array(y_hat_arr)
    vt_arr=np.array(vt_arr)
    y_pred_arr=np.subtract(y_df_flat,y_hat_arr)

    if stage=='train':
        if gapactive:
            y_mask=gapind #list(range(gapstart,gapstart+gapsize))
            y_df_flat_masked=np.delete(y_df_flat,y_mask)
            y_hat_arr_masked=np.delete(y_hat_arr,y_mask)
        else:
            y_df_flat_masked=y_df_flat
            y_hat_arr_masked=y_hat_arr 

        # print(smooth_pred+" "+"rmse for {} data = {}".format(stage,np.sqrt(mean_squared_error(y_df_flat_masked,y_hat_arr_masked))))
        # print(smooth_pred+" "+"mape for {} data = {}".format(stage,np.sqrt(mean_absolute_percentage_error(y_df_flat_masked,y_hat_arr_masked))))
        # print(smooth_pred+" "+"rms innovation error  for {} data = {}".format(stage,np.sqrt(mean_squared_error(vt_arr,np.zeros_like(vt_arr)))))
        print("rms prediction error for {} data = {}".format(stage,np.sqrt(mean_squared_error(vt_arr,np.zeros_like(vt_arr)))))
    elif stage=='val':
        if valintervaleval:       
            y_mask=valintervalind #list(range(valgapstart,valgapstart+valgapsize))
            # print(y_mask)
            # print(y_df_flat[y_mask])
            y_df_flat_masked=y_df_flat[y_mask]
            y_hat_arr_masked=y_hat_arr[y_mask]
            # print(smooth_pred+" "+"rmse for {} data = {}".format(stage,np.sqrt(mean_squared_error(y_df_flat_masked,y_hat_arr_masked))))
            # print(smooth_pred+" "+"mape for {} data = {}".format(stage,np.sqrt(mean_absolute_percentage_error(y_df_flat_masked,y_hat_arr_masked)))) 
            print("rms prediction error for {} data = {}".format(stage,np.sqrt(mean_squared_error(vt_arr,np.zeros_like(vt_arr)))))
        elif trainval_fit=='trainval':
            y_df_flat_masked=y_df_flat
            y_hat_arr_masked=y_hat_arr
#            print(smooth_pred+" "+"rmse for {} data = {}".format(stage,np.sqrt(mean_squared_error(y_df_flat_masked,y_hat_arr_masked))))
            # print(smooth_pred+" "+"mape for {} data = {}".format(stage,np.sqrt(mean_absolute_percentage_error(y_df_flat_masked,y_hat_arr_masked)))) 
            print("rms prediction error for {} data = {}".format(stage,np.sqrt(mean_squared_error(vt_arr,np.zeros_like(vt_arr)))))
        else:
            print("val test interval inactive")
        # if valgapactive:       
        #     y_mask=valgapind #list(range(valgapstart,valgapstart+valgapsize))
        #     # print(y_mask)
        #     # print(y_df_flat[y_mask])
        #     y_df_flat_masked=np.delete(y_df_flat,y_mask)
        #     y_hat_arr_masked=np.delete(y_hat_arr,y_mask)
        # else:
        #     y_df_flat_masked=y_df_flat
        #     y_hat_arr_masked=y_hat_arr 
    else:
        if valintervalactive:       
            y_mask=valintervalindoff #list(range(valgapstart,valgapstart+valgapsize))
            # print(y_mask)
            # print(y_df_flat[y_mask])
            y_df_flat_masked=y_df_flat[y_mask]
            y_hat_arr_masked=y_hat_arr[y_mask]
#            print(smooth_pred+" "+"rmse for {} data = {}".format(stage,np.sqrt(mean_squared_error(y_df_flat_masked,y_hat_arr_masked))))
            # print(smooth_pred+" "+"mape for {} data = {}".format(stage,np.sqrt(mean_absolute_percentage_error(y_df_flat_masked,y_hat_arr_masked)))) 
            print("rms prediction error  for {} data = {}".format(stage,np.sqrt(mean_squared_error(vt_arr,np.zeros_like(vt_arr)))))
        else:
            y_df_flat_masked=y_df_flat
            y_hat_arr_masked=y_hat_arr
#            print(smooth_pred+" "+"rmse for {} data = {}".format(stage,np.sqrt(mean_squared_error(y_df_flat_masked,y_hat_arr_masked))))
            print("rms prediction error  for {} data = {}".format(stage,np.sqrt(mean_squared_error(vt_arr,np.zeros_like(vt_arr)))))
            # print(smooth_pred+" "+"mape for {} data = {}".format(stage,np.sqrt(mean_absolute_percentage_error(y_df_flat_masked,y_hat_arr_masked))))     
        #     y_mask=fitgapind #list(range(valgapstart,valgapstart+valgapsize))
        #     # print(y_mask)
        #     # print(y_df_flat[y_mask])
        #     y_df_flat_masked=np.delete(y_df_flat,y_mask)
        #     y_hat_arr_masked=np.delete(y_hat_arr,y_mask)
        #     y_mask=valgapindoff # for plotting only
        # else:
        #     y_df_flat_masked=y_df_flat
        #     y_hat_arr_masked=y_hat_arr 

        # print("rmse for {} data = {}".format(stage,np.sqrt(mean_squared_error(y_df_flat_masked,y_hat_arr_masked))))
        # print("mape {} data = {}".format(stage,np.sqrt(mean_absolute_percentage_error(y_df_flat_masked,y_hat_arr_masked))))
 




    fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    ax[0].plot(datetime_arr,y_hat_arr,color='violet',alpha=0.8, label=legenlbl)
    ax[0].plot(datetime_arr,y_df_flat,color='r',alpha=0.6, label="raw data")
    if stage=='val' and valintervaleval:
        ax[0].vlines(datetime_arr[y_mask[0]],ymin=0, colors='k', linestyles='dashed',ymax=np.max(y_df_flat))
        ax[0].vlines(datetime_arr[y_mask[-1]],ymin=0, colors='k', linestyles='dashed',ymax=np.max(y_df_flat))
    ax[0].legend(loc='best')


    # ax[1].plot(datetime_arr,y_pred_arr,color='r',alpha=0.6, label="prediction error")
    # # if stage=='val' and valintervaleval:
    # #     ax[1].vlines(datetime_arr[y_mask[0]],ymin=0, colors='k', linestyles='dashed',ymax=np.max(y_df_flat))
    # #     ax[1].vlines(datetime_arr[y_mask[-1]],ymin=0, colors='k', linestyles='dashed',ymax=np.max(y_df_flat))
    # ax[1].legend(loc='best')

    ax[1].plot(datetime_arr,stdpos_arr,color='violet',alpha=0.8, label="upper bound")
    ax[1].plot(datetime_arr,stdneg_arr,color='cyan',alpha=0.7, label="lower bound")
    ax[1].plot(datetime_arr,y_df_flat,color='r',alpha=0.6, label="raw data")
    if stage=='val' and valintervaleval:
        ax[1].vlines(datetime_arr[y_mask[0]],ymin=0, colors='k', linestyles='dashed',ymax=np.max(y_df_flat))
        ax[1].vlines(datetime_arr[y_mask[-1]],ymin=0, colors='k', linestyles='dashed',ymax=np.max(y_df_flat))
    ax[1].legend(loc='best')


    #plotsfile="KF"+"_"+mean_or_sum+"_"+stage+"_"+smooth_pred+"_"+model+"_"+"noise_bound_div"+"_"+str(noise_bound_divisor)+"_"+firstdate+"_"+lastdate+'.pdf'
    plotsfile="KF"+"_"+mean_or_sum+"_"+stage+"_"+smooth_pred+"_"+model+"_"+firstdate+"_"+lastdate+'.pdf'
    fig.savefig(plotsdir / plotsfile)
    plt.close(fig)

    # y_mask=range(gapstart,gapstart+gapsize)
    # y_df_flat_masked=np.delete(y_df_flat,y_mask)
    # y_hat_arr_masked=np.delete(y_hat_arr,y_mask)

    # print("rmse for {} data = {}".format(stage,np.sqrt(mean_squared_error(y_df_flat_masked,y_hat_arr_masked))))
    # print("mape {} data = {}".format(stage,np.sqrt(mean_absolute_percentage_error(y_df_flat_masked,y_hat_arr_masked))))

    

