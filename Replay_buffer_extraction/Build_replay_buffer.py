### Hypothesis of dataset that I want
### (Trajectory ID, Time (Timestep in trajectory), state, action, next state, terminal, reward1, reward2, reward3)

### Still open: replicate illegal intermediate rewards of paper within endstage reward
### Still open: Engineer own intermediate rewards based on SOFA score and lactate values

## Adaption of AI_clinician code to build replay buffer for own offline learning application in particular 
## 1. Perform state discretization using Kmeans as done here, do that once for entire dataset to get states     
##    (Follow input space, Kmeans specification, but apply to full dataset once)
## 2. Derive actions (vasopressor dosing)     
##    (Follow work done here, understand whether there is one or 2 action variables)
## 3. Derive reward (intermediates created by me and end stage)     
##    (Take inspiration from end stage rewards specified here, specify intermediate rewards based on tabular data SOFA score and test)
## 3. Populate next state information      
##    (See how this is done here although, treansition matrix approach used may not be optimal)


##(if nothing else works, we could pluck our Pareto-Q-learning algorithm into this scheme)

import pickle 
import numpy as np 
import pandas as pd 
from scipy.stats import zscore, rankdata
import math 
import scipy.io as sio 
import datetime 
from scipy.stats.mstats import mquantiles
from mdptoolbox.mdp import PolicyIteration
from reinforcement_learning_mp import offpolicy_multiple_eval_010518
from kmeans_mp import kmeans_with_multiple_runs 
from multiprocessing import freeze_support 

def my_zscore(x):
    return zscore(x,ddof=1),np.mean(x,axis=0),np.std(x,axis=0,ddof=1)


# In[ ]:


######### Functions used in Reinforcement Learning ######## 


class PolicyIteration_with_Q(PolicyIteration):
    def __init__(self, transitions, reward, discount, policy0=None,max_iter=1000, eval_type=0, skip_check=False):
        # Python MDP toolbox from https://github.com/sawcordwell/pymdptoolbox
        # In Matlab MDP Toolbox, P = (S, S, A), R = (S, A) 
        # In Python MDP Toolbox, P = (A, S, S), R= (S, A)

        transitions = np.transpose(transitions,(2,0,1)).copy() # Change to Action First (A, S, S)
        skip_check = True # To Avoid StochasticError: 'PyMDPToolbox - The transition probability matrix is not stochastic.'

        PolicyIteration.__init__(self, transitions, reward, discount, policy0=None,max_iter=1000, eval_type=0, skip_check=skip_check)
    
    def _bellmanOperator_with_Q(self, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, Q, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.S,), (1, self.S)), "V is not the "                     "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = np.empty((self.A, self.S))
        for aa in range(self.A):
            Q[aa] = self.R[aa] + self.discount * self.P[aa].dot(V)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        return (Q.argmax(axis=0), Q, Q.max(axis=0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)
    
    def run(self):
        # Run the policy iteration algorithm.
        self._startRun()

        while True:
            self.iter += 1
            # these _evalPolicy* functions will update the classes value
            # attribute
            if self.eval_type == "matrix":
                self._evalPolicyMatrix()
            elif self.eval_type == "iterative":
                self._evalPolicyIterative()
            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, Q, null = self._bellmanOperator_with_Q()
            del null
            # calculate in how many places does the old policy disagree with
            # the new policy
            n_different = (policy_next != self.policy).sum()
            # if verbose then continue printing a table
            if self.verbose:
                _printVerbosity(self.iter, n_different)
            # Once the policy is unchanging of the maximum number of
            # of iterations has been reached then stop
            if n_different == 0:
                if self.verbose:
                    print(_MSG_STOP_UNCHANGING_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break
            elif self.iter > 20 and n_different <=5 : # This condition was added from the Nature Code 
                if self.verbose: 
                    print((_MSG_STOP))
                break 
            else:
                self.policy = policy_next

        self._endRun()
        
        return Q 
    


# In[ ]:


if __name__ == '__main__': 
    
    freeze_support()
    
    # To ignore 'Runtime Warning: Invalid value encountered in greater' caused by NaN 
    np.warnings.filterwarnings('ignore')

    # Load pickle 
    with open('step_4_start.pkl', 'rb') as file:
        MIMICtable = pickle.load(file)
    
    print(type(MIMICtable))
    print(MIMICtable.columns.get_loc("Arterial_lactate"))
    #############################  MODEL PARAMETERS   #####################################

    print('####  INITIALISATION  ####') 

    nr_reps=500               # nr of repetitions (total nr models) % 500 
    nclustering=32            # how many times we do clustering (best solution will be chosen) % 32
    prop=0.25                 # proportion of the data we sample for clustering
    gamma=0.99                # gamma
    transthres=5              # threshold for pruning the transition matrix
    polkeep=1                 # count of saved policies
    ncl=750                   # nr of states
    nra=5                     # nr of actions (2 to 10)
    ncv=5                     # nr of crossvalidation runs (each is 80% training / 20% test)
    OA=np.full((752,nr_reps),np.nan)       # record of optimal actions
    recqvi=np.full((nr_reps*2,30),np.nan)  # saves data about each model (1 row per model)
    # allpols=[]  # saving best candidate models
    
    
    # #################   Convert training data and compute conversion factors    ######################

    # all 47 columns of interest
    colbin = ['gender','mechvent','max_dose_vaso','re_admission'] 
    colnorm= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',        'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',        'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',        'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance'] 
    collog=['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly']

    colbin=np.where(np.isin(MIMICtable.columns,colbin))[0]
    colnorm=np.where(np.isin(MIMICtable.columns,colnorm))[0]
    collog=np.where(np.isin(MIMICtable.columns,collog))[0]

    # find patients who died in ICU during data collection period
    # ii=MIMICtable.bloc==1&MIMICtable.died_within_48h_of_out_time==1& MIMICtable.delay_end_of_record_and_discharge_or_death<24;
    # icustayidlist=MIMICtable.icustayid;
    # ikeep=~ismember(icustayidlist,MIMICtable.icustayid(ii));
    reformat5=MIMICtable.values.copy() 
    # reformat5=reformat5(ikeep,:);
    icustayidlist=MIMICtable['icustayid']
    icuuniqueids=np.unique(icustayidlist) # list of unique icustayids from MIMIC
    idxs=np.full((icustayidlist.shape[0],nr_reps),np.nan) # record state membership test cohort

    MIMICraw=MIMICtable.iloc[:, np.concatenate([colbin,colnorm,collog])] 
    MIMICraw=MIMICraw.values.copy()  # RAW values
    MIMICzs=np.concatenate([reformat5[:, colbin]-0.5, zscore(reformat5[:,colnorm],ddof=1), zscore(np.log(0.1+reformat5[:, collog]),ddof=1)],axis=1)
    MIMICzs[:,3]=np.log(MIMICzs[:,3]+0.6)   # MAX DOSE NORAD 
    MIMICzs[:,44]=2*MIMICzs[:,44]   # increase weight of this variable


    # eICU section was not implemented 
    
    # compute conversion factors using MIMIC data
    a=MIMICraw[:, 0:3]-0.5 
    b= np.log(MIMICraw[:,3]+0.1)
    c,cmu,csigma = my_zscore(MIMICraw[:,4:36])
    d,dmu,dsigma = my_zscore(np.log(0.1+MIMICraw[:,36:47]))
    

    ####################### Main LOOP ###########################
    bestpol = 0 
    
    N=icuuniqueids.size # total number of rows to choose from
    grp=np.floor(ncv*np.random.rand(N,1)+1);  #list of 1 to 5 (20% of the data in each grp) -- this means that train/test MIMIC split are DIFFERENT in all the 500 models
    crossval=1;
    trainidx=icuuniqueids[np.where(grp!=crossval)[0]]
    testidx=icuuniqueids[np.where(grp==crossval)[0]]
    train=np.isin(icustayidlist,trainidx)
    test=np.isin(icustayidlist,testidx)
    X=MIMICzs[train,:]
    Xtestmimic=MIMICzs[~train,:]
    print(MIMICtable.head(30))
    blocs=reformat5[train,0]
    bloctestmimic=reformat5[~train,0]
    ptid=reformat5[train,1]
    ptidtestmimic=reformat5[~train,1] 
    outcome=9 #   HOSP _ MORTALITY = 7 / 90d MORTA = 9
    Y90=reformat5[train,outcome];   

    MIMIC_processing=MIMICtable.values.copy() 

    print("Is this SOFA?")
    print(MIMIC_processing[:20,57])

    print('########################   MODEL NUMBER : ',0)
    print(datetime.datetime.now())

    #######   find best clustering solution (lowest intracluster variability)  ####################

    ########################################
    ##### Perform state discretization #####
    ########################################

    print('####  CLUSTERING  ####') # BY SAMPLING
    N=X.shape[0] #total number of rows to choose from
    sampl=X[np.where(np.floor(np.random.rand(N,1)+prop))[0],:] #Sample 25% of data to define clusters

    C = kmeans_with_multiple_runs(ncl,10000,nclustering,sampl) #create clustering model
    
    # HERE
    #The arrays we want to reintegrate into the dataset
    idx = C.predict(X) #create clusters
    idxtest = C.predict(Xtestmimic) #create test-clusters

    #STATES IS THE VARIABLE WE WANT TO APPEND TO THE MIMIC TABLE
    #create one array of test and train for the state
    states = np.zeros(MIMICtable.shape[0], dtype='int16')
    states[train]=idx.ravel()
    states[test] = idxtest.ravel()
    print(states.shape)
    print(states[:10]) 


    #Not sure whether this is still useful for my purposes
    idxs[test,0]=idxtest.ravel()  #important: record state membership of test cohort

    
    print(type(C))
    print(idx[:10])
    print(idx.shape)
    print(X.shape)
    print(idxtest[:10])
    print(idxtest.shape)
    print(Xtestmimic.shape)
    print(MIMICraw.shape)
    print(idxs[-5:,:5]) 
    #print(idx.columns)

    ########################################
    ######## Extract action arrays #########
    ########################################
    #Bin fluid and vasopressor choice into 5 bins and code these as actions
    #actions are combination of choice of vaso-bin and intervenous fluid bin

    ############################## CREATE ACTIONS  ########################
    print('####  CREATE ACTIONS  ####') 

    nact=nra*nra

    iol=MIMICtable.columns.get_loc('input_4hourly') 
    vcl=MIMICtable.columns.get_loc('max_dose_vaso') 

    #Bin fluids
    a= reformat5[:,iol].copy() # IV fluid column
    a= rankdata(a[a>0])/a[a>0].shape[0]   # excludes zero fluid (will be action 1),
    iof=np.floor((a+0.2499999999)*4)  #converts iv volume in 4 actions

    a= reformat5[:,iol].copy() 
    a= np.where(a>0)[0]  # location of non-zero fluid in big matrix

    io=np.ones((reformat5.shape[0],1))  # array of ones, by default     
    io[a]=(iof+1).reshape(-1,1)   # where more than zero fluid given: save actual action
    io = io.ravel() 

    #Bin vaso
    vc=reformat5[:,vcl].copy() 
    vcr= rankdata(vc[vc!=0])/vc[vc!=0].size
    vcr=np.floor((vcr+0.249999999999)*4)  # converts to 4 bins
    vcr[vcr==0]=1
    vc[vc!=0]=vcr+1 
    vc[vc==0]=1

    ma1 = np.array([np.median(reformat5[io==1,iol]),np.median(reformat5[io==2,iol]),np.median(reformat5[io==3,iol]), np.median(reformat5[io==4,iol]),np.median(reformat5[io==5,iol])]) # median dose of drug in all bins
    ma2 = np.array([np.median(reformat5[vc==1,vcl]),np.median(reformat5[vc==2,vcl]),np.median(reformat5[vc==3,vcl]), np.median(reformat5[vc==4,vcl]),np.median(reformat5[vc==5,vcl])])

    med = np.concatenate([io.reshape(-1,1),vc.reshape(-1,1)],axis=1)
    print(med.shape)
    print(med[:30])

    #ACTIONBLOCK IS THE ARRAY WE WANT TO ADD TO OUR REPLAY BUFFER DATASET
    uniqueValues,actionbloc = np.unique(med,axis=0,return_inverse=True)

    print(actionbloc.shape)
    print(actionbloc[:30])

    actionbloctrain=actionbloc[train] 

    ma2Values = ma2[uniqueValues[:,1].astype('int64')-1].reshape(-1,1)
    ma1Values = ma1[uniqueValues[:,0].astype('int64')-1].reshape(-1,1)

    uniqueValuesdose = np.concatenate([ma2Values,ma1Values],axis=1) # median dose of each bin for all 25 actions 
    print(uniqueValuesdose)
    print(len(uniqueValuesdose))

    #########################################
    ####### Start working on rewards ########
    #########################################

    print('####  CREATE QLDATA3  ####')

    r=np.array([100, -100]).reshape(1,-1)
    r2=r*(2*(1-Y90.reshape(-1,1))-1)
    # because idx and actionbloctrain are index, it's equal to (Matlab's original value -1)
    qldata=np.concatenate([blocs.reshape(-1,1), idx.reshape(-1,1), actionbloctrain.reshape(-1,1), Y90.reshape(-1,1), r2],axis=1)  # contains bloc / state / action / outcome / reward     
    # 0 = died in Python, 1 = died in Matlab 
    qldata3=np.zeros((np.floor(qldata.shape[0]*1.2).astype('int64'),4))
    c=-1
    abss=np.array([ncl+1, ncl]) #absorbing states numbers # 751, 750 

    print(qldata.shape)
    print(qldata[:10])
    print(qldata3.shape)
    print(qldata3[:10])

    for i in range(qldata.shape[0]-1):
        c=c+1
        qldata3[c,:]=qldata[i,0:4] 
        if(qldata[i+1,0]==1): #end of trace for this patient
            c=c+1     
            qldata3[c,:]=np.array([qldata[i,0]+1, abss[int(qldata[i,3])], -1, qldata[i,4]]) # we add a terminal state observation here

    qldata3=qldata3[:c+1,:]

    print(qldata3.shape)
    print(qldata3[:40])

    # ###################################################################################################################################
    print("####  CREATE TRANSITION MATRIX T(S'',S,A) ####")
    transitionr=np.zeros((ncl+2,ncl+2,nact))  #this is T(S',S,A)
    sums0a0=np.zeros((ncl+2,nact)) 

    for i in range(qldata3.shape[0]-1):    
        if (qldata3[i+1,0]!=1) : # if we are not in the last state for this patient = if there is a transition to make!
            S0=int(qldata3[i,1]) 
            S1=int(qldata3[i+1,1])
            acid= int(qldata3[i,2]) 
            transitionr[S1,S0,acid]=transitionr[S1,S0,acid]+1 
            sums0a0[S0,acid]=sums0a0[S0,acid]+1

    sums0a0[sums0a0<=transthres]=0  #delete rare transitions (those seen less than 5 times = bottom 50%!!)

    for i in range(ncl+2): 
        for j in range(nact): 
            if sums0a0[i,j]==0: 
                transitionr[:,i,j]=0; 
            else:
                transitionr[:,i,j]=transitionr[:,i,j]/sums0a0[i,j]


    transitionr[np.isnan(transitionr)]=0  #replace NANs with zeros
    transitionr[np.isinf(transitionr)]=0  #replace NANs with zeros

    physpol=sums0a0/np.sum(sums0a0, axis=1).reshape(-1,1)    #physicians policy: what action was chosen in each state

    print("####  CREATE TRANSITION MATRIX T(S,S'',A)  ####")

    transitionr2=np.zeros((ncl+2,ncl+2,nact))  # this is T(S,S',A)
    sums0a0=np.zeros((ncl+2,nact))

    for i in range(qldata3.shape[0]-1) : 
        if (qldata3[i+1,0]!=1) : # if we are not in the last state for this patient = if there is a transition to make!
            S0=int(qldata3[i,1])
            S1=int(qldata3[i+1,1])
            acid= int(qldata3[i,2]) 
            transitionr2[S0,S1,acid]=transitionr2[S0,S1,acid]+1;  
            sums0a0[S0,acid]=sums0a0[S0,acid]+1

    sums0a0[sums0a0<=transthres]=0;  #delete rare transitions (those seen less than 5 times = bottom 50%!!) IQR = 2-17

    for i in range(ncl+2): 
        for j in range(nact): 
            if sums0a0[i,j]==0:
                transitionr2[i,:,j]=0 
            else: 
                transitionr2[i,:,j]=transitionr2[i,:,j]/sums0a0[i,j]

    transitionr2[np.isnan(transitionr2)]=0 #replace NANs with zeros
    transitionr2[np.isinf(transitionr2)]=0 # replace infs with zeros

    print('####  CREATE REWARD MATRIX  R(S,A) ####')
    # CF sutton& barto bottom 1998 page 106. i compute R(S,A) from R(S'SA) and T(S'SA)
    r3=np.zeros((ncl+2,ncl+2,nact))
    r3[ncl,:,:]=-100
    r3[ncl+1,:,:]=100
    R=sum(transitionr*r3)
    R=np.squeeze(R)   #remove 1 unused dimension

    print("CALCULATED REWARDS SOURCE CODE")
    print(R.shape)
    print(R[:5])

    #########################################
    ######### Build and save SAS'TR #########
    #########################################
    
    #Use similar logic to get next state indicator, if state is terminal next state will be 9999, these 9999 transitions may be deleted in later dataset
    #derive terminal state indicator
    terminal = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        else:
            if (MIMIC_processing[i+1,0] == 1):
                terminal[i] = 1
    
    print(terminal.shape)
    print(terminal[:15])

    #try coding up sparse reward myself
    sparse_90dmort_reward = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if terminal[i] == 1:
            if MIMIC_processing[i,9]==1:
                sparse_90dmort_reward[i] = -100
            elif MIMIC_processing[i,9]==0:
                sparse_90dmort_reward[i] = 100

    print(sparse_90dmort_reward.shape)
    print(sparse_90dmort_reward[:20])

    #code up next state indicator incl. absorbing states for 90d death / survival, 
    # this is the next state of our terminal observation
    next_state_array = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+1,0] != 1):
                next_state_array[i] = states[i+1]
            else:
                if sparse_90dmort_reward[i]==-100:
                    next_state_array[i] = ncl
                elif sparse_90dmort_reward[i]==100:
                    next_state_array[i] = ncl+1
    
    print(next_state_array.shape)
    print(next_state_array[:20])
    print(terminal.shape)
    print(terminal[:20])

    #Reward based on reward matrix calculated in paper (be carful in using) 
    reward_paper = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        else:
            state = states[i]
            action = actionbloc[i]
            reward_paper[i] = R[state,action]
    
    print(reward_paper.shape)
    print(reward_paper[:40])
    print(terminal.shape)
    print(terminal[:40])

    ####Create intermediate reward 1: One period change in SOFA Score
    reward_SOFA_1_continuous = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+1,0] != 1):
                curr_SOFA = MIMIC_processing[i,57]
                next_SOFA = MIMIC_processing[i+1,57]
                SOFA_change = -1*(next_SOFA-curr_SOFA) 
                reward_SOFA_1_continuous[i] = SOFA_change
            else:
                if sparse_90dmort_reward[i]==-100:
                    reward_SOFA_1_continuous[i] = -1
                elif sparse_90dmort_reward[i]==100:
                    reward_SOFA_1_continuous[i] = 1
    
    print(reward_SOFA_1_continuous.shape)
    print(reward_SOFA_1_continuous[:20])
    print(terminal.shape)
    print(terminal[:20])

    ####Create intermediate reward 2: One period change in SOFA Score binary
    reward_SOFA_1_binary = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+1,0] != 1):
                curr_SOFA = MIMIC_processing[i,57]
                next_SOFA = MIMIC_processing[i+1,57]
                SOFA_change = -1*(next_SOFA-curr_SOFA) 
                if SOFA_change < 0:
                    reward_SOFA_1_binary[i] = -1
                else:
                    reward_SOFA_1_binary[i] = 0
            else:
                if sparse_90dmort_reward[i]==-100:
                    reward_SOFA_1_binary[i] = -1
                elif sparse_90dmort_reward[i]==100:
                    reward_SOFA_1_binary[i] = 1
    
    print(reward_SOFA_1_binary.shape)
    print(reward_SOFA_1_binary[:20])
    print(terminal.shape)
    print(terminal[:20])

    ####Create intermediate reward 3: two period change in SOFA Score
    reward_SOFA_2_continuous = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        elif (i+2) > (MIMIC_processing.shape[0]-1):
            if (sparse_90dmort_reward[i]==-100) or (sparse_90dmort_reward[i+1]==-100):
                reward_SOFA_2_continuous[i] = -1
            elif (sparse_90dmort_reward[i]==100) or (sparse_90dmort_reward[i+1]==100):
                reward_SOFA_2_continuous[i] = 1
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+2,0] != 1):
                curr_SOFA = MIMIC_processing[i,57]
                next_SOFA = MIMIC_processing[i+2,57]
                SOFA_change = -1*(next_SOFA-curr_SOFA) 
                reward_SOFA_2_continuous[i] = SOFA_change
            else:
                if (sparse_90dmort_reward[i]==-100) or (sparse_90dmort_reward[i+1]==-100):
                    reward_SOFA_2_continuous[i] = -1
                elif (sparse_90dmort_reward[i]==100) or (sparse_90dmort_reward[i+1]==100):
                    reward_SOFA_2_continuous[i] = 1
    
    print(reward_SOFA_2_continuous.shape)
    print(reward_SOFA_2_continuous[:20])
    print(terminal.shape)
    print(terminal[:20])

    ####Create intermediate reward 4: Two period change in SOFA Score binary
    reward_SOFA_2_binary = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        elif (i+2) > (MIMIC_processing.shape[0]-1):
            if (sparse_90dmort_reward[i]==-100) or (sparse_90dmort_reward[i+1]==-100):
                reward_SOFA_2_binary[i] = -1
            elif (sparse_90dmort_reward[i]==100) or (sparse_90dmort_reward[i+1]==100):
                reward_SOFA_2_binary[i] = 1
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+2,0] != 1):
                curr_SOFA = MIMIC_processing[i,57]
                next_SOFA = MIMIC_processing[i+2,57]
                SOFA_change = -1*(next_SOFA-curr_SOFA) 
                if SOFA_change < 0:
                    reward_SOFA_2_binary[i] = -1
                else:
                    reward_SOFA_2_binary[i] = 0
            else:
                if (sparse_90dmort_reward[i]==-100) or (sparse_90dmort_reward[i+1]==-100):
                    reward_SOFA_2_binary[i] = -1
                elif (sparse_90dmort_reward[i]==100) or (sparse_90dmort_reward[i+1]==100):
                    reward_SOFA_2_binary[i] = 1
    
    print(reward_SOFA_2_binary.shape)
    print(reward_SOFA_2_binary[:20])
    print(terminal.shape)
    print(terminal[:20])

    ####Create intermediate reward 5: One period change in SOFA Score more than 2 binary
    reward_SOFA_change2_binary = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+1,0] != 1):
                curr_SOFA = MIMIC_processing[i,57]
                next_SOFA = MIMIC_processing[i+1,57]
                SOFA_change = -1*(next_SOFA-curr_SOFA) 
                if SOFA_change <= -2:
                    reward_SOFA_change2_binary[i] = -1
                else:
                    reward_SOFA_change2_binary[i] = 0
            else:
                if sparse_90dmort_reward[i]==-100:
                    reward_SOFA_change2_binary[i] = -1
                elif sparse_90dmort_reward[i]==100:
                    reward_SOFA_change2_binary[i] = 0
    
    print(reward_SOFA_change2_binary.shape)
    print(reward_SOFA_change2_binary[:20])
    print(terminal.shape)
    print(terminal[:20])

    ####Create intermediate reward 6: One period change in lactate levels continous
    reward_lactat_1_continous = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+1,0] != 1):
                curr_lac = MIMIC_processing[i,45]
                next_lac = MIMIC_processing[i+1,45]
                lac_change = -1*(next_lac-curr_lac) 
                reward_lactat_1_continous[i] = lac_change
            else:
                if sparse_90dmort_reward[i]==-100:
                    reward_lactat_1_continous[i] = -1
                elif sparse_90dmort_reward[i]==100:
                    reward_lactat_1_continous[i] = 1
    
    print(reward_lactat_1_continous.shape)
    print(reward_lactat_1_continous[:20])
    print(terminal.shape)
    print(terminal[:20])

    ####Create intermediate reward 7: One period change in lactate levels binary
    reward_lactat_1_binary = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+1,0] != 1):
                curr_lac = MIMIC_processing[i,45]
                next_lac = MIMIC_processing[i+1,45]
                lac_change = -1*(next_lac-curr_lac) 
                if lac_change < 0:
                    reward_lactat_1_binary[i] = -1
                else:
                    reward_lactat_1_binary[i] = 0
            else:
                if sparse_90dmort_reward[i]==-100:
                    reward_lactat_1_binary[i] = -1
                elif sparse_90dmort_reward[i]==100:
                    reward_lactat_1_binary[i] = 1
    
    print(reward_lactat_1_binary.shape)
    print(reward_lactat_1_binary[:20])
    print(terminal.shape)
    print(terminal[:20])

    ####Create intermediate reward 8: two period change in Lactate levels
    reward_lactat_2_continuous = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        elif (i+2) > (MIMIC_processing.shape[0]-1):
            if (sparse_90dmort_reward[i]==-100) or (sparse_90dmort_reward[i+1]==-100):
                reward_lactat_2_continuous[i] = -1
            elif (sparse_90dmort_reward[i]==100) or (sparse_90dmort_reward[i+1]==100):
                reward_lactat_2_continuous[i] = 1
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+2,0] != 1):
                curr_lac = MIMIC_processing[i,45]
                next_lac = MIMIC_processing[i+2,45]
                Lac_change = -1*(next_lac-curr_lac) 
                reward_lactat_2_continuous[i] = Lac_change
            else:
                if (sparse_90dmort_reward[i]==-100) or (sparse_90dmort_reward[i+1]==-100):
                    reward_lactat_2_continuous[i] = -1
                elif (sparse_90dmort_reward[i]==100) or (sparse_90dmort_reward[i+1]==100):
                    reward_lactat_2_continuous[i] = 1
    
    print(reward_lactat_2_continuous.shape)
    print(reward_lactat_2_continuous[:20])
    print(terminal.shape)
    print(terminal[:20])

    ####Create intermediate reward 9: Two period change in Lactate levels binary
    reward_lactate_2_binary = np.zeros(MIMIC_processing.shape[0])
    for i in range(MIMIC_processing.shape[0]-1):
        if (i+1) > (MIMIC_processing.shape[0]-1):
            pass
        elif (i+2) > (MIMIC_processing.shape[0]-1):
            if (sparse_90dmort_reward[i]==-100) or (sparse_90dmort_reward[i+1]==-100):
                reward_lactate_2_binary[i] = -1
            elif (sparse_90dmort_reward[i]==100) or (sparse_90dmort_reward[i+1]==100):
                reward_lactate_2_binary[i] = 1
        else:
            #print(MIMIC_processing[i+1,0])
            if (MIMIC_processing[i+2,0] != 1):
                curr_lac = MIMIC_processing[i,45]
                next_lac = MIMIC_processing[i+2,45]
                Lac_change = -1*(next_lac-curr_lac) 
                if Lac_change < 0:
                    reward_lactate_2_binary[i] = -1
                else:
                    reward_lactate_2_binary[i] = 0
            else:
                if (sparse_90dmort_reward[i]==-100) or (sparse_90dmort_reward[i+1]==-100):
                    reward_lactate_2_binary[i] = -1
                elif (sparse_90dmort_reward[i]==100) or (sparse_90dmort_reward[i+1]==100):
                    reward_lactate_2_binary[i] = 1
    
    print(reward_lactate_2_binary.shape)
    print(reward_lactate_2_binary[:20])
    print(terminal.shape)
    print(terminal[:20])

    #########################################
    ######## Save key traj variables ########
    #########################################

    #CHECK whether a reward that scales reward by severity could be helpful

    ## Key arrays mainly taken over from original paper
    MIMICtable['state'] = states
    MIMICtable['action'] = actionbloc
    MIMICtable['next_state'] = next_state_array
    MIMICtable['terminal'] = terminal
    MIMICtable['sparse_90d_rew'] = sparse_90dmort_reward
    MIMICtable['Reward_matrix_paper'] = reward_paper

    ## New rewards
    MIMICtable['Reward_SOFA_1_continous'] = reward_SOFA_1_continuous
    MIMICtable['Reward_SOFA_1_binary'] = reward_SOFA_1_binary
    MIMICtable['Reward_SOFA_2_continous'] = reward_SOFA_2_continuous
    MIMICtable['Reward_SOFA_2_binary'] = reward_SOFA_2_binary
    MIMICtable['Reward_SOFA_change2_binary'] = reward_SOFA_change2_binary

    MIMICtable['Reward_lac_1_continous'] = reward_lactat_1_continous
    MIMICtable['Reward_lac_1_binary'] = reward_lactat_1_binary
    MIMICtable['Reward_lac_2_continous'] = reward_lactat_2_continuous
    MIMICtable['Reward_lac_2_binary'] = reward_lactate_2_binary

    #Save the augmented MIMIC table
    MIMICtable.to_csv('MIMICtable_plus_SanSTR.csv')

    #Create table transition table with current info
    col_to_keep = ["bloc", "icustayid", "charttime","state","action","next_state","terminal",'sparse_90d_rew','Reward_matrix_paper',
    'Reward_SOFA_1_continous','Reward_SOFA_1_binary','Reward_SOFA_2_continous','Reward_SOFA_2_binary','Reward_SOFA_change2_binary',
    'Reward_lac_1_continous', 'Reward_lac_1_binary', 'Reward_lac_2_continous', 'Reward_lac_2_binary']
    MIMICtable_transitions = MIMICtable[col_to_keep]

    MIMICtable_transitions.to_csv('MIMICtable_transitions.csv')

    print(MIMICtable_transitions.head(5))
