import numpy as np
import pandas as pd
from numba import jit

###############################################################################
#reading data from csv file
df=pd.read_csv('./AAPL.csv')
df=df
df['Upk']=0.0
df['Lowk']=0.0
df['MA']=df['Close'].rolling(window=5).mean()
df['ROC']=df['Close'].pct_change(periods=5)
df=df.drop(['Date'], axis=1)
df=df.values

df_batch=pd.read_csv('./AAPL.csv')
df_batch=df_batch
df_batch['Upk']=0.0
df_batch['Lowk']=0.0
df_batch['MA']=df_batch['Close'].rolling(window=5).mean()
df_batch['ROC']=df_batch['Close'].pct_change(periods=5)
df_batch=df_batch.drop(['Date'], axis=1)
df_batch=df_batch.values

df=df[5:]
df_batch=df_batch[5:]
###############################################################################
window_size=120  #n
n1=15
n2=4*n1
sigma=0.015
epsilon=0.5
theta=0.1
varphi=0.0
phi=0.05
transaction_cost=0.005
num_hidden_layer=9

max_close_price=np.max(df[:,3])
min_close_price=np.min(df[:,3])

###############################################################################
@jit(nopython=True, parallel=True)
def datapreprocessing(df):
    datalength=len(df)

    for i in range(datalength-n1):
        df[i][5]=np.max(df[i:(i+n1),3])
        df[i][6]=np.min(df[i:(i+n1),3])

    for j in range(len(df[0])):
        df_max=np.max(df[:,j])
        df_min=np.min(df[:,j])
        for i in range(datalength):        
            df[i,j]=-1.0+2.0*(df[i,j]-df_min)/(df_max-df_min)

    return df
###############################################################################
df=datapreprocessing(df)

weight_list=np.zeros(len(df[0]))
for i in range(len(df[0])):
    weight_list[i]=np.corrcoef(df[:,i],df[:,3])[0,1]
weight_list=weight_list/sum(weight_list)

###############################################################################
@jit(nopython=True, parallel=True)
def greyweight(df):
    gwm=np.zeros(df.shape)
    for i in range(len(gwm[0])):
        if i==3:
            gwm[:,i]=1.0
            continue
        maxdiff=np.max(np.abs(df[:,i]-df[:,3]))
        mindiff=np.min(np.abs(df[:,i]-df[:,3]))
        for j in range(len(df)):
            gwm[j,i]=(mindiff+epsilon*maxdiff)/(np.abs(df[j,i]-df[j,3])+epsilon*maxdiff)

    return gwm
###############################################################################
grey_weight_matrix=greyweight(df)

###############################################################################
@jit(nopython=True, parallel=True)
def model_training(ith,df,gwm):
    num_feature=len(df[0])
    T_matrix=np.zeros((n1,window_size-n1+1))
    I_matrix=np.zeros((num_feature,window_size-n1+1))
    I_matrix_weight=np.zeros((num_feature,window_size-n1+1))
    beta_matrix=np.zeros((n1,num_feature))
    
    for i in range(window_size-n1+1):
        for j in range(n1):
            T_matrix[j,i]=df[ith-window_size+i+j,3]
        F_matrix=df[(ith-window_size+i-n2):(ith-window_size+i),:]
        W_matrix=gwm[(ith-window_size+i-n2):(ith-window_size+i),:]
        for j in range(num_feature):
            I_matrix[j,i]=0.0
            I_matrix_weight[j,i]=0.0
            for k in range(n2):
                I_matrix[j,i]+=W_matrix[k,j]*F_matrix[k,j]
                I_matrix_weight[j,i]+=W_matrix[k,j]
            I_matrix[j,i]=I_matrix[j,i]/I_matrix_weight[j,i]
    #print(I_matrix)
    A_matrix=np.random.normal(1.0/num_feature,1.0,(num_hidden_layer,num_feature))
    B_list=np.random.normal(0.0,1.0,(num_hidden_layer,1))

    I_matrix=np.dot(A_matrix,I_matrix)+B_list
    I_matrix=1.0/(1.0+np.exp(-I_matrix))
    #print(I_matrix)
                
    beta_matrix=np.dot(T_matrix,np.linalg.pinv(I_matrix))
    #print(beta_matrix.shape)
    
    I_prediction_list=np.zeros((num_feature,1))
    I_prediction_list_weight=np.zeros((num_feature,1))
    F_matrix=df[(ith-n2):ith,:]
    W_matrix=gwm[(ith-n2):ith,:]
    for j in range(num_feature):
        I_prediction_list[j,0]=0.0
        for k in range(n2):
            I_prediction_list[j,0]+=W_matrix[k,j]*F_matrix[k,j]
            I_prediction_list_weight[j,0]+=W_matrix[k,j]
        I_prediction_list[j,0]=I_prediction_list[j,0]/I_prediction_list_weight[j,0]
    I_prediction_list=np.dot(A_matrix,I_prediction_list)+B_list
    I_prediction_list=1.0/(1.0+np.exp(-I_prediction_list))
    #print(I_prediction_list.shape)
    T_prediction_list=np.dot(beta_matrix,I_prediction_list)

    df_max=max_close_price
    df_min=min_close_price
    for i in range(n1):
        T_prediction_list[i,0]=(T_prediction_list[i,0]*(df_max-df_min)+df_max+df_min)/2.0
    #print(T_prediction_list.shape)
    
    return T_prediction_list
###############################################################################
@jit(nopython=True, parallel=True)    
def trading_strategy(ith,df,gwm,current_action,buy_price,sell_price):
    prediction_list=model_training(ith,df,gwm) #real price
    upithnext=0.0
    lowithnext=0.0
    prediction_list_next=model_training(ith+1,df,gwm) #real price
    upithnext+=np.max(prediction_list_next)/5.0
    lowithnext+=np.min(prediction_list_next)/5.0    
    prediction_list_next=model_training(ith+2,df,gwm) #real price
    upithnext+=np.max(prediction_list_next)/5.0
    lowithnext+=np.min(prediction_list_next)/5.0    
    prediction_list_next=model_training(ith+3,df,gwm) #real price
    upithnext+=np.max(prediction_list_next)/5.0
    lowithnext+=np.min(prediction_list_next)/5.0    
    prediction_list_next=model_training(ith+4,df,gwm) #real price
    upithnext+=np.max(prediction_list_next)/5.0
    lowithnext+=np.min(prediction_list_next)/5.0    
    prediction_list_next=model_training(ith+5,df,gwm) #real price
    upithnext+=np.max(prediction_list_next)/5.0
    lowithnext+=np.min(prediction_list_next)/5.0    
    #print(prediction_list,df_batch[ith,3])
    #print(prediction_list_next)
    #next_action=0
    upith=np.max(prediction_list) 
    lowith=np.min(prediction_list)

    if current_action==1:
        if (np.abs(df_batch[ith,3]-lowith)<=sigma*df_batch[ith,3]) and (lowithnext>=lowith):
            if sell_price-df_batch[ith,3]>=varphi:
                buy_price=df_batch[ith,3]
                next_action=0
            else:
                next_action=current_action
        else:
            next_action=current_action
    else:
        if (np.abs(df_batch[ith,3]-upith)<=sigma*df_batch[ith,3]) and (upithnext<=upith):
            if (df_batch[ith,3]-buy_price)>=varphi or (buy_price-df_batch[ith,3])>=theta*buy_price:
                sell_price=df_batch[ith,3]
                next_action=1
            else:
                next_action=current_action
        else:
            next_action=current_action

    return (next_action,buy_price,sell_price)
###############################################################################
@jit(nopython=True, parallel=True) 
def strategy_test(df,gwm):
    
    current_action=1 #buy
    test_start=2200
    test_end=2400
    buy_price=min_close_price
    sell_price=max_close_price
    portfolio_value=10000.0
    for ith in range(test_start,test_end):
        next_action,buy_price,sell_price=trading_strategy(ith,df,gwm,current_action,buy_price,sell_price)
        if current_action==0 and next_action==1:
            portfolio_value+=portfolio_value*(sell_price-buy_price-transaction_cost*sell_price)/buy_price
            current_action=next_action
            print('period',ith,'trading','profit:',(sell_price-buy_price-transaction_cost*sell_price)/buy_price)
        else:
            current_action=next_action
            portfolio_value=portfolio_value
        #print("current value",portfolio_value)
    if next_action==0:
        portfolio_value+=portfolio_value*(df_batch[test_end-1,3]-buy_price-transaction_cost*df_batch[test_end-1,3])/buy_price
    
    print('Period price change',(df_batch[test_end,3]-df_batch[test_start,3])/df_batch[test_start,3])
    
    return portfolio_value  
###############################################################################
@jit(nopython=True, parallel=True)
def error_test(df,gwm):
    max_error_rate=0.0
    mean_error_rate=0.0
    test_start=200
    test_end=3000 
    for ith in range(test_start,test_end):
        prediction=model_training(ith,df,gwm)
        #print(prediction[:,0])
        actual=df_batch[(ith+1):(ith+n1+1),3]
        #print(actual)
        max_error_rate+=(np.max(np.abs(prediction[:,0]-actual)/np.mean(actual)))/float(test_end-test_start)
        mean_error_rate+=(np.mean(np.abs(prediction[:,0]-actual)/np.mean(actual)))**2.0/float(test_end-test_start)
    return (max_error_rate,mean_error_rate)
###############################################################################    
final_value=strategy_test(df,grey_weight_matrix)
print(final_value)
max_error_rate,mean_error_rate=error_test(df,grey_weight_matrix)
print(max_error_rate,mean_error_rate)



