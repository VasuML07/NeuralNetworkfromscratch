import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_curve

# =========================
# ACTIVATIONS
# =========================
def sigmoid(Z): return 1/(1+np.exp(-Z))
def relu(Z): return np.maximum(0,Z)
def relu_deriv(Z): return (Z>0).astype(float)

# =========================
# INIT
# =========================
def initialize(layers):
    params={}
    for l in range(1,len(layers)):
        params[f"W{l}"]=np.random.randn(layers[l],layers[l-1])*np.sqrt(2/layers[l-1])
        params[f"B{l}"]=np.zeros((layers[l],1))
    return params

# =========================
# FORWARD
# =========================
def forward(X,params):
    cache={}
    A=X
    L=len(params)//2
    for l in range(1,L):
        Z=np.dot(params[f"W{l}"],A)+params[f"B{l}"]
        A=relu(Z)
        cache[f"Z{l}"]=Z
        cache[f"A{l}"]=A
    ZL=np.dot(params[f"W{L}"],A)+params[f"B{L}"]
    AL=sigmoid(ZL)
    cache[f"A{L}"]=AL
    return AL,cache

# =========================
# LOSS
# =========================
def compute_loss(AL,Y):
    eps=1e-15
    return -np.mean(Y*np.log(AL+eps)+(1-Y)*np.log(1-AL+eps))

# =========================
# BACKPROP
# =========================
def backward(X,Y,params,cache):
    grads={}
    m=X.shape[1]
    L=len(params)//2
    dZ=cache[f"A{L}"]-Y
    for l in reversed(range(1,L+1)):
        A_prev=X if l==1 else cache[f"A{l-1}"]
        grads[f"dW{l}"]=(1/m)*np.dot(dZ,A_prev.T)
        grads[f"dB{l}"]=(1/m)*np.sum(dZ,axis=1,keepdims=True)
        if l>1:
            dA=np.dot(params[f"W{l}"].T,dZ)
            dZ=dA*relu_deriv(cache[f"Z{l-1}"])
    return grads

# =========================
# ADAM
# =========================
def update(params,grads,opt,lr):
    beta,beta2,eps=0.9,0.999,1e-8
    for k in params:
        g=grads["d"+k]
        v=opt.get("v_"+k,0)
        s=opt.get("s_"+k,0)
        v=beta*v+(1-beta)*g
        s=beta2*s+(1-beta2)*(g**2)
        params[k]-=lr*(v/(1-beta))/(np.sqrt(s/(1-beta2))+eps)
        opt["v_"+k]=v
        opt["s_"+k]=s
    return params,opt

# =========================
# TRAIN
# =========================
def train(X,Y,X_val,Y_val,layers,epochs=30,lr=0.001):
    params=initialize(layers)
    opt={}
    history={"train_loss":[],"val_loss":[]}

    for epoch in range(epochs):
        AL,cache=forward(X,params)
        grads=backward(X,Y,params,cache)
        params,opt=update(params,grads,opt,lr)

        train_loss=compute_loss(AL,Y)
        val_pred,_=forward(X_val,params)
        val_loss=compute_loss(val_pred,Y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    return params,history

# =========================
# PREDICT
# =========================
def predict(X,params,threshold=0.5):
    probs,_=forward(X,params)
    return (probs>threshold).astype(int),probs

# =========================
# THRESHOLD
# =========================
def best_threshold(y,probs):
    fpr,tpr,th=roc_curve(y,probs)
    return th[np.argmax(tpr-fpr)]

# =========================
# PLOT
# =========================
def plot_history(history,save_path):
    plt.figure()
    plt.plot(history["train_loss"],label="Train")
    plt.plot(history["val_loss"],label="Val")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(save_path)
    plt.close()

# =========================
# EXPERIMENT RUNNER
# =========================
def run_experiments(X_train,y_train,X_val,y_val,X_test,y_test):

    input_size = X_train.shape[0]   # 🔥 FIX

    configs=[
        {"lr":0.001,"layers":[input_size,64,32,1]},
        {"lr":0.0005,"layers":[input_size,128,64,1]},
        {"lr":0.001,"layers":[input_size,64,32,16,1]}
    ]

    best_f1=0
    best_model=None

    os.makedirs("experiments",exist_ok=True)

    for i,config in enumerate(configs):
        print(f"\nRunning Experiment {i}")

        params,history=train(
            X_train,y_train,X_val,y_val,
            config["layers"],
            lr=config["lr"]
        )

        _,probs=predict(X_test,params)
        th=best_threshold(y_test.flatten(),probs.flatten())

        y_pred,_=predict(X_test,params,threshold=th)

        f1=f1_score(y_test.flatten(),y_pred.flatten())

        print("F1:",f1)

        plot_history(history,f"experiments/run_{i}.png")
        pd.DataFrame(history).to_csv(f"experiments/run_{i}.csv")

        if f1>best_f1:
            best_f1=f1
            best_model=params

    print("\nBest F1:",best_f1)
    return best_model

# =========================
# LOAD DATA
# =========================
df=pd.read_csv("creditcard.csv")
df=df.drop(columns=["Time"])

X=df.drop("Class",axis=1).values
y=df["Class"].values.reshape(1,-1)

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(
    X,y.T,test_size=0.2,stratify=y.T,random_state=42
)

X_train,X_val,y_train,y_val=train_test_split(
    X_train,y_train,test_size=0.1,random_state=42
)

X_train,X_val,X_test=X_train.T,X_val.T,X_test.T
y_train,y_val,y_test=y_train.T,y_val.T,y_test.T

print("Input shape:", X_train.shape)  # debug

# =========================
# RUN
# =========================
best_model=run_experiments(X_train,y_train,X_val,y_val,X_test,y_test)

# =========================
# SAVE
# =========================
pickle.dump({"params":best_model,"scaler":scaler},open("best_model.pkl","wb"))

print("Best model saved ✅")