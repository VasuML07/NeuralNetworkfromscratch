flowchart LR

%% ---------- INPUT LAYER ----------
subgraph INPUT_LAYER["Input Layer (30 Features)"]
direction TB
X1(("x1"))
X2(("x2"))
X3(("x3"))
X4(("⋯"))
X30(("x30"))
end

%% ---------- HIDDEN LAYER ----------
subgraph HIDDEN_LAYER["Hidden Layer (16 Neurons • ReLU)"]
direction TB
H1(("h1"))
H2(("h2"))
H3(("h3"))
H4(("h4"))
H5(("h5"))
H6(("h6"))
H7(("h7"))
H8(("h8"))
H9(("h9"))
H10(("h10"))
H11(("h11"))
H12(("h12"))
H13(("h13"))
H14(("h14"))
H15(("h15"))
H16(("h16"))
end

%% ---------- OUTPUT ----------
subgraph OUTPUT_LAYER["Output Layer (Sigmoid)"]
direction TB
O1(("Prediction"))
end

%% ---------- CONNECTIONS ----------
X1 --> H1
X1 --> H2
X2 --> H3
X2 --> H4
X3 --> H5
X3 --> H6
X4 --> H7
X30 --> H16

H1 --> O1
H2 --> O1
H3 --> O1
H4 --> O1
H5 --> O1
H6 --> O1
H7 --> O1
H8 --> O1
H9 --> O1
H10 --> O1
H11 --> O1
H12 --> O1
H13 --> O1
H14 --> O1
H15 --> O1
H16 --> O1

%% ---------- STYLING ----------
classDef input fill:#dbeafe,stroke:#1e40af,stroke-width:2px,color:#000;
classDef hidden fill:#dcfce7,stroke:#166534,stroke-width:2px,color:#000;
classDef output fill:#fee2e2,stroke:#991b1b,stroke-width:2px,color:#000;

class X1,X2,X3,X4,X30 input
class H1,H2,H3,H4,H5,H6,H7,H8,H9,H10,H11,H12,H13,H14,H15,H16 hidden
class O1 output

%% ---------- TOOLTIP INTERACTIVITY ----------
click O1 "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html" "Dataset used for prediction"
