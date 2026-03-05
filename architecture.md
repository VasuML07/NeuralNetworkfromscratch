%%{init: {'theme': 'base', 'flowchart': {'htmlLabels': true, 'curve': 'basis'}}}%%

flowchart LR

%% ========== 📥 INPUT LAYER ==========
subgraph INPUT["📥 INPUT LAYER • 30 Clinical Features"]
    direction TB
    F1["📏 Radius/Texture"]
    F2["📐 Perimeter/Area"]
    F3["🌀 Smoothness/Compactness"]
    F4["📊 Concavity/Symmetry"]
    F5["🎯 Fractal Dimension"]
    note1["✨ Standardized • Zero-Mean • Unit-Variance"]
end

%% ========== 🧠 HIDDEN LAYER ==========
subgraph HIDDEN["🧠 HIDDEN LAYER • 16 Neurons"]
    direction TB
    N1(("•")) & N2(("•")) & N3(("•")) & N4(("•"))
    N5(("•")) & N6(("•")) & N7(("•")) & N8(("•"))
    N9(("•")) & N10(("•")) & N11(("•")) & N12(("•"))
    N13(("•")) & N14(("•")) & N15(("•")) & N16(("•"))
    note2["⚡ ReLU • BatchNorm • Dropout 30%"]
end

%% ========== 🎯 OUTPUT LAYER ==========
subgraph OUTPUT["🎯 OUTPUT LAYER • Binary Classification"]
    direction TB
    PRED(("🔮 Malignant / Benign"))
    SIGMOID["σ(x) = 1/(1+e⁻ˣ)"]
    THRESHOLD["⚖️ Threshold: 0.5"]
end

%% ========== 🔗 CONNECTIONS ==========
F1 & F2 & F3 & F4 & F5 -->|30 Features → 16 Neurons| N1
N1 & N2 & N3 & N4 & N5 & N6 & N7 & N8 -->|Feature Fusion| PRED
N9 & N10 & N11 & N12 & N13 & N14 & N15 & N16 --> PRED

%% ========== 🗝️ LEGEND ==========
subgraph LEGEND["🗝️ Legend"]
    direction LR
    L1["🔵 Input"] 
    L2["🟢 Hidden"] 
    L3["🟠 Output"]
end

%% ========== 🎨 STYLING ==========
classDef inputLayer fill:#dbeafe,stroke:#1e40af,stroke-width:2px,color:#1e3a8a,font-weight:bold
classDef hiddenLayer fill:#dcfce7,stroke:#166534,stroke-width:2px,color:#14532d,font-weight:bold
classDef outputLayer fill:#fef3c7,stroke:#b45309,stroke-width:3px,color:#92400e,font-weight:bold
classDef note fill:#f8fafc,stroke:#94a3b8,stroke-width:1px,color:#475569,font-size:11px,font-style:italic
classDef math fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#5b21b6,font-family:monospace

class F1,F2,F3,F4,F5,note1,inputLayer inputLayer
class N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15,N16,note2 hiddenLayer
class PRED,SIGMOID,THRESHOLD,OUTPUT outputLayer
class note1,note2,THRESHOLD note
class SIGMOID math
class L1 inputLayer
class L2 hiddenLayer
class L3 outputLayer

%% ========== 🖱️ INTERACTIVITY (Optional - Remove if rendering fails) ==========
%% click PRED href "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html" "📊 Dataset Docs"
