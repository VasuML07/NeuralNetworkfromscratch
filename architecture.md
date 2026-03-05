flowchart LR

%% ========== 🎭 GLOBAL STYLING ==========
%% Enable smooth curves & professional spacing
%%: { "theme": "base", "themeVariables": { "primaryColor": "#fff", "edgeLabelBackground":"#f8fafc", "tertiaryColor": "#f1f5f9" } }

%% ========== 📥 INPUT LAYER ==========
subgraph INPUT["📥 INPUT LAYER • 30 Clinical Features"]
    direction TB
    subgraph FEATURES["🔬 Feature Groups"]
        direction LR
        F1["📏 Radius/Texture"]
        F2["📐 Perimeter/Area"]
        F3["🌀 Smoothness/Compactness"]
        F4["📊 Concavity/Symmetry"]
        F5["🎯 Fractal Dimension"]
    end
    note1["✨ Standardized • Zero-Mean • Unit-Variance"]:::note
end

%% ========== 🧠 HIDDEN LAYER ==========
subgraph HIDDEN["🧠 HIDDEN LAYER • 16 Neurons"]
    direction TB
    subgraph NEURONS["⚡ ReLU Activation • L2 Regularization"]
        direction LR
        N1(("•"))
        N2(("•"))
        N3(("•"))
        N4(("•"))
        N5(("•"))
        N6(("•"))
        N7(("•"))
        N8(("•"))
        N9(("•"))
        N10(("•"))
        N11(("•"))
        N12(("•"))
        N13(("•"))
        N14(("•"))
        N15(("•"))
        N16(("•"))
    end
    note2["🔁 BatchNorm • Dropout 30% • He Initialization"]:::note
end

%% ========== 🎯 OUTPUT LAYER ==========
subgraph OUTPUT["🎯 OUTPUT LAYER • Binary Classification"]
    direction TB
    PRED(("🔮 Malignant / Benign"))
    SIGMOID["σ(x) = 1/(1+e⁻ˣ)"]:::math
    THRESHOLD["⚖️ Threshold: 0.5"]:::note
end

%% ========== 🌊 CONNECTIONS ==========
%% Input → Hidden (elegant flow)
FEATURES == "✨ Weighted Connections\n🎯 30×16 = 480 Parameters" ==> NEURONS

%% Hidden → Output (converging prediction)
NEURONS ==>|"🔗 16 Final Weights\n🧠 Non-linear Feature Fusion"| PRED

%% ========== 🎨 DECORATIVE ELEMENTS ==========
%% Data flow indicators
linkStyle 0 stroke:#3b82f6,stroke-width:3px,fill:none,stroke-dasharray:5 5
linkStyle 1 stroke:#10b981,stroke-width:4px,fill:none,stroke-linecap:round

%% ========== 🏷️ LAYER BADGES ==========
INPUT:::inputLayer
HIDDEN:::hiddenLayer
OUTPUT:::outputLayer

%% ========== 🖱️ INTERACTIVITY ==========
click PRED href "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html" "📊 Explore Breast Cancer Dataset" _blank
click FEATURES href "https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+diagnostic" "🔍 Feature Documentation" _blank

%% ========== 🎨 PROFESSIONAL STYLING ==========
classDef inputLayer fill:#dbeafe,stroke:#1e40af,stroke-width:3px,color:#1e3a8a,font-weight:bold,font-size:14px
classDef hiddenLayer fill:#dcfce7,stroke:#166534,stroke-width:3px,color:#14532d,font-weight:bold,font-size:14px
classDef outputLayer fill:#fef3c7,stroke:#b45309,stroke-width:4px,color:#92400e,font-weight:bold,font-size:16px
classDef note fill:#f8fafc,stroke:#94a3b8,stroke-width:1px,color:#475569,font-size:11px,font-style:italic
classDef math fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#5b21b6,font-family:monospace

class note1,note2,THRESHOLD note
class SIGMOID math

%% ========== ✨ ANIMATION HINTS (Mermaid Live) ==========
%% Add class "animate-pulse" to elements for subtle motion
%% Requires: %%{init: {'theme': 'base', 'flowchart': {'htmlLabels': true}}}%%

%% ========== 🏆 LEGEND ==========
subgraph LEGEND["🗝️ Visual Legend"]
    direction LR
    L1["🔵 Input: Raw Features"]:::inputLayer
    L2["🟢 Hidden: Feature Learning"]:::hiddenLayer
    L3["🟠 Output: Final Decision"]:::outputLayer
end
