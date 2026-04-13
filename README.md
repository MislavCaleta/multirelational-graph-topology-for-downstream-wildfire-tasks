**Wildfire GNN: Multirelational Forensic Attribution**

This repository implements a relational framework for classifying wildfire ignitions as anthropogenic or natural. By moving beyond "topologically blind" I.I.D. models like MLPs, this approach leverages the spatio-temporal dependencies inherent in fire regimes.

**Core Concept**

Traditional models treat fires as isolated points. Our multirelational graph construction proves that how events are connected in space and time is a more significant driver of predictive accuracy than raw environmental features alone.

**Key Components**

- Multirelational Topology: Custom graph construction encoding spatial and temporal dependencies.

- Structural Ablation: Proof that relational context (even with minimal features) outperforms feature-heavy MLPs.

- GNN Architectures: Implementations of GAT, GCN, and TransformerConv optimized for forensic attribution.

- Interpretability: SHAP-based analysis of structural and environmental feature importance.

**Results**

Our findings demonstrate that combining relational context with descriptive environmental data provides a significantly more robust framework for forensic fire attribution than traditional point-based methods.
