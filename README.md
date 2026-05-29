# XDFC-IDS 
## Repository structure
```text
xdfc-ids/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── cnn_gru.py
│   │   ├── lstm.py
│   │   └── factory.py
│   ├── federated/
│   │   ├── aggregation.py
│   │   ├── collaborator.py
│   │   └── dfl_trainer.py
│   ├── incremental/
│   │   ├── continual_dataset.py
│   │   └── ewc.py
│   ├── explainability/
│   │   └── shap_explainer.py
│   └── utils/
│       ├── metrics.py
│       └── seed.py
└── scripts/
    └── run.py
```