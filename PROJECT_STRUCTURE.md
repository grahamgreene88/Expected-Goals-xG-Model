# Project Structure

project_root/
├── data/
│ ├── raw/
│ ├── processed/
│ └── models/
├── src/
│ ├── api/
│ │ ├── api_client.py
│ │ └── data_parsers.py
│ ├── features/
│ │ ├── feature_scaling.py
│ │ └── feature_engineering.py
│ ├── modeling/
│ │ ├── train_models.py
│ │ ├── model_utils.py
│ │ └── evaluation.py
│ ├── utils/
│ │ └── logging_utils.py
│ └── init.py
├── notebooks/
│ ├── 01_api_testing.ipynb
│ ├── 02_feature_debugging.ipynb
│ └── 03_model_experiments.ipynb
├── models/
│ ├── scalers/
│ └── saved_models/
├── environment.yml / requirements.txt
└── README.md

