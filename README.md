
```
session10_sagemaker_ec2/
├── pipeline.py          # Training: load data, train 3 model, simpan winner
├── deploy_endpoint.py   # Deploy model.tar.gz ke SageMaker endpoint
├── streamlit_app.py     # Streamlit UI yang memanggil endpoint
├── user-data.sh         # Bootstrap EC2 (User Data script)
├── train.csv            # ← TARUH DI SINI sebelum jalankan pipeline.py
├── README.md
└── src/
    ├── data.py          # Load + preprocess Spaceship Titanic CSV
    ├── models.py        # Logistic Regression, Random Forest, XGBoost
    ├── evaluate.py      # Metrik + comparison table
    ├── inference.py     # SageMaker inference entry point
    └── requirements.txt
```
