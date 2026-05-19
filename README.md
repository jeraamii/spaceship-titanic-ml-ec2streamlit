# Session 10 – SageMaker Training & EC2 Deployment (Spaceship Titanic)

Dataset: **Spaceship Titanic** (binary classification — `Transported`: True/False)

---

## Struktur File

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

---

## QUESTION 1 – Training & Deploy ke SageMaker Endpoint

### Step 1 – Buka SageMaker Notebook Instance

1. Login ke AWS Academy → SageMaker → Notebook Instances
2. Buka (atau buat) notebook instance → **Open JupyterLab**
3. Upload **seluruh folder** `session10_sagemaker_ec2/` ke notebook (termasuk `train.csv`)

### Step 2 – Install dependencies

Buka Terminal di JupyterLab, lalu:

```bash
pip install xgboost scikit-learn pandas joblib sagemaker boto3
```

### Step 3 – Jalankan training

```bash
cd session10_sagemaker_ec2
python pipeline.py
```

Output yang diharapkan:
```
Loading dataset...
Dataset shape: (8693, 14)
Train: (6954, 13), Test: (1739, 13)

Training logistic_regression...
Training random_forest...
Training xgboost...

Model              Train Acc   Test Acc    Test F1
------------------------------------------------------
logistic_regression   0.7912     0.7810     0.7809
random_forest         0.9998     0.8052     0.8050
xgboost               0.9412     0.8120     0.8118

Winner: xgboost
...
Packaged: model_artifact/model.tar.gz
```

### Step 4 – Upload model ke S3

```bash
# Buat bucket (ganti nama sesuai keinginan, harus unik global)
aws s3 mb s3://spaceship-model-<nama-anda> --region us-east-1

# Upload model
aws s3 cp model_artifact/model.tar.gz s3://spaceship-model-<nama-anda>/spaceship/model.tar.gz
```

### Step 5 – Edit & jalankan deploy_endpoint.py

Edit bagian berikut di `deploy_endpoint.py`:

```python
BUCKET        = "spaceship-model-<nama-anda>"   # ← ganti
MODEL_S3_KEY  = "spaceship/model.tar.gz"
ENDPOINT_NAME = "spaceship-endpoint"
```

Jalankan:

```bash
python deploy_endpoint.py
```

Tunggu 5–8 menit hingga endpoint aktif.

**Screenshot yang diperlukan:**
- Hasil output terminal `pipeline.py` (terlihat comparison table + Winner)
- Halaman SageMaker Endpoints di AWS Console (status: InService)
- Pastikan seluruh layar ter-screenshot (termasuk nama akun AWS di kanan atas)

---

## QUESTION 2 – Deploy Streamlit ke EC2

### Step 1 – Push code ke GitHub

Upload seluruh folder ke repository GitHub kamu (public atau private).

### Step 2 – Launch EC2 Instance

1. EC2 → Launch Instance
2. AMI: **Amazon Linux 2023**
3. Instance type: `t2.micro` (atau `t3.micro`)
4. Security Group: buka port **8501** (Custom TCP) dan **22** (SSH) dari `0.0.0.0/0`
5. IAM Instance Profile: pilih **LabInstanceProfile**
6. **Advanced Details → User Data**: paste isi `user-data.sh`, edit 2 variabel:
   ```bash
   GIT_REPO="https://github.com/<username>/<repo>.git"
   ENDPOINT_NAME="spaceship-endpoint"
   ```
7. Launch!

### Step 3 – Akses Streamlit

Setelah EC2 running (~3 menit), buka browser:

```
http://<EC2-Public-IP>:8501
```

### Step 4 – Jika ingin run manual (tanpa user-data)

SSH ke EC2:
```bash
ssh -i your-key.pem ec2-user@<EC2-Public-IP>
```

Lalu:
```bash
sudo dnf install -y python3 python3-pip git
git clone https://github.com/<username>/<repo>.git app
cd app
pip3 install streamlit boto3
export ENDPOINT_NAME="spaceship-endpoint"
export AWS_REGION="us-east-1"
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

**Screenshot yang diperlukan:**
- Halaman Streamlit di browser (tampil form input + hasil prediksi)
- Halaman EC2 Instances di AWS Console (status: Running)
- Pastikan seluruh layar ter-screenshot (termasuk nama akun AWS)

---

## Cara Kerja Preprocessing

Dataset Spaceship Titanic memiliki kolom mixed (numerik, kategorikal, boolean):

| Kolom | Tipe | Handling |
|-------|------|---------|
| HomePlanet, Destination | Kategorikal | Fill mode → Label encode |
| CryoSleep, VIP | Boolean | Fill False → int |
| Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck | Numerik | Fill median |
| Cabin | String "Deck/Num/Side" | Split → CabinDeck, CabinSide → encode |
| TotalSpend | Derived | Sum semua pengeluaran |

---

## Format Input Endpoint

```json
{
  "instances": [
    [1, 0, 2, 27.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 5, 0, 0.0]
  ]
}
```

Urutan: `HomePlanet, CryoSleep, Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, CabinDeck, CabinSide, TotalSpend`

Encoding:
- HomePlanet: Earth=0, Europa=1, Mars=2
- Destination: "55 Cancri e"=0, "PSO J318.5-22"=1, "TRAPPIST-1e"=2
- CabinDeck: A=0, B=1, C=2, D=3, E=4, F=5, G=6, T=7
- CabinSide: P=0, S=1
