# 🧠 Image Classification Research

Welcome to the **Image Classification Research** repository — a modular and reproducible deep learning pipeline for image classification tasks using PyTorch. This repo includes training scripts, inference demos, and evaluation tools to accelerate your computer vision experiments.

---

## 🚀 Getting Started

Follow the steps below to set up your environment and start training your models.

### 📦 1. Install Python Virtual Environment (Ubuntu)

```bash
sudo apt install -y python3-virtualenv
```

### 🔧 2. Create & Activate Virtual Environment

```bash

virtualenv ENV_NAME
source ENV_NAME/bin/activate

```

### 📚 3. Install Required Dependencies

```bash

pip install -r requirements.txt

```

### 🧠 4. Register Jupyter Kernel (Optional)

```bash

python -m ipykernel install --name "ENV_NAME" --user

```

## 📁 Available Datasets

The links to the datasets can be found in the [fetch script](https://github.com/bekhzod-olimov/ImageClassificationSampleProjects/blob/13624441bbd226daf7fb3e3d56ed98fbbb230df2/data/fetch.py#L7).
 * Animals Dataset;

 * Apple Disease Dataset; 

 * Car Brands Dataset;

 * Dog Breeds Dataset;
 
 * Facial Expressions Dataset;

 * Geo Scene Dataset;

 * Lentils Dataset;

 * Pet Disease Dataset;

 * Rice Leaf Disease Dataset;

 * Pokemon Dataset;

 * Remote Sensing (PatternNet) Dataset 

 * The more is coming...

 ## 🛠️ Manual to Use This Repo

 ### 🔁 Train and Evaluate a Model
Run the training and evaluation pipeline:

```bash
python main.py --ds_nomi pet_disease --dataset_root PATH_TO_YOUR_DATA --batch_size 32 --device "cuda"

```

### 🌐 Streamlit Demo

```bash

streamlit run demo.py

```

## 🤝 Contributing
Have an idea or found a bug? Feel free to open an issue or a pull request. Contributions are welcome!

## 📃 License
This project is open source and available under the MIT License.
