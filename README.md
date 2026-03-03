# **End-to-End MLOps Project - Crop Prediction**

---

## **Table of Contents**

1. [Introduction](#introduction)  
2. [Architecture Overview](#architecture-overview)  
3. [Technologies Used](#technologies-used)  
4. [Project Structure](#project-structure)  
5. [Development Pipeline](#development-pipeline)  
6. [Setup and Execution](#setup-and-execution)  
7. [Testing](#testing)  
8. [License](#license)

---

## **Introduction**

This project introduces an **End-to-End MLOps pipeline** for building a machine learning solution focused on **crop prediction**. The core objective is to recommend the most suitable crop type based on soil parameters, enabling better agricultural decision-making.

The emphasis of this project is on **MLOps best practices**, such as the automation of machine learning workflows, reproducible pipelines, version control for models, and deployment-ready APIs.

### **Goals:**
- Build a fully automated pipeline covering data processing, model training, and deployment.  
- Deploy a RESTful API to make model predictions available in a containerized environment on AWS.

> **Note:** The dataset is intentionally simple. The focus of this project is not model performance or complex data, but the design of a robust and well structured architecture that follows best practices across the entire machine learning lifecycle.

---

## **Architecture Overview**

The project architecture is modular to ensure scalability and maintainability. The following components are included in the pipeline:

1. **Data Exploration and Processing:**   

   Initial exploration of soil datasets, data cleaning, and preprocessing.
   
3. **Feature Engineering:**  

   Features are preprocessed after the Data Exploration.
   
5. **Model Training and Experiment Tracking:**  

   Training the model using the Pycaret benchmark as a base, with the experiments being tracked using MLFlow.
   
7. **Model Deployment:**  

   The trained model is provided as a RESTful API, allowing external services to request predictions.
   
9. **Continuous Integration/Deployment:**  

   Using **GitHub Actions** for running tests and deploying services.
   

---

### **Architecture Diagram (It will change)**

Below is the general solution diagram that represents the system flow:

![Architecture Diagram](https://github.com/user-attachments/assets/f4ecbe19-9d86-456b-ad24-b3cb57584bb3)

---

## **Technologies Used**

### Core Libraries:
- Scikit-Learn 
- PyCaret 
- MLFlow 
### Backend and Deployment:
- FastAPI  
- Uvicorn 
- Docker
- Docker Compose 

### Data Handling and Visualization:
- Pandas 
- Seaborn   
- Pydantic 

### Integration and CI/CD:
- GitHub Actions  
- Pytest 

---

## **Project Structure**

The project is organized as follows:

```
end-to-end-mlops/
│
├── notebooks/
│   └── Data exploration and experimentation notebooks.
│
├── data/
│   └── Datasets used for training and testing.
│
├── config/
│   └── Configuration files for the pipeline (e.g., YAML, JSON).
│
├── model/
│   └── Trained models, saved checkpoints, and model artifacts.
│
├── AI_reports/
│   └── Reports created by the AI ​​agent.
│
├── src/
│   ├── Preprocessing scripts.
│   ├── Training pipeline.
│
├── tests/
│   └── Unit and integration test scripts.
│
├── Dockerfile
│   └── Instructions to containerize the API and training scripts.
│
├── compose.yaml
│   └── Docker Compose file that orchestrates services (API, training, MLFlow).
│
├── app.py
│   └── FastAPI script serving trained models as a REST API.
│
├── pyproject.toml
│   └── Lists project dependencies and environment configuration.
│
└── README.md
    └── Overview of the project (current file).
```

---

## **Development Pipeline**

The development lifecycle is divided into clear stages as follows:

### **1. Data Exploration and Preprocessing**
- Exploration: The raw soil dataset is analyzed and explored.
- Preprocessing: Features are normalized (e.g., Z-score normalization), and categorical variables are encoded using LabelEncoder.

### **2. Feature Engineering**
- Numerical characteristics are standardized and categorical are processed.

### **3. Model Training and Validation**
- PyCaret is used to rapidly benchmark models and evaluate performance metrics.
- Training the best model using the Pycaret benchmark as a base.

### **4. Experiment Tracking**
- MLFlow Autologging is enabled to log all hyperparameters, training results, and performance metrics for every experiment.  
- Manual logs are added for preprocessing steps and specific configurations.

### **5. Model Deployment**
- The trained model is deployed as an API using FastAPI.
- The API exposes endpoints for making predictions with the trained model.  
- All services are containerized using Docker for easy distribution.
- Deployed in AWS EC2 using Github Actions

---

## **Setup and Execution**

### Prerequisites:
- **Python 3.8+**
- **Docker and Docker-Compose**

### Installation Steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Hotarouuu/end-to-end-mlops.git
   cd end-to-end-mlops
   ```

2. Build containers:
   ```bash
   docker compose up --build
   ```

3. Access the API documentation:
   Visit [http://localhost:8000/docs](http://localhost:8000/docs) where you can test the endpoints.

---

## **Testing**

Automated testing ensures the pipeline functions reliably. The following testing layers are included:

1. **Unit Tests:**  
   Validate individual components like preprocessing, feature engineering, and model inference functions.  

2. **Integration Tests:**  
   Verify the interaction between multiple modules, including end-to-end pipeline workflows.

### Run Tests:
To execute all tests, run the following:
```bash
pytest tests/
```

---


## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.



