# Roadmap for this refactor


For this refactor I'm planning to make it the version "1.0.0" in the release section of this repository. I would say that the current master branch is the version 1.0.0, but with all the mistakes I made I want to properly make a real "production level" release.

My vision is to make this a full production-level MLOPs project hosted in AWS (specifically EC2, not sagemaker yet) with all the best practices we have right now on this field.


# 1. Exploratory Data Analysis Re-do

The main problem this project had was the poor insights we had from the original dataset. 

- EDA Refactor
    - **Better Data Analysis:** Understand the basic statistics, data distributions and possible outliers
    - **Patterns understanding:** Find the general patterns for each of the classes, focusing on better model understanding

- Public notebook hosted on Kaggle

# 2. Tests Refactor

The previous tests were generated with vibecoding and they were poorly reviewed. With the full tests refactor I now have understanding what each functions do

- Unit tests refactor
    - **Mocking MLFlow:** For unit tests we are mocking MLFlow, since it is the main library technology we are using in this MLOps Solution
    - **Unit Model:** The model script was refactored. Now we are only testing the import_model function from MLFLow. The purpose is to train the LOGIC, not the model itself
    - **Unit API:** We are testing the logic of the 2 endpoints we have in the API. With the MLFlow mock, we can test them without having a live server

- Integration Test:
    - **Full Dockerize Test**: Instead of testing only the training, we are going to test the full service including all the three conteiner. This solution is better since this project heavily depends on MLFlow

# 3. More advanced and secure API

The current API is basic, with only two endpoints, no authentication and no error treatment.

- More Endpoints:
    - **SHAP Values Endpoint**: Shap Endpoint to avoid any possible latency for the predict endpoint. It isnt defined yet

- Security Implementation:
    - **Implement API keys:** Implementation of keys to improve the security
    - **Cloudfare tunnel:** Implementation of Cloudfare Tunnel service to avoid exposing doors to the internet.

# 4. Error treatment

Implementation of error treatment is crucial for this project, combining with better logging.

# 5. Example UI

Simple UI Vibe-coded or with streamlit library to use the project properly with these functions:

- Input and Prediction: Basic functionally with input the data and the model prediction
- Model interpretability:
    - **Return SHAP Values:** For more advanced users it will generate the shap values for the input, with the possibility of generating graphs to understand how the model produced the prediction
    - **Gemini recommendation:** For more basic users and for the advanced as well it will have Gemini integration for natural language interpretation of the model and possible courses of action given the prediction.



