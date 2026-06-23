# Roadmap for the 1.0.0 release


For this refactor I'm planning to make it the version "1.0.0" in the release section of this repository. I would say that the current master branch is the version 1.0.0, but with all the mistakes I made I want to properly make a real "production level" release.

My vision is to make this a full production-level MLOPs project hosted in AWS (specifically EC2, not sagemaker yet) with all the best practices we have right now on this field.


# 1. Exploratory Data Analysis Re-do

The main problem with the project was the poor insights from the original dataset.

- EDA Refactor
    - **Better Data Analysis:** Understand the basic statistics, data distributions and possible outliers
    - **Pattern understanding:** Find the general patterns for each of the classes, focusing on better model understanding

- Public notebook hosted on Kaggle

# 2. Tests Refactor

The previous tests were generated with vibecoding and were poorly reviewed. With the full tests refactor, understanding of each function was gained.

- Unit test refactor
    - **Mocking MLFlow:** For unit tests, MLFlow is mocked, since it is the main library technology used in this MLOps Solution
    - **Unit Model:** The model script was refactored. Now only the import_model function from MLFlow is tested. The purpose is to train the LOGIC, not the model itself
    - **Unit API:** The logic of the 2 endpoints in the API is tested. With the MLFlow mock, they can be tested without a live server

- Integration test:
    - **Full Dockerized test:** Instead of testing only the training, the full service including all three containers is tested. This solution is better since the project heavily depends on MLFlow

# 3. More advanced and secure API

The current API is basic, with only two endpoints, no authentication and no error handling.

- More endpoints:
    - **SHAP values endpoint:** SHAP endpoint to avoid any possible latency for the predict endpoint. It is not defined yet

- Security implementation:
    - **API keys:** Implementation of keys to improve security
    - **Cloudflare tunnel:** Implementation of Cloudflare Tunnel service to avoid exposing doors to the internet.

# 4. Error handling

Implementation of error handling is crucial for the project, combining with better logging.

# 5. UI

Simple UI Vibe-coded or with streamlit library to use the project properly

- Input and Prediction: Basic functionality with inputting the data and the model prediction
- Model interpretability:
    - **Return SHAP values:** For advanced users, the shap values for the input are generated, with the possibility of generating graphs to understand how the model produced the prediction
    - **Gemini recommendation:** For both basic and advanced users, Gemini integration is provided for natural language interpretation of the model and possible courses of action given the prediction.



