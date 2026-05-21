# Analysis of Home Mortgage Data: Predictive Modeling and Test for Discriminatory Practice

This project analyzes home mortgage loan applications to predict acceptance rates and audit the lending system for potential unfair discrimination. The dataset consists of approximately 440,000 records.

The project accomplishes two goals:
* **Predictive Modeling**: Determining if loan acceptance can be accurately predicted based on publicly available Home Mortgage Disclosure Act (HMDA) data.
* **Ethical Auditing**: Investigating whether evidence of discrimination exists based on identity and demography features, specifically race and sex, after accounting for all legitimate financial factors.

### Background
The Home Mortgage Disclosure Act of 1975 requires financial institutions to report information about loan applications to ensure lending practices serve the public and remain equitable. This analysis seeks to identify which factors are most relevant to loan acceptance and whether subjective review processes introduce bias into the system.

## Data Preparation & Feature Engineering
The analysis utilized a substantial dataset of nearly 440,000 complete records. Data categories included property location, loan information (lender, amount, purpose), applicant information (income, race, sex), and tract-level census data.

Early review of categorical data shows some data is informative about loan acceptance rates.  Somed of these features are "identity" / demographic features, which could represent unfair discrimination.

<img width="500" alt="cat-features-univariate-predictions" src="https://github.com/user-attachments/assets/4c2bb5cf-e7b1-4f62-96ca-f988f6d2ff2c" /> <br>

Checking numeric features data for colinearities shows few, although there is a clean correlation between the region population and the number of the property owner's units that are occupied.

<img width="600" alt="cross-scatter-of-selected-numeric-features" src="https://github.com/user-attachments/assets/1f5b0013-bcc6-4c59-8427-53227f851b78" /> <br>

### Unique Feature Engineering: Binning High-Cardinality Categoricals
A unique challenge in this dataset was the presence of "big categoricals"—features with immense variety like `lender` (6,111 unique categories) and `county_code` (318 unique categories). 

To use these effectively without overwhelming the models:
* **Thresholding**: Categories with fewer than 50 records were grouped as "unknown" to ensure reliable statistica l estimates.
* **Acceptance Rate Binning**: Remaining categories were grouped into distinct "bins" based on their mean loan acceptance rates in steps of 10 percentage points. 
* **Result**: This transformation allowed the model to efficiently capture the massive variation in lender behavior—where some lenders accept less than 10% of applications and others over 90%—contributing to a high independent predictive accuracy of 0.66 for the `lender` feature alone.

## Modeling and Results
Multiple machine learning algorithms were evaluated to determine the upper limits of predictive performance on this dataset.

| Model | Accuracy (%) |
| :--- | ---: |
| Logistic Regression | 69.1 |
| Decision Tree (max depth 12) | 69.4 |
| Random Forest | 69.0 |
| Support Vector Machines | 69.1 |
| AdaBoost | 69.6 |
| Neural Network (30, 30 hidden layers) | 70.6 |

### Key Findings
**Prediction**: Logistic regression set a benchmark score of 69.1.  Despite the power of the other models attempted, they barely outperformed the simplest model.  The underlying trend in the data is essentially linear.

**Feature Efficiency**: Using an independent predictive accuracy threshold, the feature set was cut in half. Results showed that the stronger half - those features with independent predictive power of at least 0.55 - showed nearly the same predictive power as the full set, with only a 0.2-0.3 percentage point improvement gained by including the rest.

## Test for Discrimination
The project employed a stepwise auditing framework to detect discrimination, measuring the marginal impact of protected characteristics after controlling for legitimate baseline variables.

To test accurately, three categories of data were distinguished.
1. **"Fair Features"** - Features that represent legitimate factors in evaluating a loan, such as the applicant income and loan amount.  
1. **"Identity Features"** - Features of the applicant that would represent unfair discrimination:  race,  sex, and similar.  
1. **Proxy Features** - Features that do not directly represent unfair discrimination, but correlate with those features.

The test trains and compares two models: one trained only on Fair Features, and a second that combines Fair and Identity Features.  To avoid information spillover either way, Proxy Features are left out.  If the model with identity features shows a significant predictive advantage over the pure-fair model, this would support the conclusion that mortgage lending exhibited a degree of unfair discrimination.

### Results
The result of the audit is only a tiny amount of predictive power when discriminatory features are included, supporting the conclusion that demographic features did not play a significant role in decision-making by mortgage lenders overall.

|Feature Set | Model Accuracy |
|---|---:|
|Fair Features Only (Baseline) |	68.6%|
|Fair + Identity Features |	68.8%|
|Marginal Difference | 0.2%|

<img width="500" alt="Code_Generated_Image" src="https://github.com/user-attachments/assets/ffbef24e-740f-4836-9b89-c0b193e3d97c" />

## About the Repository
This repository hosts:
- Source data (code folder)
- Python code for data processing (code folder)
- Guide to Features, explaining features and their encoding
- Executive Report, describing the project in detail
