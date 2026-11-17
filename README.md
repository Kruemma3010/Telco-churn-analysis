# Telco Churn: Customer Prioritisation for Retention

A customer analytics project comparing logistic regression and random forest to prioritise customers for a retention campaign with limited contact capacity.

**Key result:** Selecting the highest-scoring 20% of test customers identifies approximately half of all observed churners. At the same contact budget, the models identify 182–187 churners, compared with approximately 75 expected from random selection.

## Project Background

This work builds on an individual project that I completed independently during the **Machine Learning in Practice** course at the **London School of Economics and Political Science (LSE) Summer School**.

For this portfolio version, I revised the analysis around a focused business question, reproducible preprocessing pipelines and an evaluation based on customer prioritisation. The project compares two supervised models and connects their results to a practical retention workflow. The original project also had a part focussing on unsupervised learning clusters, which I omitted, because I think I did a terrible job at the analysis. 

## Business Question

**If a retention team can contact only 20% of customers, how effectively can a model prioritise those at risk of churn?**

The 20% contact budget represents an assumed capacity constraint. It is not a monetary budget or an optimised campaign size. This rate was chosen by the typical pareto principle and can be altered. Customers are ranked by their model scores, and performance is evaluated within the highest-scoring group. This rate was chosen by the typical pareto principle

## Data Source and Download

Download the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same directory as the notebook.

The Kaggle listing identifies the data as IBM sample data and states
“Data files © Original Authors” under its license information.
The raw dataset is therefore not redistributed in this repository.

The original file contains 7,043 customer records and 21 columns, including tenure, contract type, internet service, monthly charges, total charges and observed churn.

- Eleven records with missing total charges are excluded, leaving **7,032 customers**.
- The excluded records all have zero months of tenure, so the analysis does not cover this group of new customers.
- **1,869 customers (26.58%)** in the cleaned sample have an observed churn outcome.
- Customer IDs are retained for matching predictions to customers but excluded from model inputs.

## Approach

1. **Data preparation:** check duplicates and customer identifiers, convert total charges to numeric values, and encode the target.
2. **Exploration:** examine churn rates by contract, tenure, technical support and internet service using only the training sample.
3. **Model fitting:** compare regularised logistic regression with a random forest using the same input features.
4. **Model selection:** use a small parameter grid and five-fold stratified cross-validation, selecting by average precision.
5. **Test evaluation:** measure average precision, ROC-AUC and performance at the fixed contact budget.
6. **Interpretation:** use permutation importance and translate model scores into a ranked customer review list.

The data is split into 80% training and 20% test customers using a fixed random seed. Scaling and one-hot encoding are fitted inside each cross-validation fold through scikit-learn pipelines.

## Results

The test sample contains **1,407 customers**, including **374 observed churners**. Rounding up the 20% contact budget gives **282 customers** to prioritise.

| Selection method             | Average precision |         ROC-AUC | Churners identified | Precision at budget | Recall at budget | Lift at budget |
| ---------------------------- | ----------------: | --------------: | ------------------: | ------------------: | ---------------: | -------------: |
| Random selection — expected |                — |              — |                  75 |               26.6% |            20.0% |           1.00 |
| Logistic regression          |             0.623 |           0.836 |       **187** |     **66.3%** |  **50.0%** | **2.49** |
| Random forest                |   **0.644** | **0.840** |                 182 |               64.5% |            48.7% |           2.43 |

All three approaches use the same 282-customer contact budget. The random-selection figures are expectations, not the results of a particular random draw.

- **Precision at budget:** the proportion of selected customers who actually churned.
- **Recall at budget:** the proportion of all observed churners included in the selection.
- **Lift at budget:** precision divided by the overall test churn rate.

The random forest was selected before test evaluation because its cross-validated average precision was slightly higher: approximately **0.666**, compared with **0.660** for logistic regression.

Logistic regression nevertheless identifies five more churners at the specific 20% contact budget. The results therefore do not establish a clear practical advantage for the more complex model. Better average precision across thresholds does not necessarily mean better performance at one contact limit.

## Business Interpretation

For the same 282 contacts, model-based prioritisation includes approximately **107–112 more historical churners** than random selection. This demonstrates the potential to direct limited retention capacity towards a more relevant customer group.

A ranked customer review list connects these scores with account information such as tenure, contract type and monthly charges. It illustrates how predictions could support a sales or retention team's review process.

**Identifying a churner does not mean preventing their cancellation.** The analysis does not measure how customers respond to an intervention or how much profit a campaign would generate. A controlled campaign test would be needed to estimate incremental retention and contribution margin after contact and offer costs. However, it can serve as a first line of approach for any sales rep.

## Feature Interpretation

Tenure, internet service, total charges and contract type are among the most important features for the random forest according to permutation importance.

These importances describe the fitted model's use of information, not causal effects. In particular, tenure and billing features are related: shuffling one feature can create unrealistic combinations, while correlated features may partly substitute for one another. Error bars in the importance chart represent variation across repeated permutations, not confidence intervals.

## Limitations

- The random customer split evaluates performance within this dataset, not forecasting into a later period.
- The same dataset was used in the earlier academic project, so this is not a completely new external evaluation.
- Before deployment, feature availability must be checked against a defined prediction date and churn horizon.
- Model scores support ranking; their calibration as individual churn probabilities is not assessed.
- Results come from one train/test split and do not establish a statistically reliable advantage between the models.
- Neither campaign effectiveness nor financial returns can be inferred from the churn labels alone.

## Running the Notebook

Place the CSV in the same directory as `Telco_Churn_Analysis.ipynb`, or update `DATA_PATH` in the first code cell.

Install the required packages in your Python environment:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab ipykernel
```

Open the notebook in JupyterLab or VS Code, select that environment as the kernel, and run all cells from the beginning.

```bash
python -m jupyterlab Telco_Churn_Analysis.ipynb
```

The notebook includes saved tables and charts for review without execution. Small numerical differences may occur across package versions.

## Tools

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Jupyter

## Author

Marc Krümpelmann
