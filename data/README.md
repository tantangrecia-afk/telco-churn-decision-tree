# Original customer dataset

- Publisher: IBM.
- Dataset: Telco Customer Churn, sample data for a fictional telecom business.
- Source: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
- Download URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Keep raw/Telco-Customer-Churn.csv unchanged. Perform later cleaning in code.
- Target: Churn (Yes = left; No = stayed).
- Each row is one customer. customerID identifies the record and should not be used as a predictor.

## Column guide

| Column | Meaning |
| --- | --- |
| customerID | Customer identifier |
| gender | Recorded gender category |
| SeniorCitizen | Senior citizen indicator (0/1) |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Months with the company |
| PhoneService | Whether the customer has phone service |
| MultipleLines | Whether the customer has multiple phone lines |
| InternetService | Internet service type |
| OnlineSecurity | Online security subscription |
| OnlineBackup | Online backup subscription |
| DeviceProtection | Device protection subscription |
| TechSupport | Technical support subscription |
| StreamingTV | Streaming TV subscription |
| StreamingMovies | Streaming movies subscription |
| Contract | Contract type |
| PaperlessBilling | Whether billing is paperless |
| PaymentMethod | Payment method |
| MonthlyCharges | Monthly charge amount |
| TotalCharges | Total charge amount; inspect numeric conversion in Phase 3 |
| Churn | Outcome to predict (Yes/No) |

The sample does not establish that it represents Philippine telecom customers. We will discuss this limitation in the project presentation.
