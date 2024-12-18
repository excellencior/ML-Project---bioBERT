# Machine Learning Project: Predictive Diagnosis System

## Project Overview

The system builds personalized predictive diagnostic profiles based on medical history, current symptoms, age, sex, etc. Using the pretrained BioBERT model, fine-tuned on a preprocessed dataset (ddxplus), the system generates a list of possible diagnoses with their respective probabilities.

## Dataset and Processing

- **Dataset**: [ddxplus](https://huggingface.co/datasets/aai530-group6/ddxplus-french)
- **Preprocessing**: BioASQ QA format transformed into `input_texts: antecedents, symptoms, labels: differential diagnosis`.

## Training Details

- **Training Samples**:
  - **10k Samples**: Used to train the initial model.
  - **20k Samples**: Used to train the improved model.
- **Evaluation Metrics**:
  - Test Accuracy: ~95%
  - Test Hamming Loss: ~0.048
- **Test Set Sizes**: 2.5k, 5k, and 10k random samples.

## Results: Evaluation Metrics and Top Predictions

### Model Trained on 10k Samples

#### Test Set (2.5k Samples)
| Example | Diagnosis 1                | Prob. 1 | Diagnosis 2         | Prob. 2 | Diagnosis 3  | Prob. 3 | Diagnosis 4               | Prob. 4 | Diagnosis 5                   | Prob. 5 |
|---------|----------------------------|---------|---------------------|---------|--------------|---------|--------------------------|---------|-------------------------------|---------|
| 1       | Cluster headache           | 0.7273  | Anemia              | 0.5783  | Chagas       | 0.1202  | Bronchitis              | 0.0546  | Chronic rhinosinusitis        | 0.0365  |
| 2       | Cluster headache           | 0.6973  | Anemia              | 0.5373  | Chagas       | 0.1499  | Sarcoidosis             | 0.0356  | Bronchitis                    | 0.0349  |
| 3       | Cluster headache           | 0.5922  | Anemia              | 0.2828  | Bronchitis   | 0.0857  | Chagas                  | 0.0731  | URTI                          | 0.0417  |
| 4       | Anemia                     | 0.9710  | Acute dystonic reactions | 0.9503 | Guillain-Barré syndrome | 0.9323 | Anaphylaxis             | 0.8772  | Scombroid food poisoning      | 0.8328  |
| 5       | Chagas                     | 0.1918  | Anemia              | 0.1564  | Scombroid food poisoning | 0.0935 | Anaphylaxis             | 0.0691  | Cluster headache              | 0.0587  |

#### Test Set (5k Samples)
| Example | Diagnosis 1                | Prob. 1 | Diagnosis 2         | Prob. 2 | Diagnosis 3      | Prob. 3 | Diagnosis 4              | Prob. 4 | Diagnosis 5                   | Prob. 5 |
|---------|----------------------------|---------|---------------------|---------|------------------|---------|--------------------------|---------|-------------------------------|---------|
| 1       | Acute laryngitis           | 0.2447  | Viral pharyngitis   | 0.2333  | Chagas           | 0.1374  | Possible NSTEMI / STEMI | 0.1322  | Cluster headache              | 0.0690  |
| 2       | Possible NSTEMI / STEMI    | 0.7201  | Unstable angina     | 0.4661  | Stable angina    | 0.4127  | Viral pharyngitis        | 0.3728  | Acute laryngitis              | 0.3198  |
| 3       | Bronchitis                 | 0.8932  | Tuberculosis        | 0.4472  | Pneumonia        | 0.4262  | Pulmonary embolism       | 0.1449  | Pulmonary neoplasm            | 0.1069  |
| 4       | Possible NSTEMI / STEMI    | 0.6940  | Unstable angina     | 0.4755  | Stable angina    | 0.4326  | GERD                     | 0.3415  | Pulmonary embolism           | 0.2480  |
| 5       | Cluster headache           | 0.7120  | Anemia              | 0.5331  | Chagas           | 0.1463  | Bronchitis               | 0.0720  | Chronic rhinosinusitis        | 0.0454  |

### Model Trained on 20k Samples

#### Test Set (10k Samples)
| Example | Diagnosis 1                | Prob. 1 | Diagnosis 2         | Prob. 2 | Diagnosis 3      | Prob. 3 | Diagnosis 4              | Prob. 4 | Diagnosis 5                   | Prob. 5 |
|---------|----------------------------|---------|---------------------|---------|------------------|---------|--------------------------|---------|-------------------------------|---------|
| 1       | Cluster headache           | 0.4536  | Anemia              | 0.1508  | Bronchitis       | 0.0765  | Chagas                  | 0.0648  | Acute otitis media            | 0.0564  |
| 2       | Possible NSTEMI / STEMI    | 0.8689  | Unstable angina     | 0.6147  | Stable angina    | 0.5792  | Pericarditis            | 0.4361  | GERD                         | 0.4239  |
| 3       | Possible NSTEMI / STEMI    | 0.5201  | Unstable angina     | 0.2445  | Stable angina    | 0.2189  | Pulmonary embolism       | 0.2042  | Pericarditis                 | 0.0896  |
| 4       | Possible NSTEMI / STEMI    | 0.6947  | Unstable angina     | 0.4787  | Viral pharyngitis | 0.4692  | Stable angina           | 0.4326  | Acute laryngitis              | 0.3177  |
| 5       | Bronchitis                 | 0.4588  | Acute dystonic reactions | 0.3407 | Pulmonary embolism | 0.2220 | Myocarditis             | 0.1965  | Guillain-Barré syndrome       | 0.1942  |

## Conclusion

The fine-tuned BioBERT model demonstrates high predictive accuracy (~95%) and low Hamming loss (~0.048) in generating probabilistic differential diagnoses. This approach effectively addresses diagnostic uncertainty by providing a ranked list of potential causes.
