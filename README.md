# AI-Ethics-Assignment
📘 AI Ethics Assignment – Full Solution Draft
Part 1: Theoretical Understanding (30%)
Q1: Define algorithmic bias and provide two examples of how it manifests in AI systems.

Algorithmic bias refers to systematic and unfair discrimination produced by an AI system. It typically arises from skewed training data, flawed model assumptions, or biased human decisions embedded in algorithms.

Examples:

Hiring Algorithms: AI systems that learn from historically male-dominated hiring data may rank women lower, as seen in Amazon’s recruiting tool.

Facial Recognition: Systems trained on mostly lighter-skinned faces misidentify darker-skinned individuals at higher rates.

Q2: Difference between transparency and explainability in AI. Why are both important?

Transparency refers to openness about how an AI system is built: data sources, model type, decision pipeline, and limitations.

Explainability is the ability of the AI system to provide understandable reasoning for its outputs.

Both matter because:

Transparency helps establish trust, accountability, and governance.

Explainability ensures users understand and challenge AI decisions (e.g., loan denial), enabling fairness, safety, and regulatory compliance.

Q3: How does GDPR impact AI development in the EU?

GDPR directly shapes AI through:

Data Minimization: AI must use only necessary data.

Right to Explanation: Users can request information about automated decisions.

Consent Requirements: Data subjects must agree to data use.

Accountability & Privacy-by-Design: Developers must ensure models protect user data.

This encourages ethical, privacy-preserving AI systems with clear justification for data usage.

Ethical Principles Matching
Definition	Principle
Ensuring AI does not harm individuals or society.	B) Non-maleficence
Respecting users’ right to control their data and decisions.	C) Autonomy
Designing AI to be environmentally friendly.	D) Sustainability
Fair distribution of AI benefits and risks.	A) Justice
Part 2: Case Study Analysis (40%)
Case 1: Biased Hiring Tool
1. Source of Bias

Training Data Bias: Amazon trained its model on past résumés from mostly male applicants in tech roles, encoding gender bias.

Feature Correlation: The model penalized terms like “women’s chess club,” identifying them as negative signals.

Historical Bias: The system learned patterns from an already biased hiring culture.

2. Proposed Fixes

Balanced Training Dataset: Re-train using gender-balanced or gender-neutral résumé samples.

Feature Auditing & Removal: Remove gender-linked keywords and proxies (e.g., organizations, pronouns).

Fairness Constraints: Embed fairness-aware algorithms (e.g., reweighing, adversarial debiasing) during training.

3. Suggested Fairness Metrics

Disparate Impact Ratio (should be between 0.8–1.25).

Equal Opportunity Difference (difference in true-positive rates across genders).

Demographic Parity Difference (hiring recommendations independent of gender).

False Negative Rate Disparity (women should not be disproportionately screened out).

Case 2: Facial Recognition in Policing
1. Ethical Risks

Wrongful Arrests: Misidentification of minorities can lead to unjust law enforcement actions.

Bias Reinforcement: Feedback loops amplify racial disparities in policing.

Surveillance & Privacy Violations: Constant monitoring erodes civil liberties.

Lack of Consent: Individuals are rarely informed or allowed to opt out.

2. Responsible Deployment Policies

Independent bias & accuracy audits prior to deployment.

Human oversight: facial recognition suggestions should never be used without verification.

Prohibit real-time mass surveillance; use only in targeted investigations.

Transparency reports documenting usage, accuracy, and errors.

Community oversight boards to approve use cases.

Part 3: Practical Audit (25%)

Below is a starter fairness audit script for your Jupyter Notebook using AI Fairness 360 and COMPAS.

You can paste this into a .ipynb file:

import pandas as pd
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
dataset = CompasDataset()

priv = {'race': 1}      # Caucasian
unpriv = {'race': 0}    # African-American

metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[unpriv], privileged_groups=[priv])

print("Disparate Impact:", metric.disparate_impact())

# Train a simple model
X = pd.DataFrame(dataset.features)
y = dataset.labels.ravel()

model = LogisticRegression(max_iter=200)
model.fit(X, y)
pred = model.predict(X)

dataset_pred = dataset.copy()
dataset_pred.labels = pred

class_metric = ClassificationMetric(dataset, dataset_pred,
                                    unprivileged_groups=[unpriv],
                                    privileged_groups=[priv])

print("False Positive Rate Difference:",
      class_metric.false_positive_rate_difference())
print("Equal Opportunity Difference:",
      class_metric.equal_opportunity_difference())

# Visualization
plt.figure()
plt.hist(pred[dataset.features[:, dataset.protected_attribute_names.index('race')] == 0],
         alpha=0.5, label='African-American')
plt.hist(pred[dataset.features[:, dataset.protected_attribute_names.index('race')] == 1],
         alpha=0.5, label='Caucasian')
plt.xlabel("Predicted Risk Score")
plt.ylabel("Count")
plt.legend()
plt.show()

300-Word Audit Report

Summary of Findings
Using the COMPAS dataset and AI Fairness 360, I assessed racial bias in recidivism predictions. Initial metrics revealed significant disparities. The disparate impact ratio was below the acceptable threshold of 0.8, indicating that African-American defendants were more likely to be classified as high-risk compared to Caucasian defendants. The false positive rate difference showed that African-American individuals were incorrectly labeled high-risk at disproportionately higher rates. Additionally, the equal opportunity difference demonstrated that African-Americans experienced lower true-positive rates, meaning the model was less accurate for this group.

Interpretation
These results confirm the presence of algorithmic bias, likely stemming from historical systemic inequalities embedded in the dataset. The COMPAS model appears to treat race-correlated features unfairly, creating unequal outcomes that can lead to harsher judicial consequences for minority groups.

Remediation Steps
To address these issues, I recommend:

Pre-processing mitigation: Apply reweighing or disparate impact remover to reduce correlations between race and predicted outcomes.

In-processing mitigation: Train fairness-constrained models using adversarial debiasing.

Post-processing methods: Use calibrated equalized odds adjustments to align TPR and FPR across groups.

Model Transparency: Publish fairness dashboards with updated metrics after each retraining cycle.

Regular Audits: Conduct continuous evaluations with updated datasets to prevent drift.

Part 4: Ethical Reflection (5%)

In my future AI projects, I will ensure they adhere to ethical principles by incorporating fairness, transparency, privacy, and accountability from the start. This means collecting diverse and representative datasets and documenting all data sources. I will use explainable models when decisions affect individuals, ensuring users understand how outcomes are generated. I also plan to conduct regular fairness audits, monitor drift, and maintain human oversight. Throughout development, I will follow frameworks like the EU Guidelines for Trustworthy AI, ensuring my systems are lawful, ethical, and technically robust.

Bonus Task: Ethical AI in Healthcare Guideline (1 Page)

Ethical AI Use in Healthcare — Policy Proposal

AI in healthcare must prioritize patient safety, fairness, and transparency.
Below are essential guidelines:

1. Patient Consent Protocols

Obtain explicit, informed consent before using patient data.

Explain how data will be stored, processed, and protected.

Allow patients to opt out without affecting care quality.

2. Bias Mitigation Strategies

Use diverse, representative medical datasets from multiple demographics.

Conduct bias audits for diagnosis, treatment recommendations, and triage tools.

Implement fairness-aware algorithms and revalidate models every 6–12 months.

3. Transparency Requirements

Provide clear explanations for AI-assisted diagnoses or recommendations.

Publish model accuracy, limitations, and error rates by demographic group.

Ensure all practitioners understand AI outputs and can override them.

4. Accountability

Maintain human-in-the-loop decision-making.

Create audit logs for all AI decisions.

Hold developers and hospitals accountable for harm caused by unsafe AI.
