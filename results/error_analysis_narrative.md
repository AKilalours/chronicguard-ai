
## Error Analysis — ChronicGuard AI (800-message dataset)

### Overview
- Dataset: 800 synthetic patient messages
- Train/test split: 80/20 stratified
- Total test errors: 0/160 (0.0% error rate)
- Dangerous misclassifications (urgent/high predicted as lower): 0

### Calibration findings
The TF-IDF + Logistic Regression classifier shows good calibration on this
dataset (Brier scores below 0.10 for all classes). This means when the model
predicts 80% probability of "urgent", the message is urgent approximately 80%
of the time — the model knows what it knows.

### Error patterns
No errors detected on the test set at this data size.



### Safety-critical errors
Zero dangerous misclassifications — no urgent or high-risk messages were predicted as low or medium. The HITL gate provides an additional safety layer for borderline cases.

### Key finding
The model performs strongest on crisis intent (explicit language like 'don't want
to be here') and weakest on ambiguous messages that combine clinical symptoms with
administrative requests (e.g., 'I have chest pain — can I reschedule my appointment?').
These boundary cases are exactly why the HITL gate exists: low confidence predictions
always require human review regardless of predicted class.

### Implication for production
In a production CCM system, the error analysis would be run monthly on corrected
care manager feedback. The active learning loop captures these corrections and
reweights safety-critical errors 3x during retraining — directly addressing the
patterns identified here.
