# Error Analysis — ChronicGuard AI

## Overview

- Total risk classification errors: 0
- Total intent classification errors: 0
- Safety-critical false negatives (high/urgent predicted as low/medium): 0

## Safety-Critical Misclassifications

These are the most dangerous errors: cases where the model predicted low or medium risk
for a message that was actually high or urgent. In a production system, these could
result in delayed care or missed escalation.

## Most Common Risk Errors


## Most Common Intent Errors


## Safety Implications

The primary concern in this system is the downward misclassification of urgent and high-risk messages. Upward misclassification (predicting high when actually low) results in unnecessary care manager workload but does not create patient safety risk. The model is tuned to minimize downward errors at the cost of precision on the low and medium classes.
