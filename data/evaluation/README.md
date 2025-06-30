# Evaluation Process for SofAgent

This folder contains the questions and responses used to evaluate the performance of the `SofAgent` system and the structure of the data files contained in this directory.

## Overview

The primary goal of this evaluation is to benchmark the performance of `SofAgent` against a single-agent baseline powered by GPT-4o. The evaluation focuses on two key hypotheses:
1.  A multi-agent architecture enhances the accuracy of information retrieval.
2.  Grounding the system in a factual knowledge base mitigates LLM hallucinations.

The evaluation is performed within the context of a high-end furniture retail environment (Natuzzi).

### Metrics

1.  **Accuracy**: A binary judgment (`TRUE`/`FALSE`) indicating whether the system's response was factually correct, complete, and fully addressed the user's query.

2.  **Error Classification**: For incorrect responses, one of the following error codes was assigned:
    -   **`FIH` (Factual Inaccuracy & Hallucination)**: The system provided information that was verifiably false or invented details not present in the knowledge base (e.g., non-existent products, incorrect prices).
    -   **`FMR` (Failure to Meet User Requirements)**: The system failed to retrieve relevant information that *was* present in the knowledge base, ignored a user constraint, or misinterpreted the query's intent.
    -   **`RGI` (Response Generation Issue)**: A technical failure occurred, such as exceeding the model's token limit, resulting in an incomplete or failed response.

This folder contains a complete, auditable record of the evaluation and are used to generate the summary statistics presented in the paper.
