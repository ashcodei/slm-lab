"""Rerun the complete independent audit with one explicit train-source helper qualification."""
from pathlib import Path

import audit_execution_code_feedback_v1 as audit
from slm_lab.execution_java_training_suites_v2 import TrainingSuites

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    audit.DATA=ROOT/'data/execution-code-feedback-v2'
    audit.REPORTS=ROOT/'reports/execution-code-feedback-v2'
    audit.PublishedSuites=TrainingSuites
    audit.main()
