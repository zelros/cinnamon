from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

clf = CatBoostClassifier()
clf.load_model('models/model_random_split')

from src.drift_explainer import DriftExplainer

with open('data/train_random_split.pkl', 'rb') as f:
    train = pkl.load(f)

with open('data/valid_random_split.pkl', 'rb') as f:
    valid = pkl.load(f)

with open('data/test_random_split.pkl', 'rb') as f:
    test = pkl.load(f)


num_features = [
    'caseData_life_BLOOD_PRESSURE_SYSTOLIC',
    'caseData_life_BLOOD_PRESSURE_DIASTOLIC',
    'caseData_life_AGE',
    'caseData_life_BMI',
    'caseData_life_WEIGHT',
    'caseData_life_HEIGHT',
    'caseData_life_TOTAL_CUMULATED_SUM_DC_PTIA_ACC',
    'caseData_life_TOTAL_CUMULATED_SUM_DC_PTIA_AM',
    'caseData_life_TOTAL_CUMULATED_SUM_IT_ACC',
    'caseData_life_TOTAL_CUMULATED_SUM_IT_AM',
    'ratio_poids_taille',
    'non_all_questions']

cat_features = ['caseData_case_CLIENT_PROCESS',
                'caseData_case_PRODUCT_NAME',
                'caseData_case_CHANNEL',
                'caseData_life_GENDER',
                'caseData_life_TITLE',
                'caseData_life_SMOKED_BQ',
                'caseData_case_INSTITUTION_CODE',
                'caseData_EQUIPMENT_CONTRACT0_OTHER_CONTRACT_RISK_NAME',
                'caseData_EQUIPMENT_CONTRACT0_NAME',
                'caseData_EQUIPMENT_CONTRACT0_OTHER_CONTRACT_CAUSE',
                'caseData_EQUIPMENT_CONTRACT0_OTHER_CONTRACT_RISK_TYPE',
                'caseData_EQUIPMENT_CONTRACT0_OTHER_CONTRACT_STATUS',
                'caseData_EQUIPMENT_CONTRACT0_OTHER_CONTRACT_STATUS_TITLE',
                'caseData_life_RISK_TYPES',
                'caseData_life_UNDERWRITING_APPLICATION',
                'caseData_case_CERAMIKPRO_FRUCTIPRO',
                'caseData_life_aggDC_PTIA_CAUSE',
                'caseData_life_aggDC_PTIA_RISK_NAME',
                'caseData_life_BLOOD_PRESSURE',
                'caseData_life_UNDER_TREATMENT',
                'underwriting_life_unansweredQuestions',
                'caseData_case_INSTITUTION',
                'Q_PY_BQ1_',
                'Q_PY_BQ2_',
                'Q_PY_BQ3_',
                'Q_PY_BQ4_',
                'Q_ADD_MED_INFO_BQ1_',
                'Q_ADD_MED_INFO_BQ2_',
                'Q_ADD_INFO_MED_BQ3_',
                'Q_ADD_INFO_MED_BQ4_',
                'Q_OVER10Y_SEC1_BQ1_',
                'Q_OVER10Y_SEC1_BQ2_',
                'Q_OVER10Y_SEC1_BQ3_',
                'Q_OVER10Y_SEC1_BQ4_',
                'Q_OVER10Y_SEC1_BQ5_',
                'Q_OVER10Y_SEC2BQ1_',
                'Q_OVER10Y_SEC2BQ2_',
                'Q_OVER10Y_SEC2BQ3_',
                'Q_OVER10Y_SEC2BQ4_',
                'Q_OVER10Y_SEC2BQ5_',
                'Q_OVER10Y_SEC2BQ6_',
                'Q_OVER10Y_SEC2BQ7_',
                'Q_OVER10Y_SEC2BQ8_',
                'Q_OVER10Y_SEC2BQ9_',
                'Q_OVER10Y_SEC2BQ10_']

nlp_te_FEATURES = ['concat_question_sup_1years_nlp_te_acc',
                   'concat_question_sup_1years_nlp_te_ref',
                   'concat_question_sup_1years_nlp_te_cs',
                   'concat_question_sup_add_info_nlp_te_acc',
                   'concat_question_sup_add_info_nlp_te_ref',
                   'concat_question_sup_add_info_nlp_te_cs',
                   'concat_question_sup_10years_nlp_te_acc',
                   'concat_question_sup_10years_nlp_te_ref',
                   'concat_question_sup_10years_nlp_te_cs']

drift_explainer = DriftExplainer()
drift_explainer.fit(model=clf,
                    X1=valid[num_features + cat_features + nlp_te_FEATURES],
                    X2=test[num_features + cat_features + nlp_te_FEATURES],
                    #y1=valid['target'],
                    #y2=test['target']
                    )


drift_explainer.generate_html_report('report.html')
