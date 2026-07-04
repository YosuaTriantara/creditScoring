"""
src/model_v2.py — daftar eksperimen untuk pipeline 34 fitur (jalur 'b').
Arsitektur ANN tetap reuse dari model.build_ann_model (TIDAK diduplikasi).
"""

EXPERIMENTS_V2 = {
    'E1': {'description': 'Baseline ANN (34 fitur)',                  'use_dropout': False, 'use_batchnorm': False, 'use_class_weight': False, 'use_smote': False},
    'E2': {'description': 'ANN + Dropout',                            'use_dropout': True,  'use_batchnorm': False, 'use_class_weight': False, 'use_smote': False},
    'E3': {'description': 'ANN + Batch Normalization',                'use_dropout': False, 'use_batchnorm': True,  'use_class_weight': False, 'use_smote': False},
    'E4': {'description': 'ANN + Dropout + Batch Normalization',      'use_dropout': True,  'use_batchnorm': True,  'use_class_weight': False, 'use_smote': False},
    'E5': {'description': 'ANN + Class Weight',                      'use_dropout': False, 'use_batchnorm': False, 'use_class_weight': True,  'use_smote': False},
    'E6': {'description': 'ANN + Dropout + Class Weight',             'use_dropout': True,  'use_batchnorm': False, 'use_class_weight': True,  'use_smote': False},
    'E7': {'description': 'ANN + SMOTE (arsitektur sama seperti E1)', 'use_dropout': False, 'use_batchnorm': False, 'use_class_weight': False, 'use_smote': True},
}
