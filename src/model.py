from tensorflow import keras
from tensorflow.keras import layers

HIDDEN_UNITS = [256, 128, 64, 32, 16]


def build_ann_model(
    input_dim: int,
    n_classes: int = 4,
    use_dropout: bool = False,
    use_batchnorm: bool = False,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    name: str = 'credit_risk_ann',
) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name='input_features')
    x = inputs
    for i, units in enumerate(HIDDEN_UNITS, start=1):
        x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f'batchnorm_{i}')(x)
        if use_dropout:
            x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


EXPERIMENTS = {
    'E1': {'description': 'Baseline ANN', 'use_dropout': False, 'use_batchnorm': False, 'use_class_weight': False},
    'E2': {'description': 'ANN + Dropout', 'use_dropout': True, 'use_batchnorm': False, 'use_class_weight': False},
    'E3': {'description': 'ANN + Batch Normalization', 'use_dropout': False, 'use_batchnorm': True, 'use_class_weight': False},
    'E4': {'description': 'ANN + Dropout + Batch Normalization', 'use_dropout': True, 'use_batchnorm': True, 'use_class_weight': False},
    'E5': {'description': 'ANN + Class Weight', 'use_dropout': False, 'use_batchnorm': False, 'use_class_weight': True},
    'E6': {'description': 'ANN + Dropout + Class Weight', 'use_dropout': True, 'use_batchnorm': False, 'use_class_weight': True},
}
