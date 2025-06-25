# model.py

from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import EfficientDetLite0Spec

def get_spec():
    return EfficientDetLite0Spec()

def create_model(train_data, val_data, spec):
    model = object_detector.create(
        train_data=train_data,
        validation_data=val_data,
        model_spec=spec,
        epochs=40,
        batch_size=32,
        train_whole_model=True,
        do_train=True
    )

    return model