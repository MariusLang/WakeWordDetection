from model.wake_word_cnn import WakeWordCNN
from model.crnn_with_mbconv import CRNN_with_MBConv
from model.crnn_with_mbconv_non_gru import CRNN_Own_GRU

MODEL_REGISTRY = {
    'cnn': WakeWordCNN,
    'crnn': CRNN_with_MBConv,
    'crnn_own_gru': CRNN_Own_GRU,
}


def get_model(model_name, input_shape, num_classes):
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(input_shape=input_shape, num_classes=num_classes)


def list_models():
    return list(MODEL_REGISTRY.keys())
