class DataConfig:
    RAW_DATA_PATH = '../OpenVideoSceneDetectionDataset/video/'
    PROCESSED_DATA_PATH = '../dataset/'
    SHOW_EXTRACTED = False


class ModelConfig:
    DATA_PATH = '../dataset/'
    SVM_PATH = '../svm.pickle'
    INFERENCE_DATA_PATH = '../demo.mp4'
