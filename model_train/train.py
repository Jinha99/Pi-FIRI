# train.py

from tflite_model_maker import object_detector, model_spec
from tflite_model_maker.config import ExportFormat

from utils.logger import get_logger
from utils.plot import plot_training_metrics
from model import create_model

def main():
    logger = get_logger('train_logger')

    # 1) Load Data
    train_data = object_detector.DataLoader.from_pascal_voc(
        'dataset/train',
        'dataset/train',
        ['Fire', 'Smoke']
    )

    val_data = object_detector.DataLoader.from_pascal_voc(
        'dataset/valid',
        'dataset/valid',
        ['Fire', 'Smoke']
    )

    logger.info("Start Training...")

    # 2) Set EfficientDet Lite0
    spec = model_spec.get('efficientdet_lite0')

    # 3) Training Model
    model = create_model(train_data, val_data, spec)

    logger.info("Training Complete")

    history = model.model.history.history
    for epoch, _ in enumerate(history["loss"]):
        log_line = f"Epoch {epoch+1} - " + ", ".join([f"{k}: {v[epoch]:.4f}" for k, v in history.items()])
        logger.info(log_line)

    # 4) Evaluate & Save Plot
    eval_result = model.evaluate(val_data)
    logger.info(f"Evaluation Results: {eval_result}")

    plot_training_metrics(history, save_path='training_logs/loss_plot.png')
    logger.info("Result Plot Saved")

    # 5) Model Export
    model.export(
        export_dir='exported_model',
        export_format=[ExportFormat.TFLITE]
    )
    
    logger.info("Model Export Complete")
    

if __name__ == '__main__':
    main()