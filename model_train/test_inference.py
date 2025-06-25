import os
import cv2
import numpy as np
import tensorflow as tf
from logger import get_logger

def apply_nms(boxes, scores, classes, image_shape, iou_threshold=0.5):
    h, w = image_shape[:2]
    ymins = boxes[:, 0] * h
    xmins = boxes[:, 1] * w
    ymaxs = boxes[:, 2] * h
    xmaxs = boxes[:, 3] * w
    pixel_boxes = np.stack([xmins, ymins, xmaxs, ymaxs], axis=1)

    indices = tf.image.non_max_suppression(
        boxes=pixel_boxes,
        scores=scores,
        max_output_size=20,
        iou_threshold=iou_threshold
    ).numpy()

    return boxes[indices], scores[indices], classes[indices]

def run_inference(interpreter, image):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 전처리
    h_input, w_input = input_details[0]['shape'][1:3]
    resized = cv2.resize(image, (w_input, h_input))
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 출력 매핑 (로그로 확인한 순서에 맞춰 인덱스 지정)
    # output_details 순서: [scores, boxes, count, classes]
    scores  = interpreter.get_tensor(output_details[0]['index'])[0]          # [N]
    boxes   = interpreter.get_tensor(output_details[1]['index'])[0]          # [N,4]
    count   = int(interpreter.get_tensor(output_details[2]['index'])[0])     # scalar
    classes = interpreter.get_tensor(output_details[3]['index'])[0].astype(int)  # [N]

    # 실제 검출 개수만큼 자르기
    return boxes[:count], classes[:count], scores[:count]

def draw_predictions(image, boxes, classes, scores, class_names, threshold=0.5):
    h, w, _ = image.shape
    for box, cls, score in zip(boxes, classes, scores):
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        if not (0.0 <= ymin <= 1.0 and 0.0 <= xmin <= 1.0 and
                0.0 <= ymax <= 1.0 and 0.0 <= xmax <= 1.0):
            continue

        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)

        label = f"{class_names[cls]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label,
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    return image

def main():
    os.makedirs('vis_results', exist_ok=True)
    logger = get_logger('test_logger', log_file='test_log/test.log')

    interpreter = tf.lite.Interpreter(model_path='exported_model/model.tflite')
    interpreter.allocate_tensors()
    logger.info(f"Output details: {interpreter.get_output_details()}")
    logger.info("TFLite model loaded")

    class_names    = ['fire', 'smoke']
    test_image_dir = 'dataset/test'

    # 테스트 폴더 내 모든 이미지 파일 순회
    for filename in sorted(os.listdir(test_image_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(test_image_dir, filename)
        image_bgr  = cv2.imread(image_path)
        if image_bgr is None:
            logger.warning(f"이미지 읽기 실패, 건너뜀: {filename}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, classes, scores = run_inference(interpreter, image_rgb)
        # boxes, scores, classes = apply_nms(
        #     np.array(boxes),
        #     np.array(scores),
        #     np.array(classes),
        #     image_rgb.shape
        # )

        logger.info(f"Processing {filename} ({len(boxes)} detections)")
        for i in range(len(boxes)):
            logger.info(
                f"  [{i}] {class_names[classes[i]]} "
                f"score={scores[i]:.2f}, box={boxes[i]}"
            )

        vis = draw_predictions(image_rgb.copy(), boxes, classes, scores, class_names)
        save_path = os.path.join('vis_results', filename)
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    logger.info("모든 이미지 처리 완료")

if __name__ == '__main__':
    main()
