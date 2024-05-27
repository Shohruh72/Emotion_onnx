import torch
from PIL import Image

from util import *

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

detector = FaceDetector('./weights/detection.onnx')
fer = HSEmotionRecognizer('./weights/emotion.onnx')


def detect_face(frame):
    boxes = detector.detect(frame, (640, 640))
    return boxes if boxes is not None and len(boxes) else None


stream = cv2.VideoCapture(-1)
frame_width = int(stream.get(3))
frame_height = int(stream.get(4))
fps = stream.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

while True:
    ret, frame = stream.read()
    if ret:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detect_face(image)

        if boxes is not None:
            for box in boxes.astype('int32'):
                x1, y1, x2, y2 = box[:4]
                face_image = image[y1:y2, x1:x2]
                pil_image = Image.fromarray(face_image).convert('RGB')
                pil_image = pil_image.resize((224, 224))  # Adjust size as per model requirements

                emotion, scores = fer.predict_emotions(np.array(pil_image), logits=False)
                draw_emotion_bars(fer, frame, scores, (x2 + 10, y1), bar_height=15, width=100)

        cv2.imshow('Emotion Recognition', cv2.resize(frame, (1280, 960)))
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Camera index error')
        break

stream.release()
out.release()
cv2.destroyAllWindows()
