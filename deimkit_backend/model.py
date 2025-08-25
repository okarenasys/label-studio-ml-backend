from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

ort.preload_dlls()

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    session = None
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

        self.session = self.load_model()

    def load_model(self):
        providers = ["CPUExecutionProvider"]
        try:
            session = ort.InferenceSession("model.onnx", providers=providers)
            return session
        except Exception as e:
            print("Error while loading the model:", e)
            return None

    def load_image_from_url(self, url: str) -> np.ndarray:
        headers = {}
        api_key = os.environ.get("LABEL_STUDIO_API_KEY")
        if api_key:
            headers["Authorization"] = f"Token {api_key}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise if unauthorized or not found

        image_np = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]

        predictions = []

        # Load class names
        try:
            with open("classes.txt", "r") as f:
                class_names = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Could not load class names: {e}")
            return ModelResponse(predictions=[])

        label_studio_url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")

        for task in tasks:
            image_url = task["data"]["image"]
            if image_url.startswith("/data/local-files/"):
                full_url = label_studio_url.rstrip("/") + image_url
            else:
                full_url = image_url

            image = self.load_image_from_url(full_url)

            if image is None:
                raise RuntimeError(f"Failed to read image: {full_url}")

            detections = self.run_inference(
                frame=image,
                session=self.session,
                threshold=0.5,
                target_width=640
            )

            img_h, img_w = image.shape[:2]

            # Format Label Studio results
            ls_results = []
            for det in detections:
                x_min, y_min, x_max, y_max = det["bbox"]
                width = x_max - x_min
                height = y_max - y_min

                ls_results.append({
                    "from_name": "label",  # Must match config
                    "to_name": "image",  # Must match config
                    "type": "rectanglelabels",
                    "value": {
                        "x": (x_min / img_w) * 100,
                        "y": (y_min / img_h) * 100,
                        "width": (width / img_w) * 100,
                        "height": (height / img_h) * 100,
                        "rectanglelabels": [class_names[det["class_id"]]]
                    }
                })

            predictions.append({
                "model_version": "1.0",
                "score": max([d["score"] for d in detections], default=0.0),
                "result": ls_results
            })

            print("Predictions returned to Label Studio:")
            print(json.dumps(predictions, indent=2))
        
        return ModelResponse(predictions=predictions)

    def format_prediction(self, bbox, class_id, class_names, image_shape):
        x_min, y_min, x_max, y_max = bbox
        img_h, img_w = image_shape[:2]

        return {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": (x_min / img_w) * 100,
                "y": (y_min / img_h) * 100,
                "width": ((x_max - x_min) / img_w) * 100,
                "height": ((y_max - y_min) / img_h) * 100,
                "rectanglelabels": [class_names[class_id]]
            }
        }

    def run_inference(
            self,
            frame: np.ndarray,
            session: ort.InferenceSession,
            threshold: float,
            target_width: int
    ):
        height, width = frame.shape[:2]
        scale = target_width / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)

        y_offset = (target_width - new_height) // 2
        x_offset = (target_width - new_width) // 2

        model_input = np.zeros((target_width, target_width, 3), dtype=np.uint8)
        model_input[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = cv2.resize(frame, (new_width, new_height))

        im_data = np.ascontiguousarray(model_input.transpose(2, 0, 1), dtype=np.float32)
        im_data = np.expand_dims(im_data, axis=0)
        orig_size = np.array([[target_width, target_width]], dtype=np.int64)

        input_name = session.get_inputs()[0].name
        outputs = session.run(
            output_names=None,
            input_feed={input_name: im_data, "orig_target_sizes": orig_size},
        )

        labels, boxes, scores = outputs
        boxes = boxes[0]
        scores = scores[0]
        labels = labels[0]

        # Scale boxes back
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_offset) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_offset) / scale

        results = []
        for i in range(len(boxes)):
            if scores[i] >= threshold:
                results.append({
                    "bbox": boxes[i].tolist(),
                    "class_id": int(labels[i]),
                    "score": float(scores[i])
                })
        return results
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

