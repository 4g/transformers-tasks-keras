from tensorflow import lite as tflite
import cv2
import numpy as np

class TFLiteModel:
    def load_model(self, model_path):
        self.interpreter = tflite.Interpreter(
            model_path=model_path, num_threads=4)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        return self


    def get_model_output(self):
        image = np.zeros(tfutil.input_shape, dtype=np.int64)
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.set_tensor(self.input_details[1]['index'], image)


        self.interpreter.invoke()

        outputs = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        outputs = [np.squeeze(x) for x in outputs]
        return outputs

    def get_model_details(self):
        return self.interpreter.get_tensor_details()

    def model_path(self):
        return None


if __name__ == "__main__":
    import argparse, os, glob, traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="pass either a directory or a tflite model", required=True)

    args = parser.parse_args()

    path = args.model
    try:
        tfutil = TFLiteModel().load_model(path)
        print(tfutil.input_details)

        from tqdm import tqdm
        for i in tqdm(range(10000)):
            output = tfutil.get_model_output()

    except:
        print("Exception in user code:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
