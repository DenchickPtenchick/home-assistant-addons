from flask import Flask, request, jsonify
import numpy as np
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/infer", methods=["POST"])
def infer():
    data = request.json.get("input")
    if not data:
        return jsonify({"error": "Missing 'input'"}), 400
    input_data = np.array(data, dtype=np.float32).reshape(input_details[0]["shape"])
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return jsonify({"output": output_data.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
