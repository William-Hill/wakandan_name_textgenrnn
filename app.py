from flask import Flask, request, jsonify
# from flask_cors import CORS
import wakandan_textgenrnn
app = Flask(__name__)


# model = wakandan_textgenrnn.load_model('wakandan_names_weights.hdf5', 'wakandan_names_vocab.json', 'wakandan_names_config.json')
model_api = wakandan_textgenrnn.get_model_api()

def get_name():
    name = wakandan_textgenrnn.generate_name(model, return_as_list=True)
    print("name:", name)

@app.route('/')
def hello_world():
    return 'Hello'

@app.route('/generate', methods=['POST'])
def generate():
    # get_name()
    # return name[0]
    output_data = model_api()
    response = jsonify(output_data)
    return response