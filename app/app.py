from flask import Flask, request, jsonify, render_template, redirect, make_response, url_for
from PIL import Image
from werkzeug.utils import secure_filename
from srcnn import SRCNN
from torchvision.utils import save_image
import torch
import os
import numpy as np


app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "static/uploads"
app.config["IMAGE_GENERATED"] = "static/outputs"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["png", "jpg", "jpeg"]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def SRCNN_image(filename, device):
    model = SRCNN().to(device)
    model.load_state_dict(torch.load('static/model/SRCNN_model.pth', map_location=device))
    image = Image.open(app.config['IMAGE_UPLOADS'] + "/" + filename).convert('RGB')
    image = np.array(image, dtype=np.float32)
    image /= 255.
    image = image.transpose([2, 0, 1])
    mat = torch.tensor(image, dtype=torch.float)
    image_data = mat.to(device)
    output = model(image_data)
    save_image(output, f"static/outputs/{filename}")
    return filename


def allowed_image(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.lower() not in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return False
    return True


@app.route('/')
def home():
    return render_template('upload_image.html')


@app.route("/", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files["image"]

    if image.filename == "":
        print("invalid filename")
        return redirect(request.url)

    if not allowed_image(image.filename):
        print("extension not allowed")
        return redirect(request.url)

    else:
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
        SRCNN_image(filename, device)

        return render_template("./upload_image.html", filename=filename)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename="uploads/"+filename, code=301))


@app.route('/display_enhanced/<filename>')
def display_new_image(filename):
    return redirect(url_for('static', filename="outputs/"+filename, code=301))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
