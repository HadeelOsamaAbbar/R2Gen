from flask import Flask, jsonify,  redirect, url_for, request, render_template
from PIL import Image
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from main_test import *
import time
from user_keywords import *
from search import *
from full_report_translate import *
app = Flask(__name__)

# # preprocess Image #

# # get predictions #


def preprocess_Image(path1, path2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485),
                             (0.229))])
    image_1 = Image.open(path1).convert("RGB")  # (512, 420)
    image_2 = Image.open(path2).convert("RGB")  # (512, 420)
    image_1 = transform(image_1)  # [3, 224, 224]
    image_2 = transform(image_2)  # [3, 224, 224]
    image = torch.stack((image_1, image_2), 0)  # [2, 3, 224, 224]
    return image


def make_prediction(path1, path2, model):
    image = preprocess_Image(path1, path2).unsqueeze(0)  # [2,3,224,224]
    # weight of size [64, 3, 7, 7]
    output = model(image.to('cpu'), mode='sample')
    report = model.tokenizer.decode_batch(output.cpu().numpy())
    return report
# Load your trained model


# parse arguments
args = parse_agrs()

# fix random seeds
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# create tokenizer
tokenizer = Tokenizer(args)
model = R2GenModel(args, tokenizer)
load_path = "results/iu_xray/model_best.pth"
# model = model.load("results/iu_xray/model_best.pth",map_location=torch.device('cpu'))
checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
# @app.route('/')
# def hello():
#     return 'Hello World!'


@app.route('/')
def index():
    # Main page
    print("helloooo")
    return render_template('home.html')

# move from home page to doctor page:


@app.route('/getStart', methods=['GET', 'POST'])
def getStart():
    if request.method == 'POST':
        print("starttttt")
        return render_template('index.html')


# @app.route('/')
# def doctorPage():
#     print("doctorPage??????")
#     return render_template('index.html')

# function  accepts only POST requests:
@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        seconds1 = time.time()
        # Get the file from post request
        f = []
        for item in request.files.getlist('file1'):
            f.append(item)
        f1 = request.files['file1']
        f2 = request.files['file1']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path1 = os.path.join(
            basepath, 'uploads', secure_filename(f[0].filename))
        file_path2 = os.path.join(
            basepath, 'uploads', secure_filename(f[1].filename))
        f[0].save(file_path1)
        f[1].save(file_path2)

        # Make prediction
        pred_report = make_prediction(file_path1, file_path2, model)

        # result = str(pred_class[0][0][1])               # Convert to string
        seconds2 = time.time()
        print("totalTime: ", seconds2-seconds1)
        importantKeywords = getkewords(pred_report[0])
        res = pred_report[0]  # + "\n\n Important Keywords: "
        keys = ""
        source_lang = 'english'
        target_lang = 'arabic'

        for word in importantKeywords:
            keys += word + " : "
            all_possible_meaing = understand_words(
                word, source_lang, target_lang)
            all_possible_meaing = all_possible_meaing[:3]
            del all_possible_meaing[0]
            L = len(all_possible_meaing)
            count = 0
            for term in all_possible_meaing:

                keys += term
                count += 1
                if count <= L-1:
                    keys += " - "
            keys += " <br>    "
        # res[-1].'.'
        source_lang = 'en'
        target_lang = 'ar'
        full_report = understand_report(res, source_lang, target_lang)

        returnValue = res + "," + "Important Keywords: "+","+keys+"," + full_report

        # print(all_possible_meaing)

        return returnValue
    return None


if __name__ == '__main__':
    app.run()


# #  run a Flask development server by typing
# # $ FLASK_ENV=development FLASK_APP=app.py flask run

# # When you visit http://localhost:5000/ in your web browser, you will be greeted with Hello World! text
