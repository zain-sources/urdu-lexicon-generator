import warnings
import warnings
warnings.filterwarnings("ignore")

import os
from flask import Flask, request ,  render_template, jsonify
from model import G2PModel
import re

UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'static\\downloads'
ALLOWED_EXTENSIONS = set(['txt'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
g2p_model = G2PModel("models")
g2p_model.load_decode_model()

spaceregex = re.compile("\s*")
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get('/')
def index():
    return render_template('index.html')

@app.post("/g2p")
def api():
    json = request.json
    text = spaceregex.split(json["text"].replace(u'۔',' '))
    out = g2p_model.decode(text)
    return '\n'.join(out)

@app.get('/file')
def files_endpoint():
    return render_template('file.html')

@app.post('/file/upload')
def upload_file():
    file = request.files['file']
    if file and file.filename.lower().endswith(('.txt')):
        # ceating locations
        filename = f"{os.urandom(16).hex()}.txt"
        up_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        down_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        # Create a unique filename
        file.save(up_dir)
        # Construct the download URL
        download_url = f"/static/downloads/{filename}"
        with open(down_dir, 'w') as dfile:
            with open(up_dir, 'r', encoding='utf-8') as ufile:
                ucontents = ufile.read()
                utext_list = spaceregex.split(ucontents.replace(u'۔',' '))
                pred = g2p_model.decode(utext_list)
                wcontent = '\n'.join(pred)
                dfile.write(wcontent)
        
        return jsonify({'downloadUrl': download_url}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False)
