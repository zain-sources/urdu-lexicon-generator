# Welcome to Lexicon Generator

This project is a **grapheme-to-phoneme (G2P)** converter for Urdu language. It can generate lexicons for Urdu words using a deep learning model.

## Files

- `main.py`: This is a summarized inference code implementation.
- `app.py`: This is a complete flask app and an API to generate lexicons.

## Web App

You can run this app by the following command:

```bash
python app.py
```
This will open a nice web UI for you.

## API Endpoint

The API is also defined in `app.py` and when we run the web UI, the API also runs. You can access the API at `http://localhost:5000/g2p` using a POST request. This API requires a JSON object like this:

```json
{"text": "رکوا\nایران\nخرید"}
```

Each word is separated by a newline character (`\n`).

## Steps to Run this Model

1. Create an Anaconda environment with Python 3.6.

```bash
conda create -n env_name python=3.6 anaconda
```

2. Reach inside this directory and activate the conda environment:

```bash
conda activate env_name
```

3. Run the following command to install TensorFlow 1.0.0:

```bash
pip install tensorflow-1.0.0-cp36-cp36m-win_amd64.whl
```

4. Run the following command to install other requirements:

```bash
pip install -r requirements.txt
```

5. Now you are ready to go:

```bash
python app.py
```


I hope it was helpful. 😊
