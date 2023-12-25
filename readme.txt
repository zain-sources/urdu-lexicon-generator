                    --------------Welcome to Lexicon generator-----------------
main.py  -- this is a summerized inference code implementation
app.py   -- this is a complete flask app and an api to generete lexicons
	    # Web App
		you can run this app by (python app.py) command. this will open a nice web ui for you.
	    # API enpoint
		api is also defined in app.py and when we run web ui, api also runs
		you can access the api at (http://locahost:5000/g2p) using post request
		this api requires a json object like this ie. {"text": "رکوا\nایران\nخرید"} each word is seperated by a newline character (\n)





#######( Steps to run this Model )########
1) create an anaconda environment with python 3.6



2) reach inside this directory and activate conda invironment.
	conda activate env_name



3) run below command to install requirements.txt
	pip install tensorflow-1.0.0-cp36-cp36m-win_amd64.whl



4) run below command to install requirements.txt
	pip install -r requirements.txt


5) Now you are ready to go
	python app.py