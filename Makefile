venv:
	python3 -m venv venv

install:
	. venv/bin/activate && pip install -r requirements.txt
