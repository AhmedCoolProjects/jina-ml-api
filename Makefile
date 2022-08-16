dev2:
	uvicorn --port 5000 --host 127.0.0.1 main:app --reload
dev:
	uvicorn main:app --reload
freeze:
	pip freeze > requirements.txt
install:
	pip install -r requirements.txt