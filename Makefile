ali:
	pipenv run python generate_aligned_faces.py about_1.jpg 

rec:
	pipenv run python face_recognizer.py

fea:
	pipenv run python generate_feature_dictionary.py face001.jpg

d-build:
	docker build -t linux-python-api --platform linux/x86_64 . 