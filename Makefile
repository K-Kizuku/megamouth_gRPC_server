ali:
	pipenv run python generate_aligned_faces.py about_1.jpg 

rec:
	pipenv run python face_recognizer.py

fea:
	pipenv run python generate_feature_dictionary.py face001.jpg