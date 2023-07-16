from generate_aligned_faces import generate_aligned_faces
from  generate_feature_dictionary import generate_feature_dictionary
from face_recognizer import face_recognizer

image_path = "about_2.jpg"
face_path = "face/*.jpg"
generate_aligned_faces(image_path)
generate_feature_dictionary(face_path)
face_recognizer(image_path)

