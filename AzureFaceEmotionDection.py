from os.path import dirname, join, realpath, basename
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import FaceAttributeType

# NOTE: for this component to work, the following pip command must be used:
# pip install --upgrade azure-cognitiveservices-vision-face

def get_face_emotion_information(input_file_path):
    """This function takes in a full file location and sends it to the Microsoft Azure API to detect faces within the image
    and then detect the emotions that said face has. The overall emotion analysis of all faces is then returned in a dictionary.
    The returned dictionary is a dictionary of dictionaries, with each inner dictionary holding the results of a single detected face.

    input_file_path - A string representation of the full path of the image
    
    Emotion analysis results are represented by a numerical float value between 0 and 1, with the added total of all 8 emotions equal to 1.

    The first dictionary (face1) in the returned dictionary holds the primary(largest area of the face bounding box) found by the API.
    All other face information is secondary or has a smaller bounding box.
    
    Note: .png, .jpg, or .gif file type only. Allowed file size is from 1KB to 6MB"""

    # The resource key for the MistyFaceEmotionDetection Resource on Azure
    KEY = "1837b9d29e0b4a22843d103a7ca8b3c9"

    # The variable is for the ACS API, created with the cognitive services resource
    ENDPOINT = "https://westus2.api.cognitive.microsoft.com/"

    # Authenticating the Client
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    # Creating an image stream given an input image file.
    # (returns only the number of faces found, up to 100)
    with open(input_file_path, "rb") as face_fd:
        detected_faces = face_client.face.detect_with_stream(face_fd, return_face_attributes=FaceAttributeType.emotion)

    # Variables used later on.
    image_name = basename(input_file_path)
    returning_dictionary = {}
    count = 0

    # Case: There are no faces detected in the image.
    if not detected_faces:
        raise Exception('No face detected from image {}'.format(input_file_path))

    # Creating a unique dictionary object for each face detected from the given file and adding it 
    # to the returning_dictionary dictionary.
    for face in detected_faces:
        count += 1
        Dict = {}
        Dict["anger"] = face.face_attributes.emotion.anger
        Dict["contempt"] = face.face_attributes.emotion.contempt
        Dict["disgust"] = face.face_attributes.emotion.disgust
        Dict["fear"] = face.face_attributes.emotion.fear
        Dict["happiness"] = face.face_attributes.emotion.happiness
        Dict["neutral"] = face.face_attributes.emotion.neutral
        Dict["sadness"] = face.face_attributes.emotion.sadness
        Dict["surprise"] = face.face_attributes.emotion.surprise
        returning_dictionary["face"+str(count)] = Dict
    return returning_dictionary

# Example of usage:
#
# from AzureFaceEmotionDection import get_face_emotion_information 
# get_face_emotion_information("C:/Users/Carson/Pictures/ImageName.png")
