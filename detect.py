# import cv2
# import math
# import argparse

# def highlightFace(net, frame, conf_threshold=0.7):
#     frameOpencvDnn = frame.copy()
#     frameHeight = frameOpencvDnn.shape[0]
#     frameWidth = frameOpencvDnn.shape[1]
#     blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

#     net.setInput(blob)
#     detections = net.forward()
#     faceBoxes = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > conf_threshold:
#             x1 = int(detections[0, 0, i, 3] * frameWidth)
#             y1 = int(detections[0, 0, i, 4] * frameHeight)
#             x2 = int(detections[0, 0, i, 5] * frameWidth)
#             y2 = int(detections[0, 0, i, 6] * frameHeight)
#             faceBoxes.append([x1, y1, x2, y2])
#             cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
#     return frameOpencvDnn, faceBoxes


# parser = argparse.ArgumentParser()
# parser.add_argument('--image')
# args = parser.parse_args()

# # File paths for models
# faceProto = r"G:\Gender-and-Age-Detection-master\opencv_face_detector.pbtxt"
# faceModel = r"G:\Gender-and-Age-Detection-master\opencv_face_detector_uint8.pb"
# ageProto = r"G:\Gender-and-Age-Detection-master\age_deploy.prototxt"
# ageModel = r"G:\Gender-and-Age-Detection-master\age_net.caffemodel"
# genderProto = r"G:\Gender-and-Age-Detection-master\gender_deploy.prototxt"
# genderModel = r"G:\Gender-and-Age-Detection-master\gender_net.caffemodel"

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']

# # Load models
# faceNet = cv2.dnn.readNet(faceModel, faceProto)
# ageNet = cv2.dnn.readNet(ageModel, ageProto)
# genderNet = cv2.dnn.readNet(genderModel, genderProto)

# # Start video capture (0 for default camera)
# video = cv2.VideoCapture(args.image if args.image else 0)

# # Set video resolution (optional)
# video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# padding = 20
# frameCount = 0
# processEveryNthFrame = 3  # Adjust N to control how often frames are processed

# while True:
#     hasFrame, frame = video.read()
#     if not hasFrame:
#         break

#     frameCount += 1

#     # Skip frames to improve performance
#     if frameCount % processEveryNthFrame != 0:
#         continue

#     # Detect faces in the frame
#     resultImg, faceBoxes = highlightFace(faceNet, frame)

#     if not faceBoxes:
#         print("No face detected")
#         totalPeople = 0
#         totalMen = 0
#         totalWomen = 0
#     else:
#         totalPeople = len(faceBoxes)  # Total people detected
#         totalMen = 0
#         totalWomen = 0

#         # Iterate through each detected face
#         for faceBox in faceBoxes:
#             face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
#                          max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

#             blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

#             # Predict gender
#             genderNet.setInput(blob)
#             genderPreds = genderNet.forward()
#             gender = genderList[genderPreds[0].argmax()]

#             # Predict age
#             ageNet.setInput(blob)
#             agePreds = ageNet.forward()
#             age = ageList[agePreds[0].argmax()]

#             # Count men and women
#             if gender == "Male":
#                 totalMen += 1
#             else:
#                 totalWomen += 1

#             # Display gender and age on the frame
#             cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                         (0, 255, 255), 2, cv2.LINE_AA)

#     # Display total counts on the frame
#     cv2.putText(resultImg, f'Total: {totalPeople}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
#     cv2.putText(resultImg, f'Men: {totalMen}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(resultImg, f'Women: {totalWomen}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

#     # Show the frame with detections
#     cv2.imshow("Real-Time Age and Gender Detection", resultImg)

#     # Exit the loop if 'Esc' is pressed
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# # Release the video capture and close windows
# video.release()
# cv2.destroyAllWindows()
import cv2
import math
import argparse
import os

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Argument parser for optional image or video input
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image or video file')
args = parser.parse_args()

# Model file paths
faceProto = r"G:\Gender-and-Age-Detection-master\opencv_face_detector.pbtxt"
faceModel = r"G:\Gender-and-Age-Detection-master\opencv_face_detector_uint8.pb"
ageProto = r"G:\Gender-and-Age-Detection-master\age_deploy.prototxt"
ageModel = r"G:\Gender-and-Age-Detection-master\age_net.caffemodel"
genderProto = r"G:\Gender-and-Age-Detection-master\gender_deploy.prototxt"
genderModel = r"G:\Gender-and-Age-Detection-master\gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load the models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Check if the input is an image or video file
if args.image:
    file_extension = os.path.splitext(args.image)[1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png']:
        # Image input
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image from {args.image}")
            exit()
    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video input
        video = cv2.VideoCapture(args.image)
        if not video.isOpened():
            print(f"Could not read video from {args.image}")
            exit()
    else:
        print(f"Unsupported file type: {file_extension}")
        exit()
else:
    # Default to webcam if no image or video is provided
    video = cv2.VideoCapture(1)

padding = 20
frameCount = 0
processEveryNthFrame = 3  # Adjust N to control how often frames are processed

while True:
    if args.image and file_extension in ['.jpg', '.jpeg', '.png']:
        hasFrame = True  # Only need to process the image once
    else:
        hasFrame, frame = video.read()  # Read the next frame from the video or webcam

    if not hasFrame:
        print("No frame captured, exiting...")
        break

    frameCount += 1

    # Skip frames to improve performance for video
    if frameCount % processEveryNthFrame != 0:
        continue

    # Detect faces in the frame
    resultImg, faceBoxes = highlightFace(faceNet, frame)

    if not faceBoxes:
        print("No face detected")
        totalPeople = 0
        totalMen = 0
        totalWomen = 0
    else:
        totalPeople = len(faceBoxes)  # Total people detected
        totalMen = 0
        totalWomen = 0

        # Iterate through each detected face
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict gender
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Predict age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            # Count men and women
            if gender == "Male":
                totalMen += 1
            else:
                totalWomen += 1

            # Display gender and age on the frame
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

    # Display total counts on the frame
    cv2.putText(resultImg, f'Total: {totalPeople}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(resultImg, f'Men: {totalMen}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(resultImg, f'Women: {totalWomen}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

    # Show the frame with detections
    cv2.imshow("Age and Gender Detection", resultImg)

    # Exit the loop if 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close windows
if not args.image or file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
    video.release()
cv2.destroyAllWindows()
