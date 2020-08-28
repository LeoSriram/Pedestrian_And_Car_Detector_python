import cv2

#Our Image.
img_file = 'Car Image.jpg'
video = cv2.VideoCapture('Dashcam Pedestrians.mp4')
video = cv2.VideoCapture('Tesla Dashcam Compilation.mp4')

#Car and Pedestrian classifiers.
car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#Create car classifier.
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

#Run until the video stops.
while True:
    #Read the current frame.
    (read_successful, frame) = video.read()

    #Safe coding.
    if read_successful:
        #Must convert to grayscale.
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    #Detect Cars and Pedestrians.
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw rectangles around the cars.
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    #Draw rectangles around the pedestrians.
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    #Show the image with the faces spotted.
    cv2.imshow('Self-driven Car', frame)

    #Do not auto-close.
    key = cv2.waitKey(1)

    #Stop if Q key is pressed.
    if key==81 or key==113:
        break

#Release the VideoCapture object.
video.release()