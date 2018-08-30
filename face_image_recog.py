import face_recognition
import argparse
import pickle 
import cv2
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required = True, help = "path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-d", "--detection-method", type = str, default = "hog", help = "face detection model to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())
print("[INFO] loading encodings...")
#load known faces and embeddings
data = pickle.loads(open(args["encodings"], "rb").read())
#load input image and covert from BGR to RGB

image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#detect (x,y) coordinates for boundingboxes in each image, then get facial embeddings
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model = args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

#init list of names for face detected
names = []
#init list of distance measurements(norm of difference/norm of true) for face detected (compare encoding distances)
distNames = []
distAccuracies = []
#loope over facial embeddings
for encoding in encodings:
    #attempt to match each face in input image to our known encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance = 0.6)
    name = "Unknown"
    #we compute boolean value based on whether euclidean distance between input face and all faces is below/above threshold
    #check to see for match
    if True in matches:
        #find all indices of matched faces and init dictionary to track matches
        matchedIdxs = [i for (i,b) in enumerate(matches) if b]
        counts = {}
        
            #loop over matched indexes and maintain count for each recognized face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
            distNames+=[name]
            distAccuracies+=[(1- np.linalg.norm(encoding - data["encodings"][i])/(np.linalg.norm(data["encodings"][i])))]
            #determine recognized face with largest number of votes, in case of tie, first entry
    name = max(counts, key = counts.get)
    
    #update list of names
    names.append(name)
"""
#print out true accuracies
accuracyPerc = {}
for x in counts.keys():
	runningVals = []
	for i in range(len(distNames)):
		if x == distNames[i]:
			runningVals.append(distAccuracies[i])
	accuracyPerc[x] = sum(runningVals)/len(runningVals)
	continue
for x in accuracyPerc.keys():
	print("Being " + x + " : " + str(accuracyPerc[x]*100) + "%" ) 
"""

#return max index for person


names = [distNames[max(range(len(distAccuracies)), key = lambda x: distAccuracies[x])]]    
print("Being " + names[0] + " : " + str(distAccuracies[max(range(len(distAccuracies)), key = lambda x: distAccuracies[x])]*100)+ "%")
for ((top, right, bottom, left), name) in zip(boxes, names):
    #draw predicted face name on image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)



