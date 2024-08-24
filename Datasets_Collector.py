import cv2
import os

name = path = None
if not os.path.isdir('Dataset'):
    os.mkdir('Dataset')

def camera_details():
    print("Capturing images from Camera")
    print("Initializing face capture. Look the camera and wait")

if cv2.VideoCapture(1) == True:
    print("External Camera found")
    camera_details()
    select_cam = 1
else:
    print("No External Camera found")
    camera_details()
    select_cam = 0

cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture(1,  cv2.CAP_DSHOW)
ret, img= cam.read()

def img(name: str, limit):
    Counter = 1   # Counter for naming the images
    ret, img= cam.read()
    a = 0
    while ret:
        Counter += 1
        ret, img= cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to Gray
        img_ = cv2.resize(gray, (400, 400)) # Resize the image
        img_name = (f"file_{a}.jpg") # Create the name of the image
        fullPath = os.path.join(path, img_name) # Create the path of the image
        # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255), 2)    # Put the name of the person    
           
        if Counter >= 10:
            print(f"Saved to {fullPath}") # Print the path of the image
            cv2.imwrite(filename=fullPath, img=img_) # Save the image
            
            Counter = 0
            a += 1
        
        window_name = "Face Detection Module" # Create the window name
        cv2.imshow(window_name, img) # Show the image  # correct
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) 
        print(cv2) 

        key = cv2.waitKey(1) & 0xFF # Wait for 2 seconds and get the key
        if key == ord('b'):
            break

        if a == limit: # If the number of images is 25 then break
            print("Completed!!!!!!!!!!!!!!!")
            break
    print("Next Person Please...!!!!!!!!!!!!!!!")
    return


while True:
    name = str(input("Enter Your Full Name: ")).title()
    path = os.path.join('Dataset', name)
    if not os.path.isdir(path):
        os.mkdir(path)
        # with open(f'{path}\\{name}.txt', "w") as f:
        #     f.write(f'Name: {name}\n College_id: {College_id}@iic.edu.np\n Contace_number: {Contact_number}')
    else:
        print("Name already exists")
    
    img(name, 20)



cam.release() # Release the camera
cv2.destroyAllWindows() # Destroy all the windows
