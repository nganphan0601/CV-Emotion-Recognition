import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from mtcnn import MTCNN
import sys

# Define the emotions to use and their corresponding folders
data_folder = '/Users/gurmanhunjan/UFV/Summer 2024/COMP 381/Emotion Detection/emotiondata' #update path
selected_emotions = ['happy', 'sad', 'angry', 'surprise']

# Initialize MTCNN detector
detector = MTCNN()

def loadimagesmtcnn(folder, label):
    images = []
    labels = []
    abs_folder_path = os.path.abspath(folder)
    if not os.path.exists(abs_folder_path):
        return images, labels
    if not os.listdir(abs_folder_path):
        return images, labels

    for filename in os.listdir(abs_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(abs_folder_path, filename)
            image = cv2.imread(file_path)
            if image is not None:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_image)
                for face in faces:
                    x, y, width, height = face['box']
                    face_region = rgb_image[y:y + height, x:x + width]
                    face_region = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
                    resized_face = cv2.resize(face_region, (48, 48))
                    normalized_face = resized_face / 255.0
                    images.append(normalized_face)
                    labels.append(label)
    return images, labels
# running directly to ensure no fix loop issue
def main():
    data = []
    labels = []
    for i, emotion in enumerate(selected_emotions):
        train_folder = os.path.join(data_folder, 'train', emotion)
        test_folder = os.path.join(data_folder, 'test', emotion)

        train_images, train_labels = loadimagesmtcnn(train_folder, i)
        data.extend(train_images)
        labels.extend(train_labels)

        test_images, test_labels = loadimagesmtcnn(test_folder, i)
        data.extend(test_images)
        labels.extend(test_labels)

    if len(data) == 0:
        sys.exit(1)
    data = np.array(data)
    labels = np.array(labels)
    
    try:
        data_flattened = data.reshape(data.shape[0], -1)
    except ValueError as e:
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(data_flattened, labels, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    save_path = os.path.join(os.getcwd(), 'knn_emotion_model.pkl')
    joblib.dump(knn, save_path)

if __name__ == "__main__":
    main()
