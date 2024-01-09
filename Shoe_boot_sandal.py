from sklearn.model_selection import KFold
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Fotoğrafların bulunduğu klasörlerin yolları
bot_yol= 'C:\\Users\\alime\\Desktop\\Shoe vs Sandal vs Boot Dataset\\Boot'
sandalet_yol = 'C:\\Users\\alime\\Desktop\\Shoe vs Sandal vs Boot Dataset\\Sandal'
ayakkabı_yol = 'C:\\Users\\alime\\Desktop\\Shoe vs Sandal vs Boot Dataset\\Shoe'

# Resimleri yüklemek için bir fonksiyon
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(cv2.resize(img, (100, 100)))  # Tüm resimleri 100x100 boyutuna getir
            labels.append(label)
    return images, labels

# Tüm resimleri yükle
boot_img, boot_lbl = load_images_from_folder(bot_yol, 0)
sandal_img, sandal_lbl = load_images_from_folder(sandalet_yol, 1)
ayakkabı_img, ayakkabı_lbl = load_images_from_folder(ayakkabı_yol, 2)

# Tüm resimleri birleştir
all_images = np.array(boot_img + sandal_img + ayakkabı_img)
all_labels = np.array(boot_lbl + sandal_lbl + ayakkabı_lbl)

# Veri ve etiketleri birleştirme
data = np.array(all_images)
labels = np.array(all_labels)

# 10 katlı çapraz doğrulama oluşturma
kf = KFold(n_splits=10, shuffle=True, random_state=35)

accuracies = []  # Doğruluk değerlerini saklamak için bir liste

for train_index, test_index in kf.split(data):
    X_train, X_val = data[train_index], data[test_index]
    y_train, y_val = labels[train_index], labels[test_index]

    # Makine öğrenmesi modelini oluştur ve eğit
    model = SVC(kernel='linear')
    model.fit(X_train.reshape(len(X_train), -1), y_train)

    # Doğruluk değerini hesapla ve listeye ekle
    y_pred = model.predict(X_val.reshape(len(X_val), -1))
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)

# Ortalama doğruluk değerini hesapla
mean_accuracy = np.mean(accuracies)
print("1. Test Doğruluğu: ",accuracies[0])
print("2. Test Doğruluğu: ",accuracies[1])
print("3. Test Doğruluğu: ",accuracies[2])
print("4. Test Doğruluğu: ",accuracies[3])
print("5. Test Doğruluğu: ",accuracies[4])
print("6. Test Doğruluğu: ",accuracies[5])
print("7. Test Doğruluğu: ",accuracies[6])
print("8. Test Doğruluğu: ",accuracies[7])
print("9. Test Doğruluğu: ",accuracies[8])
print("10. Test Doğruluğu: ",accuracies[9])

print("Ortalama Test Doğruluğu (10 Katlı Çapraz Doğrulama):", mean_accuracy)

accuraciesRF = []  # Doğruluk değerlerini saklamak için bir liste

for train_index, test_index in kf.split(data):
    X_train, X_val = data[train_index], data[test_index]
    y_train, y_val = labels[train_index], labels[test_index]

    # Makine öğrenmesi modelini oluştur ve eğit
    model = RandomForestClassifier()  # RandomForestClassifier kullanılacak
    model.fit(X_train.reshape(len(X_train), -1), y_train)

    # Doğruluk değerini hesapla ve listeye ekle
    y_pred = model.predict(X_val.reshape(len(X_val), -1))
    accuracy = accuracy_score(y_val, y_pred)
    accuraciesRF.append(accuracy)

# Ortalama doğruluk değerini hesapla
mean_accuracy = np.mean(accuraciesRF)
print("1. Test Doğruluğu: ",accuraciesRF[0])
print("2. Test Doğruluğu: ",accuraciesRF[1])
print("3. Test Doğruluğu: ",accuraciesRF[2])
print("4. Test Doğruluğu: ",accuraciesRF[3])
print("5. Test Doğruluğu: ",accuraciesRF[4])
print("6. Test Doğruluğu: ",accuraciesRF[5])
print("7. Test Doğruluğu: ",accuraciesRF[6])
print("8. Test Doğruluğu: ",accuraciesRF[7])
print("9. Test Doğruluğu: ",accuraciesRF[8])
print("10. Test Doğruluğu: ",accuraciesRF[9])
print("Ortalama Test Doğruluğu (Random Forest - 10 Katlı Çapraz Doğrulama):", mean_accuracy)


accuraciesGBN = []  # Doğruluk değerlerini saklamak için bir liste

for train_index, test_index in kf.split(data):
    X_train, X_val = data[train_index], data[test_index]
    y_train, y_val = labels[train_index], labels[test_index]

    # Naive Bayes modelini oluştur ve eğit (GaussianNB kullanılacak)
    model = GaussianNB()
    model.fit(X_train.reshape(len(X_train), -1), y_train)

    # Doğruluk değerini hesapla ve listeye ekle
    y_pred = model.predict(X_val.reshape(len(X_val), -1))
    accuracy = accuracy_score(y_val, y_pred)
    accuraciesGBN.append(accuracy)

# Ortalama doğruluk değerini hesapla
mean_accuracy = np.mean(accuraciesGBN)
print("1. Test Doğruluğu: ",accuraciesGBN[0])
print("2. Test Doğruluğu: ",accuraciesGBN[1])
print("3. Test Doğruluğu: ",accuraciesGBN[2])
print("4. Test Doğruluğu: ",accuraciesGBN[3])
print("5. Test Doğruluğu: ",accuraciesGBN[4])
print("6. Test Doğruluğu: ",accuraciesGBN[5])
print("7. Test Doğruluğu: ",accuraciesGBN[6])
print("8. Test Doğruluğu: ",accuraciesGBN[7])
print("9. Test Doğruluğu: ",accuraciesGBN[8])
print("10. Test Doğruluğu: ",accuraciesGBN[9])
print("Ortalama Test Doğruluğu (Naive Bayes - 10 Katlı Çapraz Doğrulama):", mean_accuracy)