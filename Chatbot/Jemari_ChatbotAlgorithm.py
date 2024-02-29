# TfidfVectorizer: mengubah texts menjadi feature vectors dengan TF-IDF Method
from sklearn.feature_extraction.text import TfidfVectorizer
# svm (Support Vector Machine): classification algorithm
from sklearn import svm
# train_test_split: membagi data menjadi training & test sets
from sklearn.model_selection import train_test_split

# data: chat data berisi kalimat yang menggambarkan perasaan user
data = [
    "Saya merasa sedih", 
    "Saya sangat bahagia hari ini", 
    "Saya merasa tidak baik", 
    "Saya merasa gembira", 
    "Saya merasa senang", 
    "Saya merasa depresi",
    "Saya merasa lelah",
    "Saya merasa bersemangat",
    "Saya merasa takut",
    "Saya merasa cemas",
    "Saya merasa marah",
    "Saya merasa tenang",
    "Saya merasa bosan",
    "Saya merasa puas",
    "Saya merasa kecewa",
    "Saya merasa terganggu",
    "Saya merasa terkejut",
    "Saya merasa terinspirasi",
    "Saya merasa penasaran",
    "Saya merasa optimis",
    "Saya merasa pesimis",
    "Saya merasa tidak enak",
    "Saya merasa kagum",
    "Saya merasa malu",
    "Saya merasa bingung",
    "Saya merasa terpukul",
    "Saya merasa terluka",
    "Saya merasa menyesal",
    "Saya merasa terabaikan",
    "Saya merasa dihargai",
    "Saya merasa diterima",
    "Saya merasa dicintai",
    "Saya merasa dikhianati",
    "Saya merasa dihina",
    "Saya merasa dipermalukan",
    "Saya merasa diperlakukan tidak adil",
    "Saya merasa diperlakukan dengan baik",
    "Saya merasa diperlakukan dengan buruk",
    "Saya merasa diperlakukan dengan hormat",
    "Saya merasa diperlakukan dengan tidak hormat"
]
# labels: classification labels (perasaan 'positif'/'negatif') dari setiap chat data
labels = [
    "negatif", 
    "positif", 
    "negatif", 
    "positif", 
    "positif", 
    "negatif",
    "negatif",
    "positif",
    "negatif",
    "negatif",
    "negatif",
    "positif",
    "negatif",
    "positif",
    "negatif",
    "negatif",
    "positif",
    "positif",
    "positif",
    "positif",
    "negatif",
    "negatif",
    "positif",
    "negatif",
    "negatif",
    "negatif",
    "negatif",
    "negatif",
    "negatif",
    "positif",
    "positif",
    "positif",
    "negatif",
    "negatif",
    "negatif",
    "negatif",
    "positif",
    "negatif",
    "positif",
    "negatif"
]

# train_test_split membagi data & labels menjadi training & test sets
# test_size=0.2: 20% data digunakan sebagai test sets & 80% data digunakan sebagai training sets
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# TfidVectorizer(): mengubah texts data_train menjadi feature vectors dengan TF-IDF method
vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(data_train)

# svm.SVC(kernel='linear'): membuat SVM model dengan linear kernel
classifier = svm.SVC(kernel='linear')
# classifier.fit(vectors_train, labels_train): SVM model training menggunakan feature vectors & labels dari training sets
classifier.fit(vectors_train, labels_train)

# vectorizer.transform("new user chat"): mengubah new user chat menjadi feature vector
new_data = vectorizer.transform(["Akhir2 ini saya jadi gampang ngerasa sedih"])
# classifier.predict(new_data): memprediksi label dari new user chat menggunakan trained SVM model
prediction = classifier.predict(new_data)
# print(prediction): print hasil prediksi new user chat (perasaan 'positif'/'negatif')
print(prediction)