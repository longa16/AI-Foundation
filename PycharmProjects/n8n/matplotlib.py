from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

textes = [
    "J'adore ce film",  # positif
    "C'était fantastique",  # positif
    "Très mauvais",  # négatif
    "Je déteste ce produit",  # négatif
    "Incroyable et bien joué",  # positif
    "C’est nul et ennuyeux"  # négatif
]

labels = ['positif', 'positif', 'négatif', 'négatif', 'positif', 'négatif']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textes)

model = MultinomialNB()
model.fit(X, labels)

essai = input("Entrez le texte d'essai ici : ")
X_test = vectorizer.transform([essai])
prediction = model.predict(X_test)

print("Prediction : ", prediction[0])
