import re
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
import requests

maxlen = 100
#def my_tokenizer(text):
    # create a space between special characters 
#    text=re.sub("(\\W)"," \\1 ",text)

    # split based on whitespace
#    return re.split("\\s+",text)
    
#def create_feature_matrix(message_list, vectorizer):
#        doc_term_matrix = vectorizer.fit_transform(message_list)
#        return DataFrame(doc_term_matrix.toarray())

def file_to_list(file_name):
    if not os.path.isfile(file_name):
        print("[-BF] Could not find file path exiting...")
	
    headers = ["payload", "text","state"]
    df = pd.read_csv(file_name)
    return(df)


file_data = file_to_list("learn_data.csv")
html_data = file_data["html"].values
labels = file_data["state"].values

train_data, test_data, labels_train, labels_test = train_test_split(html_data, labels,  test_size=0.25)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data)
X_train = tokenizer.texts_to_sequences(train_data)
X_test = tokenizer.texts_to_sequences(test_data)
tokenizer.word_index["<PAD>"] = 0
vocab_size = len(tokenizer.word_index) + 1
X_train = pad_sequences(X_train, value=tokenizer.word_index["<PAD>"], padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, value=tokenizer.word_index["<PAD>"], padding='post', maxlen=maxlen)
embedding_dim = 50
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',              
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, labels_train,
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test, labels_test),
                    batch_size=34)
loss, accuracy = model.evaluate(X_train, labels_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))

predicted = model.predict(X_test)

wrong = 0
success = 0
for x in range(len(predicted)):
    if predicted[x] !=  labels_test[x]:
        wrong += 1
        print("[-] Predicted: ", predicted[x], "Data: ", X_test[x], "Actual: ", labels_test[x])
    elif predicted[x] ==  labels_test[x]:
        success += 1

xssSuccess = 0
for i in file_data["state"]:
    if i == 1:
        xssSuccess +=1


print("[-] Total XSS Successes existing file: ", xssSuccess)

print(f"[-] I got {wrong} wrong guesses out of total {xssSuccess} which refers to a real accuracy of: {1-(wrong/xssSuccess)}")
print(f"[-] I got {success} right guesses but is irrelevant i think")