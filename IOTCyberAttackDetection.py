from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
from sklearn.ensemble import RandomForestClassifier

import pandas as pd  
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


main = tkinter.Tk()
main.title("Cyber Risk in Internet of Things World")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test, pca, classifier
global dataset
global filename
global X, Y, label_encoder, imputer, imputer1
sc = StandardScaler()

labels = ['anomalous(DoSattack)', 'anomalous(dataProbing)', 'anomalous(malitiousControl)', 'anomalous(malitiousOperation)',
          'anomalous(scan)', 'anomalous(spying)', 'anomalous(wrongSetUp)', 'normal']

    
def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head()))

def preprocessDataset():
    global X, Y, label_encoder, sc, dataset, imputer, imputer1
    label_encoder = []
    text.delete('1.0', END)
    X = dataset.iloc[:,:-2].values
    Y = dataset.iloc[:,12].values
    label_encoder = []
    for i in range(0,11):
        le = LabelEncoder()
        X[:,i] = le.fit_transform(X[:,i].astype(str))
        label_encoder.append(le)
    imputer = SimpleImputer(missing_values=np.nan,strategy='constant')
    imputer = imputer.fit(X[:,[8]])
    X[:,[8]] = imputer.transform(X[:,[8]])
    imputer1 = SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer1 = imputer1.fit(X[:,[10]])
    X[:,[10]] = imputer1.transform(X[:,[10]])
    X = np.array(X,dtype=float)
    X = sc.fit_transform(X)
    Y_le = LabelEncoder()
    Y = Y_le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
    text.insert(END,"Total features found in dataset before apply PCA features selection: "+str(X.shape[1])+"\n\n")
    text.insert(END,"Different Security & Privacy attack found in dataset\n\n")
    text.insert(END,str(labels))
    
def featureSelection():
    global X_train, X_test, y_train, y_test
    global X, Y, pca
    text.delete('1.0', END)
    pca = PCA(n_components = 10)
    X = pca.fit_transform(X)
    text.insert(END,"Total features found in dataset after apply PCA features selection: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"Dataset training and testing split details\n")
    text.insert(END,"Dataset size: "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training: "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset used for testing: "+str(X_test.shape[0])+"\n")

def runKmeans():
    global X, Y
    global X_train, X_test, y_train, y_test
    kmeans = KMeans(n_clusters=2,n_init=1, random_state=1)
    kmeans.fit(X_train, y_train)
    centers = kmeans.cluster_centers_
    predict = kmeans.predict(X_test)
    centers = kmeans.cluster_centers_
    text.insert(END,"\nCluster Centers : "+str(centers)+"\n\n")
    text.insert(END,"Cluster Labels : "+str(kmeans.labels_)+"\n\n")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predict, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()


def runRandomForest():
    global X_train, X_test, y_train, y_test, classifier
    global X, Y, pca
    text.delete('1.0', END)

    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    text.insert(END,"Random Forest Test Data Prediction Accuracy: "+str(accuracy)+"\n\n")

    conf_matrix = confusion_matrix(y_test, y_pred) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title("Random Foresr Classification Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def predict():
    global pca, sc, classifier, label_encoder, imputer, imputer1
    text.delete('1.0', END)

    try:
        testFile = filedialog.askopenfilename(initialdir="Dataset")
        if testFile == "":
            text.insert(END, "No test file selected.\n")
            return

        dataset = pd.read_csv(testFile)
        dataset.fillna(0, inplace=True)

        test = dataset.iloc[:, :-2].values

        # Ensure same number of features as training
        if test.shape[1] < sc.n_features_in_:
            diff = sc.n_features_in_ - test.shape[1]
            test = np.hstack((test, np.zeros((test.shape[0], diff))))

        # Label Encoding
        for i in range(min(11, test.shape[1])):
            col = test[:, i].astype(str)
            known = label_encoder[i].classes_
            col = np.where(np.isin(col, known), col, known[0])
            test[:, i] = label_encoder[i].transform(col)

        # Imputation
        if test.shape[1] > 8:
            test[:, [8]] = imputer.transform(test[:, [8]])
        if test.shape[1] > 10:
            test[:, [10]] = imputer1.transform(test[:, [10]])

        # Convert to numeric
        test = np.array(test, dtype=float)

        # 🔥 SAVE NUMERIC VERSION FOR DISPLAY
        display = test.copy()

        # Scaling + PCA
        test = sc.transform(test)
        test = pca.transform(test)

        # Prediction
        predictions = classifier.predict(test)

        # Display results (numeric like screenshot)
        for i in range(len(display)):
            text.insert(
                END,
                "Test IOT Traffic: " + str(display[i].astype(int)) +
                " =====> Predicted As: " + labels[predictions[i]] + "\n\n"
            )

        text.update_idletasks()

    except Exception as e:
        text.insert(END, "Prediction Error:\n" + str(e))



def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Cyber Risk in Internet of Things World')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload IOT Cyber Attack Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

hybridMLButton = Button(main, text="Run Features Selection Algorithm", command=featureSelection)
hybridMLButton.place(x=50,y=200)
hybridMLButton.config(font=font1)

snButton = Button(main, text="Run KMeans Clustering", command=runKmeans)
snButton.place(x=50,y=250)
snButton.config(font=font1)

snButton = Button(main, text="Run Random Forest Classification Algorithm", command=runRandomForest)
snButton.place(x=50,y=300)
snButton.config(font=font1)

graphButton = Button(main, text="Predict Attack from Test Data", command=predict)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()