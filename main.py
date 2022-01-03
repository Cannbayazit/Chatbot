#-Verilerin etiketi vardır.
#-Verileri bir gruba dahil etmek için bir kural oluşturulmasını bekler.
#-Veri setini eğitim ve test olarak ayırmak gereklidir.

#gerekli kütüphaneler import edildi
from matplotlib import pyplot as plt
import json
import string
import random
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from snowballstemmer import TurkishStemmer
stemmer=TurkishStemmer()

#json dosyası okundu
data_file= open('intentstr.json').read()
veri=json.loads(data_file)


pkelimeler=[]    #bow modeli icin patternlerin kelimeleri
tkelimeler=[]  #bow modeli icin tag ler in kelimeleri
data_X=[]   #her pattern i depolamak icin
data_y=[]   #data_x içindeki patterne karşılık gelen tagler

for intent in veri["intents"]:
  for pattern in intent["patterns"]:
    tokens=nltk.word_tokenize(pattern)  #her pattern tokenize edildi
    pkelimeler.extend(tokens)                #daha sonra kelimeler dizisine eklendi
    data_X.append(pattern)              #patternler data_X e eklendi
    data_y.append(intent["tag"]),       #data_y ye pattern ile ilişkilendirilen tag lar eklendi

  if intent["tag"] not in tkelimeler:  # tag tkelimelerin içinde değilse ekle
    tkelimeler.append(intent["tag"])


pkelimeler=[stemmer.stemWord(word.lower()) for word in pkelimeler if word not in string.punctuation]  #veri setimizdeki pkelimeleri stemmer edip sonra küçük harfe dönüştürdük
pkelimeler=sorted(set(pkelimeler))    #pkelimeler aflabetik olarak sıralandı
tkelimeler=sorted(set(tkelimeler))    #tkelimeler içindeki taglar alfabetik olarak sıralandı


##
#bow model ile textler rakamlara dönüştürüldü
#pkelimeler ve tkelimeler dizileri pattern ve taglar varken
#kelimelerin listeleri ile aynı uzunlukta bir dizi oluşturduk
#bu dizinin değeri data_x deki pattern/tag ile uyuşuyorsa 1 uyuşmuyorsa 0
#böylece veriler sayılara dönüşüp iki dizide deoplanır train_X = özellikleri temsil eder ve train_Y = hedef değişkenleri temsil eder olarak

training=[]
out_empty=[0]*len(tkelimeler)

for i,doc in enumerate(data_X):
  bow=[]
  text=stemmer.stemWord(doc.lower())

  for word in pkelimeler:
    bow.append(1) if word in text else bow.append(0)

    output_row=list(out_empty)
    output_row[tkelimeler.index(data_y[i])]=1

    training.append([bow,output_row])

random.shuffle(training)
training=np.array(training,dtype=object)
train_X=np.array(list(training[:,0]))   #patterns
train_Y=np.array(list(training[:,1]))   #responses


                      #input_shape girdi eğitim verisinin yapısını girdi sayısı olarak alabiliriz

model=Sequential()
model.add(Dense(128,input_shape=(len(train_X[0]),), activation="relu")) #coklu sınıflandırma oldugu icn ReLU aktivasyon fonksiyonu hız bakımından avantajlıdır ve genellikle çıkış değil ara katmanlarda kullanılır.
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation= "softmax"))  #çok sınıflı sınıflandırma oldugu icin
                                                            #Çıkış katmanlarında genellikle Softmax aktivasyon fonksyonu kullanılır.
adam=tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

#optimizer:öğrenme oranını kontrol eder. optimizer olarak adam kullanıyoruz.
#adam genellikle birçok durumda kullanılmak için iyi bir optimizasyon algoritmasıdır
#adam algoritması , eğitim boyunca öğrenme oranını ayarlar.
#kayıp fonksyonu için categoricalcrossentropy kullanacağız bu sınıflandırma problemleri için en yaygın kulanılan fonksyondur
#eğitim sırasında modelin nasıl bir performans gösterdiğini görmek için ise accuracy metriği kullanıldı
model.compile(loss='categorical_crossentropy', #compile fonksiyonunu kullanarak modelimizi nasıl eğiteceğimizi belirttik
              optimizer=adam,                   #loss tahmindeki hatayı hesaplayan fonksyon
              metrics=["accuracy"])

print(model.summary())#param = 128*inputshape+128

#model eğitilirken modeli değerlendirmek için validation split modeli kullanılır
history=model.fit(x=train_X, y=train_Y, epochs=100,validation_split=0.2, verbose=1,batch_size=8) #modele datayı verip eğittik

#batch size : tek seferde tüm verilerin eğitilmesi yerine veri setinin alt kümeleri üzerinde eğitim yapmak için kullanılan bir tekniktir sistemdeki hafıza sınırlılığından dolayı eğitim batchlerde yapılır
#epochs tüm veri setini ileri ve geriye doğru tek bir geçiş adımıdr


plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.show()


#programımız yalnızca sayıları aldığı için kullanıcı girdilerinide işlememiz gerekiyor
#kullanıcının girdisini dizilere dönüştürecek ve bununla ilgili etiketi tahmin edecek birkaç fonksyon oluşturacağız
#kodumuz daha sonra makinenin bu etikete karşılık gelen yanıtlardan birini seçecek ve çıktı olarak gönderecektir

#bu fonksyon girdi olarak alınan veriyi tokenize eder ve turkishStemmer kullanılarak kök forma dönüştürülür.çıktı temel olarak kök biçimindeki kelimelerin bir listesidir
def clean_text(text):
  tokens=nltk.word_tokenize(text)
  tokens=[stemmer.stemWord(word) for word in tokens]
  return tokens

#bu fonksyon yukardaki fonksyonu çağırır ve bow modelini kullanarak bir diziye dönüştürür ve sonra o diziyi döndürür
def bag_of_words(text, vocab):
  tokens=clean_text(text)
  bow=[0]*len(vocab)
  for w in tokens:
    for i,word in enumerate(vocab):
      if word ==w:
        bow[i]=1
  return np.array(bow)

#bu fonksyon metin ve kelime bilgisi etiketlerini girdi olarak alır ve en yüksek olasığa karşılık gelen bir etiket içeren liste döndürür
def pred_class(text,vocab,labels):
  bow= bag_of_words(text,vocab)
  result=model.predict(np.array([bow]))[0]
  thresh=0.5
  y_pred=[[indx,res]for indx, res in enumerate(result) if res>thresh]
  y_pred.sort(key=lambda x:x[1], reverse=True)
  return_list=[]
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

#bu fonksiyon ise önceki fonksiyon tarafından döndürülen tagı alır ve bunu intentstr.json veri kümemizde aynı etikete karşılık gelen bir yanıtı rastgele seçmek için kullanır.
def get_response(intents_list,intents_json):
  if len(intents_list)== 0:
    result="Uzgunum seni anlayamadim"
  else:
    tag=intents_list[0]
    list_of_intents=intents_json["intents"]
    for i in list_of_intents:
      if i["tag"]==tag:
        result=random.choice(i["responses"])
        break
  return result

print("Botu kapatmak icin 0 a bas")
while True:
  message=input("")
  if message =="0":
    break
  intents=pred_class(message,pkelimeler,tkelimeler)
  result=get_response(intents,veri)
  print(result)


