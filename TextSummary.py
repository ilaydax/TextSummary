import tkinter as tk
from tkinter import filedialog
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
nltk.download('averaged_perceptron_tagger')


def select_file():
    file_path = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            sentences = sent_tokenize(content)
            global preprocessed_sentences
            preprocessed_sentences = preprocess_sentences(sentences)
            #create_graph_button.configure(state='normal')
            global graph
            global threshold
            graph = create_similarity_graph(preprocessed_sentences, threshold)

def preprocess_sentences(sentences):
    preprocessed_sentences = []
    stop_words = set(load_custom_stopwords()) # Kendi oluşturduğunuz stop word listesi
    for sentence in sentences:
        # Tokenization
        tokens = word_tokenize(sentence, language='turkish')
        # Stop-word Elimination and Punctuation
        filtered_tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        preprocessed_sentences.append(' '.join(filtered_tokens))
    return preprocessed_sentences

def load_custom_stopwords():
    # Kendi stop word listesinizi burada doldurun
    custom_stopwords = [
        
        ]
    return custom_stopwords

#özel isim bulma ///////////////////////////////////////////////////
def check_proper_nouns(sentence5):
    tagged_words = nltk.pos_tag(nltk.word_tokenize(sentence5))
    global proper_nouns
    global yeni_ratios
    proper_nouns = [word for word, pos in tagged_words if pos == 'NNP' or pos == 'NNPS']
    return proper_nouns
def calculate_proper_noun_ratio(sentences):
    global ratios
    ratios = []
    for sentence5 in sentences:
        words = nltk.word_tokenize(sentence5)
        proper_nouns = check_proper_nouns(sentence5)
        proper_noun_count = len(proper_nouns)
        word_count = len(words)
        if proper_noun_count>1:
           ratio = proper_noun_count / word_count
        else:
            ratio=0
        ratios.append(ratio)

        sayi = 0.4
        yeni_ratios = []

        for i in range(len(ratios)):
            carpim = ratios[i] * sayi
            yeni_ratios.append(carpim)
            #print(yeni_ratios)  
    return yeni_ratios

#numerik veri hesabı //////////////////////////////////////////////////////////////////////////////////////////
def calculate_numerical_ratio_in_sentences(preprocessed_sentences):
    global ratio6
    total_numerical_count = 0
    total_word_count = 0
    global ratio6_list
    global yeni_ratio6_list
    ratio6_list=[]
    for sentence in preprocessed_sentences:
        words = nltk.word_tokenize(sentence)
        numerical_count = 0
        for word in words:
            if word.isdigit():
                numerical_count += 1
            try:
                float(word)
                numerical_count += 1
            except ValueError:
                pass
        total_numerical_count += numerical_count
        total_word_count += len(words)
        if  total_numerical_count >1:
            ratio6 = total_numerical_count / total_word_count
        else:
            ratio6=0
        ratio6_list.append(ratio6)

        sayi = 0.4
        yeni_ratio6_list = []

        for i in range(len(ratio6_list)):
            carpim = ratio6_list[i] * sayi
            yeni_ratio6_list.append(carpim)
            #print(yeni_ratio6_list)      
    return yeni_ratio6_list

#başlıkta kelimelerin olup olmadığı kontrolü ///////////////////////////////////////////////////////////////////////

def baslik_kontrol():
    baslik = preprocessed_sentences[0].strip()
    print(baslik)
    ayrilmis_baslik = set(word_tokenize(baslik))
    global baslikoran_list
    global yeni_baslikoran_list
    baslikoran_list=[]
    for cumle in preprocessed_sentences:
        if cumle == preprocessed_sentences[0]: 
            continue
        ayrilmis_cumle = set(word_tokenize(cumle))
        common_tokens = ayrilmis_baslik & ayrilmis_cumle  # set kesişimi için & operatörünü kullanıyoruz
        if common_tokens:
            print("Başlık kelimeleri cümlede bulunuyor:", cumle)
            baslik_oran = len(common_tokens)/len(cumle)
            print("Toplam ortak kelime sayısı ve oranı:", len(common_tokens),baslik_oran)
        else:
            baslik_oran=0
        baslikoran_list.append(baslik_oran)
        print(baslikoran_list) 
        
        sayi = 0.4
        yeni_baslikoran_list = []

        for i in range(len(baslikoran_list)):
            carpim = baslikoran_list[i] * sayi
            yeni_baslikoran_list.append(carpim)
            print(yeni_baslikoran_list) 

#Her kelimenin TF-IDF değerinin hesaplanması ////////////////////////////////////////////////////////////////////////////////////////
def tema_kelime_hesaplama(preprocessed_sentences):
    toplam_metin = ' '.join(preprocessed_sentences)

    # Metindeki tüm kelimelerin frekansını hesaplama
    kelime_frekansi = Counter(toplam_metin.split())

    # Toplam kelime sayısının yüzde 10'unu hesaplama
    toplam_kelime_sayisi = sum(kelime_frekansi.values())
    encok_gecen_kelime = int(toplam_kelime_sayisi * 0.1)

    # En yüksek frekansa sahip tema kelimelerini belirleme
    tema_kelimeler = [word for word, frequency in kelime_frekansi.most_common(encok_gecen_kelime)]

    return tema_kelimeler

def cumledeki_tema_kelime_sayisi(sentence8, tema_kelimeler):
    kelime_frekansi = Counter(sentence8.split())
    tema_kelime_sayac = sum(kelime_frekansi[word] for word in tema_kelimeler)
    return tema_kelime_sayac
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def button_click():
    global has_numerical_data  
    global yeni_ratios
    yeni_ratios = calculate_proper_noun_ratio(preprocessed_sentences)
    print(preprocessed_sentences)
    print("Cümledeki özel isim oranı:", yeni_ratios)
    #proper_nouns = check_proper_nouns(preprocessed_sentences[3])  # Örnek olarak ilk cümledeki özel isimleri alma
    #print("Bulunan özel isimler:", proper_nouns)
    has_numerical_data = calculate_numerical_ratio_in_sentences(preprocessed_sentences)
    print("CÜMLEDEKİ NUMERİK VERİ ORANI:",has_numerical_data)
    #print("CÜMLEDEKİ NUMERİK VERİ ORANI:",ratio6)
    
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def onemli_kelimeler():#tema kelimesi bulan buton çalıştıran fonksiyon
    tema_kelimeler = tema_kelime_hesaplama(preprocessed_sentences)
    print("Tema Kelimeleri:", tema_kelimeler)
    global temakelimeorani_list
    global yeni_temakelime_liste
    temakelimeorani_list=[]
    for sentence in preprocessed_sentences:
        tema_kelime_sayac = cumledeki_tema_kelime_sayisi(sentence, tema_kelimeler)
        print("Cümle:", sentence)
        print("Tema Kelime Sayısı:", tema_kelime_sayac)
        if tema_kelime_sayac>1:
            temakelimeorani= tema_kelime_sayac/len(sentence)
        else:
            temakelimeorani=0
        
        temakelimeorani_list.append(temakelimeorani)        
        print(temakelimeorani_list)
        
        sayi = 0.4
        yeni_temakelime_liste = []

        for i in range(len(temakelimeorani_list)):
            carpim = temakelimeorani_list[i] * sayi
            yeni_temakelime_liste.append(carpim)
            print(yeni_temakelime_liste)
       


#////////////////////////////////////////////////////////////////////////
def puan_hesapla():
# Skorları tutacak liste
    global result
    result0 = [ yeni_ratio6_list[i] + yeni_temakelime_liste[i]+ yeni_bag_oran_list[i]+ yeni_ratios[i] for i in range(1, len(has_numerical_data))]
    #print(result0)

   #result = [x + y + z+ k for x, y, z, k in zip(has_numerical_data,ratios,temakelimeorani_list,baslikoran_list )]
    result = [x + y  for x, y in zip(result0,yeni_baslikoran_list )]

    #print(result)
    sorted_data = sorted(enumerate(result), key=lambda x: x[1])
    
    #ekrana_yazilacak_veri.insert(tk.END, sorted_data)

    for index, value in sorted_data:
        print("Index:", index, "Value:", value)

def puan_cumle_eslesmesi():
    global preprocessed_sentences
    eslesmis_liste = list(zip(preprocessed_sentences[1:], result))
    #print(eslesmis_liste)
    eslesmis_liste.sort(key=lambda x: x[1], reverse=True)  #sıralıyor

    #sorted_list1 = eslesmis_liste[:5]

    yarisi = round(len(eslesmis_liste) / 2)  # Listenin yarısını bul
    yuzde_elli_veri = eslesmis_liste[:yarisi] 
    #print("sonradan hesaplanan veri",yuzde_elli_veri)
     # İlk yarısını al
    #print(sorted_list1)
    #ekrana_yazilacak_veri.insert(tk.END, sorted_list1)
    
    for veri_cifti in yuzde_elli_veri:
        birinci_veri = veri_cifti[0]
        print(birinci_veri)

        ekrana_yazilacak_veri.insert(tk.END, birinci_veri)
    
    #for i in sorted_list1:
     #   for j in eslesmis_liste:
      #      if j[0] == i:
       #         print(j[1],j[0])
def sil():
    ekrana_yazilacak_veri.delete(1.0, tk.END)
def yaziyi_sil():
    yazi = ekrana_yazilacak_veri.get(1.0, tk.END)
    if yazi.strip() != "":
        sil()

#///////////////////////////////PUANI GÖSTEREN EKRAN///////////////////////////////
def puan_yazdir_pencere():
    print(result)
    # Yeni pencere oluşturma
    puan_pencere = tk.Toplevel(root)
    
    puan_threshold_degeri = float(score_entry.get())
    
    # Listbox oluşturma
    listbox = tk.Listbox(puan_pencere,width=80, height=40)
    listbox.pack()
    

    # Verileri Listbox'a yazdırma
    for i, veri in enumerate(result, start=2):
        if veri > puan_threshold_degeri:
            listbox.insert(tk.END, f"{i} CÜMLENİN PUANI = . {veri}")


def get_bert_embeddings(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.pooler_output
    return embeddings

# İki cümle arasındaki benzerliği kosinüs benzerliği yöntemiyle hesaplama
def calculate_similarity(sentence1, sentence2):
    embeddings1 = get_bert_embeddings([sentence1])
    embeddings2 = get_bert_embeddings([sentence2])
    similarity = cosine_similarity(embeddings1.detach().numpy(), embeddings2.detach().numpy())
    return similarity[0][0]

def calculate_button_clicked():
    num_sentences = len(preprocessed_sentences)
    benzerlik_matrisi = np.zeros((num_sentences, num_sentences))
    
    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):
            similarity = calculate_similarity(preprocessed_sentences[i], preprocessed_sentences[j])
            benzerlik_matrisi[i][j] = similarity
            benzerlik_matrisi[j][i] = similarity
    
    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):
            print("Cümle", i+1, "ile Cümle", j+1, "arasındaki benzerlik:", benzerlik_matrisi[i][j])
#//////////////////////////////////////////////////////////////////////////////////////////////
def create_similarity_graph(preprocessed_sentences,threshold):
    num_sentences = len(preprocessed_sentences)
    similarity_graph = nx.Graph()

    # Her cümleyi düğüm olarak grafa ekleyelim
    for i in range(num_sentences):
        similarity_graph.add_node(i)

    # Benzerlik skorlarına dayalı olarak kenarları grafa ekleyelim
    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):
            similarity = calculate_similarity(preprocessed_sentences[i], preprocessed_sentences[j])
            if similarity >= threshold:
                similarity_graph.add_edge(i, j, weight=similarity)
        

    return similarity_graph


def visualize_graph(similarity_graph):
    global bag_oran
    global bag_oran_list
    global yeni_bag_oran_list
    bag_oran_list=[]
    pos = nx.spring_layout(similarity_graph, k=2.4, seed=42)
    labels = {i: f"Cümle {i + 1}" for i in similarity_graph.nodes()}
    weights = nx.get_edge_attributes(similarity_graph, 'weight')

    # Grafik ayarları
    plt.figure(figsize=(12, 6))
    nx.draw_networkx_nodes(similarity_graph, pos, node_color='yellow', node_size=1400, alpha=0.9)
    nx.draw_networkx_labels(similarity_graph, pos, labels, font_size=12)
    nx.draw_networkx_edges(similarity_graph, pos, width=[w*3 for w in weights.values()], edge_color='gray', alpha=0.6)
    nx.draw_networkx_edge_labels(similarity_graph, pos, edge_labels=weights, font_size=8)

    # Eşik değerini geçen düğümlerin bağlantı sayılarını düğümlerin yanına yazma
    node_connections = dict(similarity_graph.degree())

    
    total_degree = (similarity_graph.size())


    for node, degree in node_connections.items():
        #total_degree = sum([similarity_graph.degree(x) for x in similarity_graph.nodes()])
        if degree>0:
            bag_oran = degree / total_degree
        else:
            bag_oran=0    
        #print("node bağ:",degree)
        #print("oran:",bag_oran)
        #print(total_degree)

        plt.text(pos[node][0], pos[node][1]+0.08, f"bağ sayısı: {degree}", color='green', ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.2'))
        bag_oran_list.append(bag_oran)
        #print(bag_oran_list)

        sayi = 0.4
        yeni_bag_oran_list = []

        for i in range(len(bag_oran_list)):
            carpim = bag_oran_list[i] * sayi
            yeni_bag_oran_list.append(carpim)
            print(yeni_bag_oran_list)
        
        
    # Grafik ayarları
    plt.title("Graf Yapısı", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    


def graf_olustur(error_label=None):    
    threshold_text = similarity_entry.get()  # Benzerlik eşiği metin değerini al
    if not threshold_text:  # Giriş boşsa hata mesajı ver
        error_label.config(text="Benzerlik eşiği girmelisiniz.")
        return

    try:
        threshold = float(threshold_text)  # Metin değerini float olarak döndür
        if threshold < 0 or threshold > 1:  # Geçerli aralık kontrolü yap
            raise ValueError
    except ValueError:  # Geçersiz girişse hata mesajı ver
        error_label.config(text="Geçerli bir benzerlik eşiği girin (0 ile 1 arasında).")
        return

    similarity_graph = create_similarity_graph(preprocessed_sentences,threshold)

    # Benzerlik grafiğini görselleştirin
    visualize_graph(similarity_graph)    



def birden_cok_fonksiyon():
    calculate_button_clicked()
    button_click()
    baslik_kontrol()
    onemli_kelimeler()
    puan_hesapla()



nltk.download('punkt')

root = tk.Tk()
root.title("Doküman Yükle")
button = tk.Button(root, text="Doküman Seç", command=select_file)
button.pack()

#create_graph_button = tk.Button(root, text="Grafi Oluştur", command=visualize_graph)
#create_graph_button.pack()
#create_graph_button.configure(state='disabled')


# Cümle benzerliği için threshold seçimi
frame3 = tk.Frame(root)
frame3.pack()

similarity_label = tk.Label(frame3, text="Cümle Benzerliği Threshold")
similarity_label.pack(side=tk.LEFT)

similarity_entry = tk.Entry(frame3)
similarity_entry.pack(side=tk.LEFT)

# Cümle skorunu belirlemek için threshold seçimi
frame4 = tk.Frame(root)
frame4.pack()

score_label = tk.Label(frame4, text="Cümle Skoru Threshold")
score_label.pack(side=tk.LEFT)

score_entry = tk.Entry(frame4)
score_entry.pack(side=tk.LEFT)

#METİN KUTUSU
ekrana_yazilacak_veri = tk.Text(root)
ekrana_yazilacak_veri.pack()

#////////////////////////////////////

#calculate_button = tk.Button(root, text="Benzerlik Hesapla", command=calculate_button_clicked)
#calculate_button.pack()


#button = tk.Button(root, text="özel isim oranı", command=button_click)
#button.pack()

grafbutton = tk.Button(root, text="GRAF OLUŞTUR", command=graf_olustur)
grafbutton.pack()

#baslikbutton = tk.Button(root, text="Başlıktaki kelimeleri cümlede var mı?", command=baslik_kontrol)
#baslikbutton.pack()

#totalkbutton = tk.Button(root, text="Tema kelimelerini bul:", command=onemli_kelimeler)
#totalkbutton.pack()

#puanbutton = tk.Button(root, text="puan bul:", command=puan_hesapla)
#puanbutton.pack()
 


dugme = tk.Button(root, text="PARAMETRELERİ HESAPLA", command=birden_cok_fonksiyon)
dugme.pack()
puanpenceresi = tk.Button(root, text="PUAN YAZDIR", command=puan_yazdir_pencere)
puanpenceresi.pack()

eslesmebutton = tk.Button(root, text="METİNİ ÖZETLE", command=puan_cumle_eslesmesi)
eslesmebutton.pack() 

silme_dugmesi = tk.Button(root, text="METNİ SİL", command=yaziyi_sil)
silme_dugmesi.pack()




root.mainloop()









