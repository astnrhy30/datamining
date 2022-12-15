import streamlit as slit
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle
from streamlit_lottie import st_lottie as slit_lt
from streamlit_lottie import st_lottie_spinner
from PIL import Image as img
from streamlit_option_menu import option_menu as opmen
from sklearn.impute import SimpleImputer as simp
from sklearn.compose import ColumnTransformer as colTran
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA


#--- Icon/favicon diambil dari https://www.webfx.com/tools/emoji-cheat-sheet/#tabs-3
slit.set_page_config(page_title = "20-083 Web Datamining", page_icon=":sunflower:", layout="wide")

# --- Fungsi Pemanggilan URL ---
def load_lottieurl(url):
    rq = requests.get(url)
    if rq.status_code != 200:
        return None
    return rq.json()

# -------------------------------- LOAD ASSETS && VARIABEL --------------------------------

#--- Animasi diambil dari https://lottiefiles.com/
lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_FLTWez0aGe.json")
lottie_maternal = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_jmfiilsd.json")

#-- Gambar diambil dari file pada folder images
logo_Trunojoyo = img.open("images/Logo UTM 50px.png")

# --- Variabel untuk import dataset jadi dataframe ---
df = pd.read_csv("Maternal-Health-Risk-Data-Set.csv")

# --- Variabel untuk menampung atribut reguler
var_reg = df.iloc[:, :-1].values

# --- Variabel untuk menampung atribut label/kelas
var_lbl = df.iloc[:, -1].values

# --- Mendeteksi nan dan mengisi dengan nilai lain (mean/modus)
imputer = simp(missing_values=np.nan, strategy='most_frequent')
imputer.fit(var_reg[:, 0:6])
var_reg[:, 0:6] = imputer.transform(var_reg[:, 0:6])
var_regImp = var_reg[:, 0:6]

# --- Variabel untuk mengubah label/kelas menjadi tipe data numerik
le      = LabelEncoder()
le_lbl = le.fit_transform(var_lbl)

# --- Variabel atribut train dan test
reg_train, reg_test, lbl_train, lbl_test = train_test_split(var_regImp, le_lbl, test_size=0.2, random_state=1 )

# --- Variabel skala dari train dan testnya
sc = StandardScaler()
new_regTrain = sc.fit_transform(reg_train[:])
new_regTest = sc.fit_transform(reg_test[:])

# --------------------------------
# --- Navigation Top-bar ---
selected = opmen(
    menu_title = "Menu Utama",
    menu_icon  = "cast",
    options    = ["Home", "About Dataset", "Prepocessing", "Modelling", "Prediction"],
    icons      = ["bookmark", "filter", "back", "app-indicator", "archive"],
    default_index = 0,
    orientation= "horizontal",
    styles     = {
        "options": {
            "text-align": "center"
        }
    }
)

if selected == "Home":
    # --- Head Section ---
    with slit.container():
        slit.title("Halo, saya Astia Nurrahmayanti :wave:")
        slit.subheader("Mahasiswa Program Studi S1 Teknik Informatika, Fakultas Teknik Universitas trunojoyo Madura")
        slit.write("Saat ini, saya sedang menempuh semester 5 dan mengambil Mata Kuliah Penambangan Data (Datamining) kelas C yang diampu Bapak Mula'ab, S.Si., M.Kom.")
        slit.write("Website ini dibuat untuk memenuhi tugas akhir mata kuliah tersebut.")
    # --- Content Section ---
    with slit.container():
        slit.write("---")
        left_column, right_column = slit.columns(2)
        with left_column:
            slit_lt(lottie_coding, key="datamining")
        with right_column:
            slit.header("Apa yang ada pada proyek tugas akhir ini?")
            slit.write(
                """
                Pada proyek akhir ini akan dibuat sebuah aplikasi datamining sederhana untuk melakukan prediksi dan klasifikasi dari sekelompok dataset dengan topik tertentu. Secara garis besar inti dari project akhir ini, antara lain:
                - Penyiapan atau mencari dataset yang akan diproses dalam suatu metode.
                - Pendeskripsian dataset yang dipilih, serta menyertakan sumber link.
                - Melakukan pemodelan dengan metode Decision Tree, KNN, dan Naive Bayes.
                - Melakukan perkiraan sebuah hasil prediksi yang dari nilai tertentu.
                """
            )
            slit.write("[Kunjungi juga, Github Pages Datamining milik saya >](https://astnrhy30.github.io/datamining/intro.html)")

if selected == "About Dataset":
    slit.title("Data Risiko Kesehatan Ibu (Maternal Health Risk Data)")
    # --- Menampilkan dataframe ---
    slit.dataframe(df)
    # --- Deskripsi Dataset/Main Content ---
    slit.write(
        """
        Dataset ini merupakan dataset yang telah dikumpulkan dari bermacam rumah sakit, klinik, komunitas
        serta layanan kesehatan ibu melalui sistem pemantuan resiko berbasis IoT. Tujuan utama dikumpulkannya
        data tersebut agar dapat mengetahui kondisi kesehatan apa yang merupakan indikasi terkuat untuk resiko
        kesehatan yang bagi seorang pasien hamil. Sehingga, target pasien utamanya wanita yang sedang mengandung
        atau ibu-ibu hamil.
        """
    )
    slit.write("[Sumber Dataset > www.kaggle.com/datasets/](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data?resource=download)")
    slit.write(
        """
        Dataset tersebut memiliki beberapa kolom (field) yang merupakan spesifikasi/cirikhas sebagai suatu aspek
        yang dapat mempengaruhi kesehatan seorang yang sedang mengandung. Kolom tersebut terdiri atas:
        - Usia, merupakan bilangan dalam hitungan tahun yang menentukan tingkat umur pasien mulai dari rentang usia 10 tahun.
        - Sistolik, merupakan nilai indeks batas atas dari tekanan darah. Apabila berada pada tensi normal yang tidak boleh lebih dari 120 mmHg.
        - Diastolik, merupakan nilai indeks batas atas dari tekanan darah. Apabila berada pada tensi normal yang tidak boleh kurang dari 80 mmHg.
        - Blood Glucose (BS), merupakan kadar gula darah yang dinyatakan dalam satuan konsentrasi molar, mmol/L. Kadar gula normal akan bernilai jika hasil kurang dari 7.777777 mmol/L (140 mg/dL).
        - Detak Jantung, merupakan jumlah detak jantung yang dihasilkan dalam satu menit.
        - Tingkat Risiko, merupakan kategori tingkat hasil prediksi dari data-data yang diperoleh.
        """
    )
    slit.write("TOTAL DATA/BARIS = 1024 pasien")
    slit.write("JUMLAH KOLOM     = 7")
    slit.subheader(" ")
    slit.write("---")
    slit.write("SEKILAS INFORMASI")
    slit.subheader("Apa itu maternal?")
    slit.write(
        """
        Mengutip Kamus Besar Bahasa Indonesia, maternal adalah sesuatu yang berhubungan dengan ibu. Contoh, insting seorang ibu bisa juga disebut insting maternal.
        Maternal juga bisa berarti sesuatu yang melalui ibu. Seperti misalnya, paman maternal adalah paman yang berasal dari keluarga ibu.
        """
    )
    slit_lt(lottie_maternal, height=300, key="ibu")


if selected == "Prepocessing":
    slit.title("Pemrosesan Data (Prepocessing)")
    slit.write(
        """
        Dalam pemrosesan suatu dataset maka akan dilakukan beberapa tahapan di dalamnya, antara lain:
        - Pembersihan data, hal ini dilakukan karena data awal pada umumnya masih "kotor" agar menghasilkan data yang baik (output akurat dan nilai akhir presisi).
        - Integrasi data, pengkombinasian baris-baris pada dataset menjadi satu.
        - Transformasi data, mengubah data ke dalam bentuk yang sesuai.
        - Reduksi data, proses pemilihan data, pemusatan data, penyederhanaan data, hingga transformasi data kasar yang ditemukan di lapangan. Lalu, di dalamnya juga ada proses dikritisasi data.
        """
    )
    slit.write("---")
    slit.header("Tahap Prepocessing")
    slit.write("1) Memisahkan kolom atribut reguler dengan atribut kelas.")
    col_reg, col_lbl = slit.columns(2)
    with col_reg:
        slit.write(var_reg)
    with col_lbl:
        slit.write(var_lbl)
    slit.write("2) Mengecek adakah nilai data pada suatu baris di kolom reguler tertentu yang kosong atau tidak. Lalu, mengisinya dengan nilai mayoritas.")
    slit.write(var_regImp)
    slit.write("3) Mengubah data yang bertipe data nominal ke bentuk matriks/numerik.")
    slit.write(le_lbl)
    slit.write("4) Membagi data menjadi data untuk training dan testing.")
    slit.write("- Data atribut reguler")
    regTrain, regTest = slit.columns(2)
    with regTrain:
        slit.write(reg_train)
    with regTest:
        slit.write(reg_test)
    slit.write("- Data atribut label")
    lblTrain, lblTest = slit.columns(2)
    with lblTrain:
        slit.write(lbl_train)
    with lblTest:
        slit.write(lbl_test)
    slit.write("5) Membuat skala dari masing-masing data tiap kolom agar jaraknya tidak begitu lebar/selisihnya tidak banyak.")
    newRegTrain, newRegTest = slit.columns(2)
    with newRegTrain:
        slit.write(new_regTrain)
    with newRegTest:
        slit.write(new_regTest)

if selected == "Modelling":
    slit.title("Pemodelan Data")
    slit.write(
        """
        Pada bagian ini akan dilakukan pembandingan proses klasifikasi dari beberapa model/algoritma antara lain Pohon Keputusan (Decision Tree), K-NN, dan Naive Bayes.
        Pemodelan data ini bertujuan untuk mencari yang terbaik dalam klasifikasi data dari segi tingkat akurasi dan visualisasi data plotting. Berikut ini ringkasan
        mengenai masing-masing models:
        """
    )
    slit.write("a) Pohon Keputusan (Decision Tree)")
    slit.write("Decision Tree adalah pohon keputusan yang akan mengibah banyak data menjadi suatu keputusan. Dalam pengambilan keputusan dilakukan secara objektif dan detail karena dimulai dari root node hingga leaf node.")
    slit.write("b) K-Nearest Neighbors (K-NN)")
    slit.write("K-Nearest Neighbors adalah algoritma yang berfungsi untuk melakukan klasifikasi suatu data berdasarkan data pembelajaran (train data sets), yang diambil dari k tetangga terdekatnya (nearest neighbors). Dengan k merupakan banyaknya tetangga terdekat.")
    slit.write("c) Naive Bayes")
    slit.write("Naive Bayes adalah algoritma yang terinspirasi dari teorema Bayes (seorang ilmuwan inggris Thomas Bayes) dengan membuat peluang di masa depan berdasarkan pengalaman di masa lalu.")
    slit.write("---")

    slit.header("Implementasi Pemodelan Data")
    # ---- Select Box Algoritma ----
    AlgoritmaOption = slit.selectbox(
        'Pilih Algoritma',
        ('Decision Tree', 'KNN', 'Naive Bayes')
    )

    # --- Fungsi Penentuan Paramter ---
    def tambah_parameter(NamaAlgoritma):
        parameter = dict()
        if NamaAlgoritma == 'Decision Tree':
            max_depth = slit.slider('Maximal_depth', 1, 20)
            parameter['Maximal_depth'] = max_depth
            min_samples_leaf = slit.slider('Minimal_leaf', 1, 20)
            parameter['Minimal_leaf'] = min_samples_leaf
        elif NamaAlgoritma == 'KNN':
            K = slit.slider('Konstanta (K)', 1, 15)
            parameter['Konstanta (K)'] = K
        else:
            alpha = slit.slider('Alpa (α)', 0, 10)
            parameter['Alpa (α)'] = alpha
        return parameter

    slit.write("#### Algoritma", AlgoritmaOption)
    slit.write("Jumlah Baris dan Kolom = ", df.shape)
    slit.write("Jumlah Kelas/Label     = ", len(np.unique(var_lbl)))
    parameter = tambah_parameter(AlgoritmaOption)
    
    # --- Fungsi Penentuan Metode Klasifikasi  ---
    def pilih_klasifikasi(Algoritma, params):
        algo = None
        if Algoritma == 'Decision Tree':
            algo = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=params['Maximal_depth'], min_samples_leaf=parameter['Minimal_leaf'])
        elif Algoritma == 'KNN':
            algo = KNeighborsClassifier(n_neighbors=params['Konstanta (K)'])
        else:
            algo = GaussianNB(priors=None, var_smoothing=1*params['Alpa (α)']-0.9)
        return algo
    algo = pilih_klasifikasi(AlgoritmaOption, parameter)

    # --- PROSES KLASIFIKASINYA ---
    algo.fit(reg_train, lbl_train)
    lbl_pred = algo.predict(reg_test)

    # --- Variabel Akurasi ---
    acu = accuracy_score(lbl_test, lbl_pred)
    slit.write("Akurasi                = ", acu)

    # --- PLOT DATA SET ---
    pca = PCA(2) # memproyeksikan data ke dalam 2 komponen PCA
    reg_projected = pca.fit_transform(var_reg)
    var_reg1 = reg_projected[:, 0]
    var_reg2 = reg_projected[:, 1]
    fig = plt.figure(figsize=(7,7))
    plt.scatter(var_reg1, var_reg2, c=le_lbl, alpha=0.8, cmap='plasma')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plot1, plot2 = slit.columns(2)
    with plot1:
        slit.pyplot(fig)
      
if selected == "Prediction":
    slit.title("Prediksi")
    slit.write("Coba pemrediksian resiko kesehatan ibu hamil (istri/diri anda) di bawah ini. Untuk memastikan kesehatan ibu dan janin :)")
    slit.write("---")
    slit.subheader("Input Data")

    input1, input2 = slit.columns(2)
    with input1:
        Usia         = slit.number_input("Age (Usia)")
        Sistol       = slit.number_input("SystolicBP (Sistol)")
        Diastol      = slit.number_input("DiastolicBP(Diastol)")
    with input2:
        GulaDarah   = slit.number_input("Blood Glucose (Gula Darah)")
        SuhuBadan    = slit.number_input("Body Temperature (Suhu Tubuh)")
        DetakJantung = slit.number_input("Heart Rate (Detak Jantung)")

    def Submit():
        # --- Mengolah data yang telah dimasukkan ---
        dataArray = np.array([[ Usia, Sistol, Diastol, GulaDarah, SuhuBadan, DetakJantung ]])
        test_data = np.array(dataArray).reshape(1, -1)
        test_data = pd.DataFrame(test_data, columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])

        # --- Normalisasi data ---
        test_data                = pd.DataFrame(test_data)
        test_data['Age']         = test_data['Age'].astype('category')
        test_data['SystolicBP']  = test_data['SystolicBP'].astype('category')
        test_data['DiastolicBP'] = test_data['DiastolicBP'].astype('category')
        test_data['BS']          = test_data['BS'].astype('category')
        test_data['BodyTemp']    = test_data['BodyTemp'].astype('category')
        test_data['HeartRate']   = test_data['HeartRate'].astype('category')

        cat_columns            = test_data.select_dtypes(['category']).columns
        test_data[cat_columns] = test_data[cat_columns].apply(lambda x: x.cat.codes)

        filename = "Maternal-Health-Risk-Data-Set.csv"
        scaler   = joblib.load(filename)

        
        test_d   = scaler.fit_transform(test_data)

        # --- Load KNN ---
        filenameModelKnnNorm = 'modelKnnNorm.pkl'
        knn = joblib.load(filenameModelKnnNorm)
        pred = knn.predict(test_d)

        # --- Load Gausian ---
        filenameModelGau = 'modelGau.pkl'
        gnb = joblib.load(filenameModelGau)
        pred = gnb.predict(test_d)

        # --- Load Decision Tree ---
        filenameModeld3 = 'modeld3.pkl'
        d3 = joblib.load(filenameModeld3)
        pred = d3.predict(test_d)

        # --- Setelah Button di Klik ---
        slit.header("Tabel Data dari Inputan")
        slit.write("Berikut ini tabel hasil input data testing yang akan diklasifikasi:")
        slit.dataframe(dataArray)

        slit.header("Hasil Prediksi")
        K_Nearest_Naighbour, Naive_Bayes, Decision_Tree = slit.tabs(["K-Nearest Neighbour", "Naive Bayes Gaussian", "Decision Tree"])
        with K_Nearest_Naighbour:
            slit.subheader("Model K-Nearest Neighbour")
            pred = knn.predict(test_d)
            if pred[0]== 0:
                slit.write("Hasil Klasifikaisi : Low Risk (Risiko Rendah)")
            elif pred[0]== 1 :
                slit.write("Hasil Klasifikaisi : Middle Risk (Risiko Sedang)")
            else:
                slit.write("Hasil Klasifikaisi : High Risk (Resiko Tinggi)")
    
        with Naive_Bayes:
            slit.subheader("Model Naive Bayes Gausian")
            pred = gnb.predict(test_d)
            if pred[0]== 0:
                slit.write("Hasil Klasifikaisi : Low Risk (Risiko Rendah)")
            elif pred[0]== 1 :
                slit.write("Hasil Klasifikaisi : Middle Risk (Risiko Sedang)")
            else:
                slit.write("Hasil Klasifikaisi : High Risk (Resiko Tinggi)")

        with Decision_Tree:
            slit.subheader("Model Decision Tree")
            pred = d3.predict(test_d)
            if pred[0]== 0:
                slit.write("Hasil Klasifikaisi : Low Risk (Risiko Rendah)")
            elif pred[0]== 1 :
                slit.write("Hasil Klasifikaisi : Middle Risk (Risiko Sedang)")
            else:
                slit.write("Hasil Klasifikaisi : High Risk (Resiko Tinggi)")        
    submitted = slit.button("Submit")
    if submitted:
        Submit()



# --- Footer Section ---
with slit.container():
    slit.write("---")
    column1, column2, column3 = slit.columns(3)
    with column1:
        slit.write("+62 838 9354 5155")
        slit.write("astiarahma131@gmail.com")
        slit.write("Surabaya, Indonesia")
    with column2:
        slit.write("[Github](https://github.com/astnrhy30)")
        slit.write("[LinkedIn](https://www.linkedin.com/in/astia-nurrahmayanti-58345b148/)")
        slit.write("[astia-rahma.blogspot.com](https://astia-rahma.blogspot.com/)")
    with column3:
        slit.image(logo_Trunojoyo)
        slit.write("UNIVERSITAS TRUNOJOYO MADURA")
        slit.write("Jl. Raya Telang,PO BOX 02 Kecamatan Kamal")
        slit.write("Bangkalan, Madura, Jawa Timur 69162 Indonesia")

#--- Credit Section ---
with slit.container():
    slit.write("---")
    slit.write("Copyright © 2022 | 2004111000083_Astia-Nurrahmayanti")