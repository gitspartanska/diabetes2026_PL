import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os # Import os for file existence checks
from model_utils import RenameDPF

# --- Project Title and Introduction ---
st.title('🩺 Przewidywanie ryzyka cukrzycy')
st.badge("MIT license", color="blue")
st.write('### Wstęp\n')
txt ='Oto prosta interaktywna aplikacja pozwalająca za pomocą modelu ML oszacować prawdopodobieństwo wystąpienia cukrzycy na podstawie uzupełnionych danych. Cukrzyca jest ogólną nazwą grupy chorób metabolicznych, które charakteryzują się podwyższonym poziomem glukozy we krwi (hiperglikemią), wywołaną zbytn niską produkcją insuliny lub jej defektem. Cukrzyca jest chorobą dotykającą miliony ludzi na całym świecie, a jej wczesne wykrycie pozwala na podjęcie natychmiastowego leczenia i zapobiegnięcie pogorszeniu sie stanu zdrowia.'
st.markdown(f'<div style="text-align: justify;">{txt}</div>', unsafe_allow_html=True)
st.write("Więcej na temat cukrzycy można znaleźć na poniższych stronach:\n * [https://pl.wikipedia.org/wiki/Cukrzyca](https://pl.wikipedia.org/wiki/Cukrzyca)\n * Cukrzyca insulinozależna [ICD10 E10](https://remedium.md/icd10/zaburzenia-wydzielania-wewnetrznego-stanu-odzywienia-i-przemiany-metabolicznej/cukrzyca-insulinozalezna) \n* Cukrzyca insulinoniezależna [ICD10 E11](https://remedium.md/icd10/zaburzenia-wydzielania-wewnetrznego-stanu-odzywienia-i-przemiany-metabolicznej/cukrzyca-insulinoniezalezna)\n * Cukrzyca związana z niedożywieniem [ICD10 E12](https://remedium.md/icd10/zaburzenia-wydzielania-wewnetrznego-stanu-odzywienia-i-przemiany-metabolicznej/cukrzyca-zwiazana-z-niedozywieniem)\n * [pacjent.gov.pl](https://pacjent.gov.pl/jak-zyc-z-choroba/jak-zyc-z-cukrzyca) ")




# --- Model Loading ---

@st.cache_resource
def load_model(MODEL_PATH, m= 1,if_lib=False):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Nie znaleziono pliku z modelem: '{MODEL_PATH}'. Proszę upewnić się, że znajduje się on w tym samym katalogu.")
        st.stop() # Halts app execution if the model is missing
    try:
        if if_lib:
            model  = joblib.load(MODEL_PATH)
        else:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        st.sidebar.success(f"Model {m} skutecznie wczytany!")
        return model
    except Exception as e:
        st.error(f"Podczs wczytywnia modelu wystąpił niespodziewany błąd: {e}")
        st.stop() # Halts app execution for any other loading errors

# --- Data Loading for Overview ---
# @st.cache_data is used to cache the dataset, loading it only once
# which prevents repeated disk reads and contributes to faster startup.
@st.cache_data
def load_overview_data():
    """Loads the diabetes dataset for overview and preprocessing insights."""
    DATA_PATH = 'diabetes.csv'
    if not os.path.exists(DATA_PATH):
        st.warning(f"Nie znaleziono pliku '{DATA_PATH}' do stworzenia przeglądu statystycznego. Niektóre sekcje mogą nie być wczytane.")
        return pd.DataFrame()
    try:
        dataset = pd.read_csv(DATA_PATH)
        return dataset
    except Exception as e:
        st.error(f"Wystąpił błąd podczas ładowania lub przetwarzania zbioru danych: {e}")
        st.exception(e) # Display full exception for debugging
        return pd.DataFrame()

# --- Image Loading Function ---
# Caching images prevents repeated loading from disk, useful for multiple image displays.
@st.cache_data
def load_image(image_path):
    """Loads an image and returns it, with caching."""
    if os.path.exists(image_path):
        return image_path
    else:
        st.warning(f"Nie znaleziono grafiki '{image_path}'. Proszę upewnij się, że grafika znajduje się w tym samym katalogu co skryt urchamiający aplikację.")
        return None # Return None if image not found



# --- Sidebar for User Input Features ---
with st.sidebar:
    st.header('Wprowadź cechy pacjenta')
    st.write('Ustaw poniższe suwaki, aby wprowadzić dane pacjenta:')

    # Function to collect user inputs via Streamlit sliders
    def user_input_features():
        pregnancies = st.sidebar.slider('Ciąże', 0, 17, 3, help='Liczba razy zajścia w ciążę.')
        glucose = st.sidebar.slider('glukoza (mg/dL)', 0, 200, 120, help='Stężenie glukozy we krwi po 2 godzinach w doustnym testu obciążenia glukozą.')
        blood_pressure = st.sidebar.slider('Ciśnienie krwi (mmHg)', 0, 122, 70, help='Ciśnienie rozkurczowe.')
        skin_thickness = st.sidebar.slider('Grubość skóry (mm)', 0, 99, 20, help='Grubość fałdu skórnego tricepsa.')
        insulin = st.sidebar.slider('Insulina (\u03BCU/ml)', 0, 846, 79, help='2 godzinne stężenie insuliny we krwi.')
        bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0, help='Wskaźnik masy ciała (masa w kg / (wzrost w m)^2).')
        diabetes_pedigree_function = st.sidebar.slider('Funkcja rodowodu cukrzycy', 0.078, 2.42, 0.471, help='Funkcja wyznaczająca wiarygodność wystąpienia cukrzycy na podstawie danych o historii rodziny. Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/?page=2')
        age = st.sidebar.slider('Wiek (w latach)', 21, 81, 33, help='Wiek pacjenta/uczestnika.')

        # Create a Pandas DataFrame from the collected inputs, as expected by the model.
        data = {
            'Ciąże': pregnancies,
            'Glukoza': glucose,
            'CiśnienieKrwi': blood_pressure,
            'GrubośćSkóry': skin_thickness,
            'Insulina': insulin,
            'BMI': bmi,
            'FunkcjaRodowoduCukrzycy': diabetes_pedigree_function,
            'Wiek': age
        }
        features = pd.DataFrame(data, index=[0]) # Single row DataFrame
        return features



# Get user inputs
df = user_input_features()

# Load the model (cached)
model1 = load_model("models/diabetes_model1.pkl")
model2 = load_model("models/diabetes_model2.pkl",2)
model3 = load_model("models/diabetes_model3.pkl","3a",True)
sacler3 =  load_model("models/scaler3.pkl","3b",True)


# Display user's input features
st.subheader('Dane wprowadzone przez użytkownika')
st.write("Aby wprowadzić własne dane do modelu, użyj suwaków znajdujących się w pasku po lewej stronie. Wprowadzone dane pokaża się w poniższej tabeli.")
st.dataframe(df,hide_index=True)

# --- Prediction Section ---

col_EN = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
col_PL = ['Ciąże','Glukoza','CiśnienieKrwi','GrubośćSkóry','Insulina', 'BMI','FunkcjaRodowoduCukrzycy','Wiek']
translate_col_names = dict(zip(col_PL, col_EN)) 
col_EN_to_PL =  dict(zip(col_EN, col_PL))

st.write("#### Przykładowe dane")
txt2 = "Wybierz jednen z wierszy, by przetestować modele na przykładowych danych. W przypadku niezaznaczenia żadnego wiersza, modele przyjmą na wejściu dane wprowadzone ręcznie przez użytkownika.\nPoniższe dane pochodzą od pacjentów z Sunyani Regional Hospital.\n"
st.markdown(f'<div style="text-align: justify;">{txt2}</div>', unsafe_allow_html=True)

df_mini =  pd.DataFrame({'Ciąże': [1,6,3,8,0],
        'Glukoza': [85,148,120,183,137],
        'CiśnienieKrwi': [66,72,70,64,40],
        'GrubośćSkóry': [29,35,20,0,35],
        'Insulina': [0,0,79,0,168],
        'BMI': [26.6,33.6,27.0,23.3,43.1],
        'FunkcjaRodowoduCukrzycy': [0.351,0.627,0.500,0.672,2.288],
        'Wiek': [31,50,35,32,33],
        'diagnoza':["raczej zdrowy","raczaj ma cukrzycę","trudno stwierdzić", 
        "wysoke ryzyko cukrzycy", "wysoke ryzyko cukrzycy"]})

event = st.dataframe(
    df_mini,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

with st.popover("Zakresy wartości parametrów u zdrowej osoby"):
    st.markdown("* Glukoza: 70-99 mg/dL\n * Ciśnienie krwi: 60-80 mmHg\n * BMI: 18.5 - 24.9\n * Funkcja Rodowodu Cukrzycy: 0.08 - 2.50")

st.subheader('Przewidziany wynik')
flex = st.container(horizontal=True, horizontal_alignment="left")


def show_results(prediction_proba, nr):
    # Display outcome
    st.markdown('---')
    st.write("Wyniki modelu 1")
    if prediction[0] == 1:
        st.error('**Predykcja: Pacjent prawdopodobnie ma cukrzycę.** 😔')
    else:
        st.success('**Predykcja: Pacjent prawdopodobnie NIE ma cukrzycy.** 😊')
    # Display confidence levels
    st.write(f"Pewność wyniku (Brak cukrzycy): **{prediction_proba[0][0]*100:.2f}%**")
    st.write(f"Pewność wyniku (Cukrzyca):\t **{prediction_proba[0][1]*100:.2f}%**")
    st.markdown('---')


# Button to trigger prediction
def make_pred(model, nr,scaler = None): 
    if model: # Ensure model is loaded before predicting
        try:
            # Perform prediction and get probability scores
            if event.selection["rows"]:
                df_tmp=df_mini.iloc[[event.selection["rows"][0]]]
                df_tmp=df_tmp.iloc[:, :-1]
                df_tmp=df_tmp.rename(columns=translate_col_names)
            else:
                df_tmp=df.rename(columns=translate_col_names)
            if scaler:
                df_tmp = scaler.transform(df_tmp)
            prediction = model.predict(df_tmp)
            prediction_proba = model.predict_proba(df_tmp)
            st.markdown('---')
            st.write(f"Wyniki modelu {nr}")
            if prediction[0] == 1:
                st.error('**Predykcja: Pacjent prawdopodobnie ma cukrzycę.** 😔')
            else:
                st.success('**Predykcja: Pacjent prawdopodobnie NIE ma cukrzycy.** 😊')
            # Display confidence levels
            st.write(f"Pewność wyniku (Brak cukrzycy): **{prediction_proba[0][0]*100:.2f}%**")
            st.write(f"Pewność wyniku (Cukrzyca):\t **{prediction_proba[0][1]*100:.2f}%**")
            st.markdown('---')
        except Exception as e:
            st.error(f"Podczas obliczeń wystąpił błąd: {e}")
            st.write("Prosimy sprawdzić poprawność danych wejściowych lub skontaktować się ze wsparciem w przypadku ponownego wystąienia problemu.")
            st.exception(e)
    else:
        st.error("Model ne został załadowany. Nie można wykonać predykcji.")

st.write("Aby wykonać predykcję na podstawie wybranych danych (domyślnie tych wprowadzonych ręcznie, natomiast w przypadku wybrania przykładow danych z tabeli zaznaczonego przykładu) naciśnij jeden z poniższych przycisków. Naciśnięcie ich spowoduje zwrócenie predykcji ryzyka wystąpienia cukrzycy przez wybrany model.")
if flex.button('Wykonaj predykcję (model 1)'):
    make_pred(model1,1)

if flex.button('Wykonaj predykcję (model 2)'):
    make_pred(model2,2)

if flex.button('Wykonaj predykcję (model 3)'):
    make_pred(model3,3,sacler3)

# Load data for overview (cached)
dataset = load_overview_data().rename(columns= col_EN_to_PL).rename(columns={"Outcome":"czy_chory"})

if not dataset.empty:
    st.write("### Opis bazy danych treningowych")
    st.write(f"Liczba wierszy: {dataset.shape[0]}")
    st.write(f"Liczba kolumn: {dataset.shape[1]}")
    st.scatter_chart(dataset,x= 'Wiek',y = 'BMI', x_label = "Wiek", color = "czy_chory")
    st.scatter_chart(dataset,x= 'Wiek',y = 'Insulina', x_label = "Wiek", color = "czy_chory")
    st.scatter_chart(dataset,x= 'Wiek',y = 'Glukoza', x_label = "Wiek", color = "czy_chory")
    st.scatter_chart(dataset,x= 'Wiek',y = 'CiśnienieKrwi', x_label = "Wiek", color = "czy_chory")



st.write("### Repozytoria\nPoniżej zostały podane linki do repozytoriów GithHub powiązanych z przedstawionymi modelami.")
st.write("* Bazowy [model1](https://github.com/Jeraldaw/huggingface-diabetes-predictor)\n * Bazowy [model2](https://github.com/faisal-titu/Diabetics_detection)\n * Bazowy [model3](https://huggingface.co/spaces/EiaminHassan5251/diabetes-prediction/tree/main)\n * [GitHub PL](https://github.com/gitspartanska) ")

st.caption("Pamiętaj, wyniki zwrócone przez modele nie stanowią podstwy do wystawienia prawnej diagnozy i nie zastąpią fachowej oceny lekarza.")
