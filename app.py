import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium


def main():
    predict()


def predict():
    st.header("Tahmin İçin Giriş Verileri")
    selected_yatakOdasi = st.number_input(
        "Yatak Odası Sayısı", min_value=1, max_value=20
    )
    selected_Metrekare = st.number_input("Metrekare", min_value=1.0, max_value=5000.0)
    selected_banyoSayisi = st.number_input("Banyo Sayısı", min_value=1, max_value=15)

    st.subheader("Harita Üzerinde Lokasyon Seçimi")
    default_Enlem = 49.26197951930051
    default_Boylam = -123.11450958251955

    map = create_map(default_Enlem, default_Boylam)
    location = st_folium(map, width=700, height=500)

    selected_Enlem = default_Enlem
    selected_Boylam = default_Boylam

    if location and "last_clicked" in location and location["last_clicked"]:
        selected_Enlem = location["last_clicked"]["lat"]
        selected_Boylam = location["last_clicked"]["lng"]
        st.write(f"Seçilen Enlem: {selected_Enlem}, Boylam: {selected_Boylam}")
    else:
        st.warning(
            "Harita üzerinde bir konum seçilmedi, varsayılan konum kullanılacak."
        )

    st.subheader("Model Seçimi")
    selected_model = st.selectbox(
        "Tahmin Modeli Seçiniz", ["Polynomial Regression", "Ridge Regression"]
    )

    prediction_value = create_prediction_value(
        selected_yatakOdasi,
        selected_banyoSayisi,
        selected_Metrekare,
        selected_Enlem,
        selected_Boylam,
    )

    if st.button("Tahmin Yap"):
        prediction_model = load_models(selected_model)

        if prediction_model is None:
            st.error("Seçilen model yüklenemedi. Lütfen dosyayı kontrol edin.")
            return

        try:
            result = predict_models(prediction_model, prediction_value)
            st.success("Tahmin Başarılı!")
            st.write("Tahmin Giriş Değerleri:")
            st.write(prediction_value)
            st.write(f"Tahmin Edilen Fiyat: {result} $")
        except Exception as e:
            st.error("Tahmin sırasında bir hata oluştu.")
            st.write(f"Hata Detayı: {e}")


def create_map(latitude, longitude):
    m = folium.Map(location=[latitude, longitude], zoom_start=10)
    folium.Marker([latitude, longitude], tooltip="Varsayılan Konum").add_to(m)
    return m


def load_models(modelName):
    try:
        if modelName == "Polynomial Regression":
            poly_model = joblib.load("polynomial_regression_model.pkl")
            poly_transformer = joblib.load("polynomial_features.pkl")
            return (poly_model, poly_transformer)
        elif modelName == "Ridge Regression":
            ridge_model = joblib.load("ridge_regression_model.pkl")
            return ridge_model
        else:
            return None
    except FileNotFoundError:
        st.error(
            f"{modelName} modeli bulunamadı. Lütfen doğru dosyayı yüklediğinizden emin olun."
        )
        return None


def create_prediction_value(Bedroom_count, Bathroom_count, Sqm, Latitude, Longitude):
    return pd.DataFrame(
        {
            "Bedroom Count": [Bedroom_count],
            "Bathroom Count": [Bathroom_count],
            "Sqm": [Sqm],
            "Latitude": [Latitude],
            "Longitude": [Longitude],
        }
    )


def predict_models(model, res):
    try:
        if isinstance(model, tuple):
            poly_model, poly_transformer = model
            res_poly = poly_transformer.transform(res)
            prediction = poly_model.predict(res_poly)
        else:
            prediction = model.predict(res)
        return str(int(prediction[0]))
    except Exception as e:
        st.error("Tahmin yaparken bir hata oluştu.")
        raise e


if __name__ == "__main__":
    main()
