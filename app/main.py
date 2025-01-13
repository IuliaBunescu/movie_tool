import pandas as pd
import streamlit as st
from src.components.custom_html import CUSTOM_ALERT, CUSTOM_FORM

st.set_page_config(layout="wide")


def main():
    st.title("Movie Recommendation Tool for Data Scientists 🎬")

    col1, col2 = st.columns([3, 6])

    with col1:
        st.header("Choose your data source", divider="gray")

        st.markdown(
            "Choose how you want to input your data. You can choose one, or multiple input types. After you finished adding all the required data click the **Submit** button and wait move over to the **Results** section."
        )

        with st.form("input_form"):

            website_for_scraping = st.text_input(
                "Reference movie (RottenTomatoes or IMDB links) ",
                "https://www.rottentomatoes.com/m/the_lion_king_2019",
            )
            if website_for_scraping:
                st.success("Your website is valid.")
                st.write(CUSTOM_ALERT, unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Choose CSV/TSV files", accept_multiple_files=True
            )

            dataframes = []  # List to store the dataframes

            for uploaded_file in uploaded_files:
                # Determine the separator based on file extension
                file_extension = uploaded_file.name.split(".")[-1]
                if file_extension in ["csv", "tsv"]:
                    separator = "," if file_extension == "csv" else "\t"

                    # Read the file into a pandas dataframe
                    df = pd.read_csv(uploaded_file, sep=separator)
                    dataframes.append(df)

                    # Optionally display the dataframe in Streamlit
                    st.write(f"Dataframe from file: {uploaded_file.name}")
                    st.dataframe(df)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")

            data_submitted = st.form_submit_button("Submit", use_container_width=True)

        st.write(CUSTOM_FORM, unsafe_allow_html=True)
        if data_submitted:
            st.write("*Your input data was submitted!*")

    with col2:
        st.header("Results", divider="gray")

        tab1, tab2 = st.tabs(
            ["Exploratory Visualizations of Your Data", "Clustering & Recommendations"]
        )

        with tab1:
            if not data_submitted:
                st.write("No data submitted.")

        with tab2:
            if not data_submitted:
                st.write("No data submitted.")


if __name__ == "__main__":
    main()
