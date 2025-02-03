# Movie Recommendation Tool

Welcome to the Movie Recommendation Tool! This application helps users discover movies tailored to their preferences.

## Live Demo

Experience the app in action: [Movie Recommendation Tool](https://juliab-movie-recommendation-tool.streamlit.app/)

## Features

- **Personalized Recommendations**: Get movie suggestions based on your unique interests.
- **User-Friendly Interface**: Enjoy a seamless and intuitive browsing experience.
- **Real-Time Search**: Instantly find movies with an efficient search feature.
- **Insightful Visualizations**: Gain a deeper understanding of the training data and clustering process through engaging visual representations.

## Installation

To run the application locally:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/IuliaBunescu/movie_tool.git
   cd movie_tool/app
   ```

2. **Install Dependencies**:

   Ensure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Application**:

   Start the Streamlit app:

   ```bash
   streamlit run app/main.py
   ```

   Access the app at `http://localhost:8501` in your browser.

## Usage

- **Search for Movies**: Enter a movie title to get relevant details and recommendations.
- **Explore Suggestions**: Browse through curated movie recommendations based on your preferences.
- **Dynamic Data Sourcing**: The app dynamically retrieves the training dataset from the TMDB API based on the user's input, a process that may take up to 3 minutes.
- **Training Process**: The clustering algorithm then takes approximately 2 more minutes to complete its training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
