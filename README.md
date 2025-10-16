# 🎬 Movie Recommendation System

An intelligent movie recommendation system using collaborative filtering with 4 ML models, TMDb API integration, and interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![TMDb](https://img.shields.io/badge/TMDb-API-green.svg)

---

## ✨ Key Features

- 🎯 **Personalized Recommendations** - Based on user preferences and movie similarities
- 🎥 **Unlimited Movie Search** - Search ANY movie via TMDb API
- 🖼️ **Real Movie Posters** - High-quality posters from TMDb
- 📊 **4 ML Models** - User-based CF, Item-based CF, SVD, Hybrid
- 📈 **Similarity Scores** - See matching percentages for recommendations
- 🎨 **Interactive Dashboard** - Beautiful visualizations and insights

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Setup

1. **Download Dataset**: [MovieLens 100K](http://files.grouplens.org/datasets/movielens/ml-100k.zip)

   - Extract `u.data` and `u.item` to project folder

2. **Run Data Exploration**

   ```bash
   python 1_data_exploration.py
   ```

3. **Train Models**

   ```bash
   python 2_model_training.py
   ```

4. **Launch App**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📂 Project Structure

```
movie-recommendation-system/
│
├── u.data                          # Raw ratings data
├── u.item                          # Movie metadata
├── 1_data_exploration.py           # EDA script
├── 2_model_training.py             # Model training
├── streamlit_app.py                # Web application
├── recommendation_models.pkl       # Trained models
├── model_comparison.csv            # Performance metrics
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 📊 Dataset

**MovieLens 100K**

- 943 users
- 1,682 movies
- 100,000 ratings
- Scale: 1-5 stars
- 19 genres

---

## 🤖 Models

1. **User-Based CF** - Recommends based on similar users
2. **Item-Based CF** - Recommends based on similar movies ⭐ Best
3. **SVD (Matrix Factorization)** - Latent factor decomposition
4. **Hybrid** - Combines Item-Based + SVD

---

## 📖 How to Use

### Mode 1: User-Based Recommendations

1. Select user ID (1-943)
2. Choose number of recommendations
3. Toggle movie posters
4. View recommendations with similarity scores

### Mode 2: Movie-Based Recommendations

1. Search ANY movie (e.g., "Avatar", "Inception")
2. Select from:
   - **Our Database** (1,600 movies) - Best recommendations
   - **TMDb** (Unlimited movies) - Any movie ever made
3. View similar movies with posters and ratings

---

## 🎨 App Features

### Pages

1. **Home** - Overview and statistics
2. **Get Recommendations** - Two recommendation modes
3. **Data Insights** - Interactive visualizations
4. **Model Performance** - Comparison and metrics

### Highlights

- Real movie posters via TMDb API
- Similarity scores (0-100%)
- Genre and rating information
- Interactive charts and graphs
- Responsive design

---

## 🔧 Technical Details

### Technologies

- **Python** - Core language
- **Pandas/NumPy** - Data processing
- **Scikit-learn** - ML algorithms
- **Streamlit** - Web framework
- **Plotly** - Visualizations
- **TMDb API** - Movie data

### Model Performance

| Model             | Precision@10 | Coverage |
| ----------------- | ------------ | -------- |
| User-Based CF     | ~0.025       | ~0.68    |
| **Item-Based CF** | **~0.031**   | ~0.75    |
| SVD               | ~0.029       | ~0.81    |
| Hybrid            | ~0.033       | ~0.78    |

---

## 📦 Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.25.0
plotly>=5.14.0
requests>=2.28.0
pillow>=9.0.0
```

---

## 🐛 Troubleshooting

**Issue: Model files not found**

- Run `2_model_training.py` first

**Issue: No movies found in search**

- Check spelling or try popular movies

**Issue: Posters not loading**

- Check internet connection
- Placeholders will show automatically

---

## 🎓 Learning Outcomes

- ✅ Collaborative filtering implementation
- ✅ Multiple ML model comparison
- ✅ API integration (TMDb)
- ✅ Web application development
- ✅ Data visualization
- ✅ End-to-end ML pipeline

---

## 📝 License

MovieLens 100K dataset for educational/research use.

---

## 🙏 Credits

- **GroupLens Research** - MovieLens dataset
- **The Movie Database (TMDb)** - Movie data and posters
- **Streamlit** - Web framework

---

**Built for ML Internship Journey**

**Happy Recommending! 🎬✨**
