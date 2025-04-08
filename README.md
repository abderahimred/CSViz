# CSViz 📊

A streamlined Streamlit application for uploading, analyzing, processing, cleaning, and visualizing your CSV data.

## ✨ Features

- ⬆️ **Upload**: Import single or multiple CSV files effortlessly.
- 🧐 **Analyze**: Get data quality insights for a single uploaded CSV.
- ⚙️ **Process**: Combine or merge multiple CSV files based on your needs.
- 🧹 **Clean**: Fix data issues like duplicates, missing values, and formatting inconsistencies (optionally using AI).
- 📈 **Visualize**: Create interactive charts and graphs to explore trends and correlations.
- 💾 **Export**: Save your cleaned data and visualizations.

## 🚀 Installation

```bash
# 1. Clone the repository (if you haven't already)
# git clone <your-repo-url>
# cd <your-repo-directory>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up OpenAI API key (Optional, for AI cleaning features)
# Create a .env file in the project root:
echo "OPENAI_API_KEY=your_api_key_here" > .env

# 4. Launch the application
streamlit run Home.py
```

## 📖 Usage Instructions

1.  **⬆️ Upload CSV Files**
    *   Drag and drop your `.csv` file(s) into the upload area on the Home page.
    *   See a quick summary of each uploaded file (name, size, rows/columns).
2.  **🧐 Analyze Single File** (if only *one* file is uploaded)
    *   Click the "📊 Analyze Data Quality" button.
    *   Review insights on data completeness, uniqueness, and potential issues.
3.  **⚙️ Process Multiple Files** (if *more than one* file is uploaded)
    *   Click the "🔗 Process Multiple Files" button.
    *   Choose a method to combine your files (e.g., append, merge).
4.  **🧹 Clean Data**
    *   Navigate through the cleaning steps.
    *   Apply transformations like removing duplicates, handling missing values, correcting data types, etc.
    *   Utilize AI-powered cleaning if your OpenAI API key is configured.
5.  **📈 Visualize Data**
    *   Select columns and chart types to generate visualizations.
    *   Explore your cleaned data visually.
    *   Export charts or the final cleaned dataset.

## 🔑 API Key Configuration (Optional)

CSViz can leverage OpenAI for advanced data cleaning features. To enable this, provide your API key using one of these methods:

1.  **`.env` File (Recommended)**: Create a file named `.env` in the project's root directory and add:
    ```
    OPENAI_API_KEY=your_api_key_here
    ```
2.  **In-App Input**: Enter the key directly in the application when prompted (less secure for regular use).

🔒 *For security, ensure your `.env` file is included in your `.gitignore` and never committed to version control.* 

## 🛠️ Requirements

- Python 3.8+
- Streamlit 1.31.0+
- Pandas 2.0.0+
- NumPy 1.24.0+
- Plotly 5.14.0+
- python-dotenv (for `.env` file loading)

## 📄 License

MIT License 