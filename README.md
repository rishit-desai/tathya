# Tathya - Media Bias Analysis Platform

Tathya is an AI-powered platform that analyzes news articles and media content for various linguistic and rhetorical features to help identify potential biases and provide deeper insights into media reporting. The system combines multi-task learning with Analysis-Augmented Generation (AAG) to provide comprehensive media analysis.

## Scientific Approach

### Multi-Task Learning Architecture

The system employs three specialized multi-task BERT models, each trained to perform multiple related tasks simultaneously. This approach offers several advantages:

1. **Shared Representation Learning**
   - Models learn common features across related tasks
   - Reduces parameter count compared to single-task models
   - Improves generalization through transfer learning
   - Enables better feature extraction for related tasks

2. **Task-Specific Heads**
   - Each model uses a shared BERT encoder with task-specific classification heads
   - Allows for specialized learning while maintaining common understanding
   - Enables efficient multi-label classification at the token level

3. **Correlation Exploitation**
   - Models leverage correlations between related tasks
   - Example: Political stance often correlates with group appeals
   - Sentiment analysis benefits from hyperbole detection
   - Readability analysis is enhanced by vagueness detection

### Analysis-Augmented Generation (AAG)

The system uses a two-stage analysis pipeline:

1. **Structured Analysis Stage**
   - Multi-task models provide detailed token-level annotations
   - Each token is classified across multiple dimensions
   - Results are structured as JSON for downstream processing

2. **Generative Explanation Stage**
   - Google's Gemini model receives structured analysis
   - Generates human-readable explanations
   - Combines multiple analysis dimensions into coherent insights
   - Provides context-aware interpretations of the findings

## Model Architecture

### 1. Political Analysis Model (group_1_model)
- Stance detection (3 labels: left/center/right)
- Group appeals detection (binary)
- Endorsement analysis (binary)
- Shared BERT encoder with task-specific classification heads

### 2. Sentiment Analysis Model (group_2_model)
- Hyperbole detection (binary)
- Sentiment classification (6 categories)
- Leverages emotional intensity for better sentiment analysis

### 3. Readability Model (group_3_model)
- Vagueness detection (binary)
- Readability scoring (6 levels)
- Uses shared linguistic features for both tasks

## Training Details

### Model Parameters
- Base Model: BERT-base-uncased
- Max Sequence Length: 512 tokens
- Batch Size: 10
- Learning Rate: 2e-5
- Epochs: 5
- Optimizer: AdamW with linear warmup
- Loss Function: Cross-entropy for each task

### Training Process
1. **Data Preparation**
   - Token-level annotation
   - Multi-label encoding
   - Train/validation split

2. **Training Strategy**
   - Joint training of all tasks
   - Task-specific loss weighting
   - Gradient accumulation for effective batch size

3. **Evaluation Metrics**
   - Per-task accuracy
   - Confusion matrices
   - Correlation analysis between tasks

## Analysis Pipeline

1. **Input Processing**
   - Text preprocessing and tokenization
   - Multi-model parallel analysis
   - Structured output generation

2. **Multi-Model Analysis**
   - Token-level classification
   - Structured JSON output
   - Feature correlation analysis

3. **AAG Generation**
   - Structured analysis → Gemini input
   - Context-aware explanation generation
   - Multi-dimensional insight synthesis

## Technical Stack

- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Models**: BERT-based architectures
- **Generation**: Google Gemini AI
- **Backend**: Flask API
- **Frontend**: React + TypeScript
- **Visualization**: Matplotlib, Seaborn

## Performance Considerations

- Model quantization for inference
- Batch processing for efficiency
- Caching of model outputs
- Stream processing for long texts

## Future Improvements

1. **Model Enhancements**
   - Larger pre-trained models
   - Task-specific fine-tuning
   - Advanced multi-task learning techniques

2. **Analysis Pipeline**
   - Real-time analysis
   - Incremental processing
   - Enhanced AAG capabilities

3. **Feature Additions**
   - More linguistic dimensions
   - Cross-article analysis
   - Temporal bias tracking

## Development Setup

### Training Environment Setup

1. **Create Project Structure** - Ensure the following directory structure:
   ```
   tathya/
   ├── dataset_creator.ipynb
   ├── data/
   │   └── article_000001.txt
   │   └── article_000002.txt
   │   └── ...
   ├── dataset/   (This directory will be created by the script for output)
   │   └── article_000000.json
   │   └── ...
   └── models/         
   │── group_1_model.pth
   │── group_2_model.pth
   │── group_3_model.pth
   |
   │── group_1_training.ipynb
   │── group_2_training.ipynb
   │── group_3_training.ipynb
   ├── web/
   │   ├── api/
   │   │   ├── backend.py
   │   │   └── utils.py
   │   └── ui/
   │       └── src/
   └── requirements.txt
   ```

2. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Training Process**
   - Run notebooks in sequence: dataset_creator.ipynb → group_1_training.ipynb → group_2_training.ipynb → group_3_training.ipynb
   - Each notebook will save its model to the root directory
   - Training logs and metrics are saved in each notebook
   - Expected training time: ~10-12 hours per model on GPU

4. **Required Dependencies**
   ```
   torch>=2.0.0
   transformers>=4.30.0
   pandas>=1.5.0
   numpy>=1.24.0
   scikit-learn>=1.0.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   jupyter>=1.0.0
   tqdm>=4.65.0
   ```

### Frontend Deployment

1. **Prerequisites**
   - Node.js 16+ and npm
   - Python 3.8+
   - Pre-trained model artifacts in root directory

2. **Backend Setup** - Flask and other dependencies should be installed in the same virtual environment.
   ```bash
   cd web/api   
   # Start Flask server
   python backend.py
   ```

3. **Frontend Setup**
   ```bash
   cd web/ui
   npm install
   # Start development server
   npm run dev
   ```

4. **Required Model Artifacts**
   - Place all three model files (generated in training) in project root:
   ```
   group_1_model.pth
   group_2_model.pth
   group_3_model.pth
   ```
   - Login to gcloud CLI to access Google Gemini API
   ```bash
   pip install --upgrade google-genai
   gcloud auth application-default login
   ```


5. **Verification**
   - Backend should be running on http://localhost:5000
   - Frontend should be running on http://localhost:3000
   - Test with sample text input
   - Check browser console for any errors

6. **Troubleshooting**
   - Ensure all model files are in correct location
   - Verify your Vertex AI SDK is configured correctly (project, region)
   - Check CORS settings if frontend can't connect to backend
   - Monitor Flask logs for any model loading errors
