# mineralbot

an intelligent chatbot for mineral identification and knowledge exploration, combining natural language processing, computer vision, and knowledge reasoning systems.

## bits.

- **multimodal interaction**: Voice and text input with speech synthesis output
- **q&a System**: Natural language question answering using TF-IDF and cosine similarity
- **image Classification**: CNN-based mineral identification from photographs
- **fuzzy logic**: Graded truth values for mineral properties
- **formal logic system**: Theorem proving for verifying mineral characteristics
- **knowledge base management**: User-editable knowledge bases with CSV storage
- **aiml conversational interface**: Natural language understanding and intent routing

## Tech Stack

- **Keras Tuner** - Hyperparameter optimization
- **scikit-learn** - Machine learning and NLP (TF-IDF, cosine similarity)
- **NLTK** - Natural language processing and logical inference
- **SpeechRecognition** - Voice input via Google Web Speech API
- **pyttsx3** - Text-to-speech output
- **pandas/numpy** - Data processing

## Installation

### prerequisites

```bash
pip install tensorflow keras keras-tuner scikit-learn nltk pandas numpy
pip install python-aiml SpeechRecognition pyttsx3
```

### Running the Chatbot

```bash
python mineral-bot.py
```

### Interaction Modes

**Text Input**: Type your question directly
```
query: what is quartz?
```

**Voice Input**: Choose voice mode at the prompt
```
enter 'voice' to use voice input or press any key to type your query: voice
```

### Commands

- **Q&A Query**: Ask factual questions about minerals
  ```
  what is the hardness of quartz?
  ```

- **Image Classification**: Identify minerals from photos
  ```
  identify this mineral
  > [provide path to image file]
  ```

- **Fuzzy Knowledge Assertion**: Add facts with graded truth
  ```
  I know that quartz is hard
  ```

- **Fuzzy/Logical Verification**: Check facts
  ```
  check that biotite is soft
  ```

- **Exit**: Type `exit` or `quit`

## Model Training

To train/tune the CNN model:

```bash
python cnn_tuned.py
```

The hyperparameter tuning process will:
- Search through filter sizes, kernel sizes, dense units, dropout rates, and learning rates
- Perform 20 random search trials
- Save the best model as `mineral_cnn_tuned.h5`

### Dataset Structure

```
minet/
├── biotite/
├── bornite/
├── chrysocolla/
├── malachite/
├── muscovite/
├── pyrite/
└── quartz/
```

## Architecture

### Knowledge Systems

1. **Q&A Subsystem**: Vectorizes questions with TF-IDF and retrieves best matches via cosine similarity
2. **Fuzzy Logic System**: Maintains graded truth values (0.0-1.0) for mineral properties
3. **Logical Reasoning**: NLTK-based resolution theorem prover for formal verification
4. **AIML Engine**: Routes user input to appropriate subsystems based on patterns

### Image Classification Pipeline

1. Load and preprocess image (64x64 RGB)
2. Normalize pixel values (0-1 range)
3. Feed through trained CNN
4. Return predicted mineral class

### CNN Architecture

- Conv2D layers with tunable filters and kernels
- MaxPooling for spatial downsampling
- Dense layers with dropout regularization
- Softmax output for 7 mineral classes
- Optimized via random hyperparameter search

## issues.

- **Small Dataset**: Mitigated overfitting through dropout, data augmentation, and conservative model capacity
- **Class Imbalance**: Used targeted augmentation and validation splitting
- **Visual Similarity**: Applied hyperparameter tuning to optimize feature discrimination
- **Multi-System Integration**: Modular design with clear routing through AIML patterns
