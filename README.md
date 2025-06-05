# Violence Detection System

A real-time violence detection system using deep learning, consisting of a React Native mobile app and a Python Flask backend.

## Project Structure

```
.
├── mobil/                 # React Native mobile application
│   └── App.js            # Main application file
├── backend/              # Python Flask backend
│   ├── services/         # Business logic services
│   ├── config.py         # Configuration
│   ├── app.py           # Main Flask application
│   └── model_utils.py   # Model architecture
└── training/            # Model training scripts
    └── colab.py         # Training notebook
```

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables (optional):
```bash
export USE_CUDA=true  # For GPU support
export DEBUG=true     # For development mode
```

4. Run the server:
```bash
python app.py
```

### Mobile App Setup

1. Install dependencies:
```bash
cd mobil
npm install
```

2. Update the API URL in `src/config/config.js` to match your backend server.

3. Run the app:
```bash
npm start
```

## Features

- Real-time video analysis
- Violence detection using deep learning
- Modern and responsive UI
- Cross-platform support (iOS/Android)
- Configurable analysis parameters

## API Endpoints

- `POST /predict`: Analyze a video frame
- `GET /status`: Check API status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 