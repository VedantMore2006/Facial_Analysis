# Mental Health Application

A comprehensive mental health platform that combines facial emotion analysis with a Django-based authentication and counselor-client matching system.

---

## 📁 Project Structure

```
Mental Health/
├── README.md
├── Dashboard/
├── image_analysis_of_face/
└── login_page/
```

---

## 📊 Directory Overview

### 1. **Dashboard/**

**Purpose:** Web-based dashboard interface for visualization and monitoring.

**Components:**
- `index.html` - Main dashboard interface for displaying mental health metrics and visualizations

**Focus:** Serves as the user-facing interface for monitoring and interacting with the application.

---

### 2. **image_analysis_of_face/**

**Purpose:** Real-time facial expression and emotion analysis using MediaPipe and Computer Vision.

This module captures video input, detects facial landmarks, extracts emotional features, normalizes them temporally, and logs the results for analysis.

#### **Directory Structure:**

```
image_analysis_of_face/
├── main.py                          # Entry point for facial analysis pipeline
├── test_mediapipe.py               # Testing/debugging MediaPipe face mesh
├── facial_signals.csv              # Output data log of extracted features
├── input/
│   └── video_source.py             # Video frame capture and sampling
├── processing/
│   ├── face_detection.py           # Face detection module (placeholder)
│   ├── feature_extraction.py       # Facial feature calculation
│   └── landmark_detection.py       # MediaPipe face mesh landmark detection
├── output/
│   ├── csv_logger.py               # CSV data logging functionality
│   └── feature_logger.py           # Feature logging (placeholder)
└── temporal/
    ├── baseline.py                 # Baseline normalization for feature scaling
    └── aggregation.py              # Temporal aggregation and windowing
```

#### **Key Modules:**

| Module | Purpose |
|--------|---------|
| **input/video_source.py** | Manages video capture from webcam or file source with FPS-based frame sampling |
| **processing/landmark_detection.py** | Uses MediaPipe Face Mesh to detect 468 facial landmarks on each frame |
| **processing/feature_extraction.py** | Extracts emotional features: Eye Aspect Ratio (EAR), mouth opening, jaw drop |
| **processing/face_detection.py** | Reserved for face detection functionality |
| **temporal/baseline.py** | Normalizes features relative to baseline values learned from warmup frames |
| **temporal/aggregation.py** | Aggregates normalized features over sliding windows with statistics (mean, variance, velocity) |
| **output/csv_logger.py** | Logs processed features to CSV with timestamp for analysis |
| **output/feature_logger.py** | Reserved for advanced logging functionality |

#### **Data Flow:**

```
Video Source → Landmark Detection → Feature Extraction → Baseline Normalization 
    → Temporal Aggregation → CSV Logger → facial_signals.csv
```

#### **Key Files:**

- **main.py** - Main pipeline orchestrating all modules with real-time visualization
- **test_mediapipe.py** - Standalone test for MediaPipe face mesh detection
- **facial_signals.csv** - Output containing temporal features with columns: timestamp, ear_norm, mouth_norm, jaw_norm, stress_flag, flat_affect_flag, arousal_flag

#### **Facial Features Extracted:**

- **Eye Aspect Ratio (EAR)** - Detects blink rate and eye closure
- **Mouth Opening Ratio** - Measures mouth opening/closing patterns
- **Jaw Drop** - Vertical jaw movement indicator
- **Normalized Values** - Baseline-adjusted features for comparison

---

### 3. **login_page/**

**Purpose:** Django-based authentication system with user management, counselor assignment, and MongoDB integration.

#### **Directory Structure:**

```
login_page/
└── authplayground/
    ├── manage.py                    # Django management script
    ├── db.sqlite3                   # SQLite database (if used)
    ├── authplayground/              # Django project configuration
    │   ├── __init__.py
    │   ├── asgi.py                 # ASGI configuration for async deployment
    │   ├── wsgi.py                 # WSGI configuration for production
    │   ├── settings.py             # Django settings and configuration
    │   └── urls.py                 # Main URL routing configuration
    └── users/                       # Django app for user management
        ├── models.py               # MongoDB document schemas (User, Counselor, Client)
        ├── views.py                # Request handlers for auth operations
        ├── urls.py                 # User app URL routes
        ├── admin.py                # Django admin customization
        ├── apps.py                 # App configuration
        ├── db.py                   # MongoDB connection initialization
        ├── tests.py                # Unit tests
        ├── migrations/             # Database migration history
        └── templates/
            └── users/
                ├── login.html          # Login page template
                ├── signup.html         # Registration page template
                ├── client_list.html    # Client list view
                ├── counselor_list.html # Counselor list view
                └── signup(figma)/      # Figma design mockups
```

#### **Key Modules:**

| Module | Purpose |
|--------|---------|
| **users/models.py** | MongoDB document schemas: User (client), Counselor, Client (with counselor reference) |
| **users/views.py** | Request handlers: signup, login, client/counselor list views, authentication logic |
| **users/db.py** | MongoDB connection configuration and initialization |
| **users/urls.py** | URL routing for user-related endpoints |
| **authplayground/settings.py** | Django configuration: installed apps, middleware, database settings |
| **authplayground/urls.py** | Main project URL routing |

#### **Database Models:**

1. **User** - Base user model with name, email, password_hash, created_at
2. **Counselor** - Counselor profile with name, specialization, email
3. **Client** - Client profile with name, email, counselor reference

#### **Authentication Flow:**

```
User Input → views.py (validate/hash) → MongoDB (User model) → Session Management
```

#### **Templates:**

- **login.html** - User login interface
- **signup.html** - User registration interface
- **client_list.html** - View all clients assigned to counselor
- **counselor_list.html** - View available counselors for assignment

---

## 🔗 Integration Between Components

1. **Authentication Layer** (`login_page/`) - Handles user registration and login
2. **Facial Analysis** (`image_analysis_of_face/`) - Analyzes client emotional states in real-time
3. **Dashboard** (`Dashboard/`) - Displays consolidated metrics and client information

**Data Flow:**
- Authenticated users via login system
- Real-time facial analysis captures emotions
- Results logged to CSV for dashboard display
- Counselors can monitor client emotional patterns

---

## 🚀 Key Technologies

- **Computer Vision:** MediaPipe Face Mesh, OpenCV
- **Backend:** Django, MongoDB
- **Frontend:** HTML/CSS, Figma designs
- **Data Processing:** NumPy, temporal aggregation
- **Logging:** CSV-based feature logging

---

## 📝 Usage

### Running Facial Analysis:
```bash
cd image_analysis_of_face
python main.py
```

### Running Authentication Server:
```bash
cd login_page/authplayground
python manage.py runserver
```

### Testing MediaPipe:
```bash
cd image_analysis_of_face
python test_mediapipe.py
```

---

## 📊 Output Formats

### facial_signals.csv
Columns: `timestamp`, `ear_norm`, `mouth_norm`, `jaw_norm`, `stress_flag`, `flat_affect_flag`, `arousal_flag`

Each row represents aggregated features over a time window, suitable for mental health analysis and counselor insights.

---

