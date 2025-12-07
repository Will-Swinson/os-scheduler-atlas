# OS Scheduler Atlas 🧮

A machine learning-powered operating system scheduling simulator that predicts optimal scheduling algorithms and provides detailed performance analysis.

## 🚀 Features

- **Smart Algorithm Prediction**: ML model predicts the best scheduling algorithm (FCFS, SJF, Round Robin) for your workload
- **Multiple Scheduling Algorithms**: First-Come-First-Served (FCFS), Shortest Job First (SJF), Round Robin (RR)
- **Performance Analytics**: Detailed metrics including waiting time, turnaround time, and CPU utilization
- **Database Integration**: Persistent storage of workloads, simulations, and predictions for analysis
- **Flexible API**: Support for both direct simulation and prediction-based workflows
- **Production Ready**: FastAPI backend with SQLAlchemy ORM and Alembic migrations

## 🏗️ Architecture

```
├── backend/
│   ├── src/
│   │   ├── api/           # FastAPI endpoints and models
│   │   ├── database/      # SQLAlchemy models and queries
│   │   ├── ml/            # Machine learning pipeline
│   │   └── services/      # Business logic and utilities
│   ├── alembic/           # Database migrations
│   └── pybind_module/     # C++ scheduling algorithms (Python bindings)
├── frontend/              # React frontend (coming soon)
└── docs/                  # Documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip or conda
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Will-Swinson/os-scheduler-atlas.git
   cd os-scheduler-atlas
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your database settings if needed
   ```

5. **Run database migrations**
   ```bash
   cd backend
   alembic upgrade head
   ```

6. **Start the API server**
   ```bash
   uvicorn src.api.main:app --reload
   ```

7. **Access the application**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## 📖 API Usage

### 1. Get Algorithm Prediction

Predict the optimal scheduling algorithm for your workload:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "processes": [
      {"pid": 1, "arrival_time": 0, "burst_time": 10},
      {"pid": 2, "arrival_time": 1, "burst_time": 5},
      {"pid": 3, "arrival_time": 2, "burst_time": 8}
    ]
  }'
```

**Response:**
```json
{
  "prediction_id": 1,
  "predicted_algorithm": "SJF",
  "model_confidence": 0.87,
  "features_used": {
    "avg_burst_time": 7.67,
    "burst_time_variance": 6.22,
    "arrival_time_spread": 2
  }
}
```

### 2. Run Simulation (Direct)

Simulate scheduling with a specific algorithm:

```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "processes": [
      {"pid": 1, "arrival_time": 0, "burst_time": 10},
      {"pid": 2, "arrival_time": 1, "burst_time": 5}
    ],
    "algorithm": "RR",
    "time_quantum": 2
  }'
```

### 3. Run Simulation (Using Prediction)

Use a previous prediction to run simulation:

```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": 1
  }'
```

**Response:**
```json
{
  "simulation_id": 1,
  "algorithm_used": "SJF",
  "results": {
    "processes": [
      {
        "pid": 1,
        "arrival_time": 0,
        "burst_time": 10,
        "waiting_time": 6,
        "turnaround_time": 16,
        "completion_time": 16
      }
    ]
  },
  "performance_metrics": {
    "avg_waiting_time": 4.33,
    "avg_turnaround_time": 11.67
  }
}
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Database (SQLite for development)
SQLALCHEMY_DATABASE_URL="sqlite:///./testdb.db"

# Database (PostgreSQL for production)
# SQLALCHEMY_DATABASE_URL="postgresql://user:password@localhost:5432/os_scheduler_db"

# API Configuration
# API_HOST="localhost"
# API_PORT=8000

# ML Model
# MODEL_PATH="./trained_model.pkl"
```

### Database Setup

**SQLite (Development - Default):**
- Automatically creates `testdb.db` file
- No additional setup required

**PostgreSQL (Production):**
1. Install PostgreSQL
2. Create database: `createdb os_scheduler_db`
3. Update `SQLALCHEMY_DATABASE_URL` in `.env`
4. Run migrations: `alembic upgrade head`

## 🧪 Testing

### Test the ML Pipeline

```bash
cd backend/src/ml
python test.py
```

### Test API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Sample Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "processes": [
      {"pid": 1, "arrival_time": 0, "burst_time": 5},
      {"pid": 2, "arrival_time": 2, "burst_time": 3},
      {"pid": 3, "arrival_time": 4, "burst_time": 8}
    ]
  }'
```

## 📊 Understanding the Results

### Scheduling Algorithms

- **FCFS (First-Come-First-Served)**: Processes are executed in arrival order
- **SJF (Shortest Job First)**: Shortest burst time processes execute first
- **RR (Round Robin)**: Each process gets a fixed time quantum in rotation

### Performance Metrics

- **Waiting Time**: Time a process spends waiting in the ready queue
- **Turnaround Time**: Total time from arrival to completion
- **Response Time**: Time from arrival to first execution
- **CPU Utilization**: Percentage of time CPU is busy

### When Each Algorithm Works Best

- **FCFS**: Simple, predictable workloads with similar burst times
- **SJF**: Mixed workloads with many short tasks
- **Round Robin**: Interactive systems requiring fairness and responsiveness

## 🗃️ Database Schema

```sql
-- Workloads: Groups of processes
workloads(id, created_at)

-- Individual processes in a workload
processes(id, arrival_time, burst_time, workload_id, created_at)

-- ML predictions for workloads
predictions(id, workload_id, predicted_algorithm, model_confidence, created_at)

-- Simulation results
simulations(id, workload_id, algorithm_used, avg_waiting_time, avg_turnaround_time, created_at)
```

## 🚢 Deployment

### Docker (Coming Soon)

```bash
docker-compose up
```

### Manual Production Deployment

1. **Set up PostgreSQL database**
2. **Configure environment variables**
3. **Run migrations**: `alembic upgrade head`
4. **Start with production server**:
   ```bash
   gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Workflow

```bash
# Database migrations
alembic revision --autogenerate -m "Description of changes"
alembic upgrade head

# Running in development
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔍 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Ensure you're in the backend directory
cd backend
uvicorn src.api.main:app
```

**Database connection errors:**
```bash
# Check your .env file configuration
# Ensure database exists (for PostgreSQL)
# Run migrations: alembic upgrade head
```

**ML model not found:**
```bash
# Train a new model
cd backend/src/ml
python test.py
```

**Negative waiting times:**
```bash
# This is a known issue being investigated
# Check GitHub issues for updates
```

## 🗺️ Roadmap

- [ ] React frontend interface
- [ ] Docker containerization
- [ ] Advanced performance metrics
- [ ] Real-time visualization
- [ ] Multi-level feedback queue algorithm
- [ ] Process priority support
- [ ] Batch job scheduling
- [ ] Performance benchmarking suite

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Database management with [SQLAlchemy](https://www.sqlalchemy.org/)
- C++ performance optimizations with [pybind11](https://pybind11.readthedocs.io/)

## 📞 Support

- 📫 Email: [will.swinson@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/Will-Swinson/os-scheduler-atlas/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Will-Swinson/os-scheduler-atlas/discussions)

---

**Built with ❤️ for operating systems education and research**
