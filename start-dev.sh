#!/bin/bash

# Privacy Guardian Development Startup Script
# This script starts both the backend and frontend in development mode

echo "🛡️  Starting Privacy Guardian Development Environment"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down development servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "📦 Installing/updating dependencies..."

# Install backend dependencies
echo "🔧 Setting up backend..."
cd backend
if [ ! -d "venv" ]; then
    echo "   Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt >/dev/null 2>&1

echo "🚀 Starting backend server (uvicorn)..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

cd ..

# Install frontend dependencies
echo "🔧 Setting up frontend..."
cd frontend
npm install >/dev/null 2>&1

echo "🎨 Starting frontend development server (Vite)..."
npm run dev &
FRONTEND_PID=$!

cd ..

echo ""
echo "✅ Privacy Guardian is now running!"
echo "================================================"
echo "🌐 Frontend: http://localhost:5173"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "================================================"

# Wait for background processes
wait
