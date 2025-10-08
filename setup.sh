#!/bin/bash

# Quick Setup Script for English-Tulu Dictionary

echo "🚀 English-Tulu Dictionary Setup"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "📦 Installing required packages..."
echo ""

# Install required packages
pip install flask python-dotenv google-generativeai simpletransformers transformers || {
    echo "❌ Failed to install packages. Please check your internet connection."
    exit 1
}

echo ""
echo "✅ Packages installed successfully!"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created!"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file and add your Gemini API key"
    echo "   Get your API key from: https://makersuite.google.com/app/apikey"
    echo ""
    echo "   Open .env file and replace 'your_gemini_api_key_here' with your actual key"
    echo ""
    read -p "Press Enter after you've added your API key to .env file..."
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application, run:"
echo "   python flask_app.py"
echo ""
echo "Then open your browser and go to:"
echo "   http://localhost:5000"
echo ""
echo "Happy translating! 📚✨"
