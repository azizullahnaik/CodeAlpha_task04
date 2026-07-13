# StreetScan AI — Intelligent Road Damage Detection

## Overview
StreetScan AI is an advanced AI-powered system for real-time detection and analysis of road infrastructure damage. Using state-of-the-art YOLOv11 computer vision technology, it automatically identifies potholes, cracks, and manholes in road images with high precision.

## Features

- **AI-Powered Detection**: Real-time identification of Potholes, Cracks, and Manholes using custom-trained YOLOv11 models
- **Dynamic Circular Interface**: Innovative circular upload zone with animated visual feedback
- **Cream Aesthetic Design**: Warm, elegant cream color scheme with glassmorphic elements
- **Extensive Animations**: Rich interactive animations including rotation, floating, pulsing, and morphing effects
- **Orb-Shaped Statistics**: Beautiful spherical stat displays with animated borders
- **Road Safety Index**: Intelligent safety scoring with severity classification (LOW/MODERATE/HIGH)
- **Memory-Safe Processing**: Automatic cache clearing and memory management for multiple uploads
- **Serverless Architecture**: Optimized for Vercel deployment with efficient resource usage

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

**Requirements:**
- `best.pt` model file in project root
- `vercel.json` configured for serverless routing
- `requirements.txt` with optimized packages

## API Endpoints

- `GET /` — Main dashboard interface with animated UI
- `POST /api/detect` — Image analysis with YOLOv11 inference
- `GET /api/health` — System health check

## Tech Stack

- **Backend**: Flask, YOLOv11, OpenCV-headless, NumPy
- **Frontend**: TailwindCSS, Vanilla JavaScript, Custom CSS Animations
- **Deployment**: Vercel Serverless Functions
- **Design**: Glassmorphism, Circular UI, Animated Components

## UI Features

- **Circular Upload Zone**: 350px animated circular drop zone with spinning borders
- **Animated Statistics**: Four orb-shaped stat cards with rotating outer rings
- **Dynamic Background**: Floating decorative circles with various animations
- **Interactive Elements**: Hover effects, scale animations, and smooth transitions
- **Responsive Design**: Optimized for all screen sizes

---

Engineered by Azizullah Naik
