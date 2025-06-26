#!/bin/bash

# Setup script to initialize ToySystemQuad as a standalone GitHub repository

echo "Setting up ToySystemQuad as standalone GitHub repository..."

# Initialize git repository
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Complete ToySystemQuad modular implementation

- Modular architecture with separate optimization models
- Capacity Expansion Model (CEM) with joint investment/operations
- Perfect Foresight Operations (DLAC-p) 
- DLAC-i Operations with rolling horizon MPC
- Comprehensive plotting and analysis modules
- Full CSV export functionality
- PMR calculation and convergence diagnostics
- 4-technology system: Nuclear, Wind, Gas, Battery"

echo ""
echo "Repository initialized locally!"
echo ""
echo "To push to GitHub:"
echo "1. Create a new repository on GitHub named 'ToySystemQuad'"
echo "2. Run these commands:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/ToySystemQuad.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Repository contents:"
ls -la

echo ""
echo "Key files for standalone operation:"
echo "- README.md: Complete documentation"
echo "- Project.toml: Julia package dependencies"
echo "- run_complete_test.jl: Main execution script"
echo "- All .jl modules are self-contained"