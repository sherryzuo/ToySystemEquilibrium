#!/usr/bin/env julia

"""
test_model_reuse_performance.jl

Simple performance test to demonstrate the benefits of model reuse with warm starts.
Compares the original vs optimized DLAC-i and SLAC implementations.

Usage:
    julia test_model_reuse_performance.jl
"""

using Revise
using ToySystemQuad

function main()
    """Run comprehensive performance tests using TestRunner."""
    
    println("🚀 MODEL REUSE AND WARM START PERFORMANCE TEST")
    println("Following the Python Gurobi pattern for Julia JuMP optimization")
    println()
    println("This test compares:")
    println("• Original: Creates new models for each rolling horizon iteration")  
    println("• Optimized: Reuses models with warm starts and constraint RHS updates")
    println()
    
    # Use the TestRunner performance test function
    results = run_performance_test_system(output_dir="performance_test_results")
    
    return results
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end