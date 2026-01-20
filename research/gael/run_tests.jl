using ProgressMeter

# Include the modules
# We assume this script is run from the same directory, or we use relative paths
# if run from project root, we might need to adjust.
# Best to rely on @__DIR__

include(joinpath(@__DIR__, "run_drift.jl"))
include(joinpath(@__DIR__, "run_scale.jl"))

using .RunDrift
using .RunScale

# Configuration
const NUM_TESTS = 5
const OUTPUT_FILE = joinpath(@__DIR__, "output.csv")

# Choose which module to run: RunDrift or RunScale
# You can change this to RunScale to switch the experiment
const TARGET_MODULE = RunDrift

const ALGORITHMS = [:PMMH, :PGIBBS, :EHMM]

function main()
    println("Running benchmarks for module: $TARGET_MODULE")
    println("Output file: $OUTPUT_FILE")
    
    open(OUTPUT_FILE, "w") do io
        # Write header
        write(io, "seed,ground_truth_posterior,algorithm,squared_error,effective_sample_size,runtime\n")
        
        @showprogress "Benchmarking..." for seed in 1:NUM_TESTS
            for algo in ALGORITHMS
                try
                    # Run experiment
                    # run_experiment returns: (gt_mean, est_mean, ess_val, elapsed_time)
                    gt, est, ess_val, time = TARGET_MODULE.run_experiment(seed, algo)
                    
                    sq_err = (gt - est)^2
                    
                    # Write row
                    write(io, "$seed,$gt,$(string(algo)),$sq_err,$ess_val,$time\n")
                    flush(io)
                catch e
                    println("\nError running seed=$seed, algo=$algo: $e")
                    rethrow(e)
                end
            end
        end
    end
    println("Results written to $OUTPUT_FILE")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
