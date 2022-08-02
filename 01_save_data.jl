using CSV
using DataFrames
using FileIO
using GraphNeuralNetworks

# load the data
@time e = DataFrame(CSV.File("data/processed/edgelist.csv", header=false))
@time n = DataFrame(CSV.File("data/processed/nodes.csv", header=false))
@time y = DataFrame(CSV.File("data/processed/labels.csv", header=false))

e = Matrix(e)
n = Matrix(n)
y = Matrix(y)

# edges start at 1
e = e .+ 1

g = GNNGraph(e[:, 1], e[:, 2], ndata = (x=n', y = y[:, 1]))
save("data/processed/tfinance.jld2", Dict("tfinance" => g); compress=true)