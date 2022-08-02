using Graphs, GraphIO
using SGtSNEpi
using Plots
using Reel
using GraphNeuralNetworks
using Graphs.LinAlg: adjacency_matrix


using CSV
using DataFrames
using FileIO

@time e = DataFrame(CSV.File("data/processed/edgelist.csv", header=false))
@time n = DataFrame(CSV.File("data/processed/nodes.csv", header=false))
@time y = DataFrame(CSV.File("data/processed/labels.csv", header=false))
e = Matrix(e)
n = Matrix(n)
y = Matrix(y)

y[:, 1]

e = e .+ 1
g = GNNGraph(e[:, 1], e[:, 2], ndata = (x=n', y = y[:, 1]))
save("data/processed/tfinance.jld2", Dict("tfinance" => g); compress=true)
@time g = load("data/processed/tfinance.jld2")

@time A = adjacency_matrix(g)
@time Y3 = sgtsnepi(A; d=3,  max_iter=500)

@time sc = scatter(Y3[:,1], Y3[:,2], Y3[:,3], 
    markersize = 1,
    alpha=0.1,
    size=(1000,1000),
    )
savefig(sc, "3d_finance.png")

fps = 30
duration = 10

rotate_fun(t::Float64, period::Integer) = 180*sin(t * 2pi / period)

function render(t, dt)
    scatter(Y3[:,1], Y3[:,2], Y3[:,3], 
    markersize = 1,
    alpha=0.1, 
    size=(1000,1000),
    legend=false,
	camera=(rotate_fun(t, fps * duration), rotate_fun(t, fps * duration)))
end

film = roll(render, fps=fps, duration=duration)
write("rotate.gif", film)
