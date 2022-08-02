using Graphs, GraphIO
using SGtSNEpi
using Plots
using Reel
using GraphNeuralNetworks
using Graphs.LinAlg: adjacency_matrix
using FileIO

g = load("data/processed/tfinance.jld2")["tfinance"]
A = adjacency_matrix(g)

# takes about ~190 seconds
@time Y3 = sgtsnepi(A; d=3,  max_iter=500)

labels = map(Int, g.ndata.y)
(sum(labels) / length(labels)) * 100

selection = map(Bool, labels)

cmap = palette([:red, :lightgrey], 2);
sc = scatter(Y3[:,1], Y3[:,2], Y3[:,3], 
    markersize = 0.5,
    color=labels,
    palette=cmap,
    alpha=0.03,
    size=(1000,1000),
    )

y = Y3[selection, :]
scatter!(y[:,1], y[:,2], y[:,3], 
    markersize = 1,
    alpha=1,
    color="red",
    markerstrokewidth=0,
    size=(1000,1000),
    )


savefig(sc, "img/3d_finance.png")

fps = 30
duration = 10
cmap = palette([:red, :lightgrey], 2);
rotate_fun(t::Float64, period::Integer) = 180*sin(t * 2pi / period)
function render(t, dt)
    scatter(Y3[:,1], Y3[:,2], Y3[:,3], 
    markersize = 1,
    alpha=0.1, 
    color=labels,
    palette=cmap,
    size=(1000,1000),
    legend=false,
	camera=(rotate_fun(t, fps * duration), rotate_fun(t, fps * duration)))
end

film = roll(render, fps=fps, duration=duration)
write("img/rotate.gif", film)
