using GLMakie
GLMakie.inline!(true)
using MLDatasets: Cora, KarateClub, OGBDataset, MovieLens 
using Graphs.LinAlg: adjacency_matrix 
using Graphs
using SGtSNEpi
using DataFrames
# using ProgressMeter
using Colors
using GraphNeuralNetworks
using GraphIO

using FileIO

ENV["DATADEPS_ALWAYS_ACCEPT"] = false

@time dataset = Cora()
@time g = mldataset2gnngraph(dataset)
A = adjacency_matrix(g)

g.ndata.features

save("cora.jld2", Dict("cora" => g))

@time load("cora.jld2")



hasproperty(dataset, :graphs)

dataset.graphs.node_data

dataset.metadata
dataset.graphs
# dataset = KarateClub()
mlg = dataset[1]
typeof(mlg)

labels = mlg.node_data.labels_clubs
# labels = mlg.node_data["user"][:gender]

labels = mlg.node_data.targets
labels = mlg.node_data.label

function ml2g(mlg)
    s, t = mlg.edge_index
    g = Graphs.DiGraph(mlg.num_nodes)
    @showprogress for (i, j) in zip(s, t)
        Graphs.add_edge!(g, i, j)
    end
    g
end

g = ml2g(mlg)


A = adjacency_matrix(g)
d = 2

Y0 = 0.01 * randn( size(A,1), d);
Y = sgtsnepi(A; d = d, Y0 = Y0, max_iter=500);
show_embedding(Y, labels; A = A)


cmap = distinguishable_colors(
           maximum(labels) - minimum(labels) + 1,
           [RGB(1,1,1), RGB(0,0,0)], dropseed=true);
typeof(cmap)

sc = scatter( Y[:,1], Y[:,2], Y[:,3], color = labels, colormap = cmap, markersize = 2 )


L = mlg.node_data.labels_clubs
cmap = distinguishable_colors(
           maximum(L) - minimum(L) + 1,
           [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

Y0 = 0.01 * randn( size(A,1), 3 );
Y = sgtsnepi(A; d = 3, Y0 = Y0, max_iter = 500);
sc = scatter( Y[:,1], Y[:,2], Y[:,3], color = L, colormap = cmap, markersize = 2 )

record(sc, "sgtsnepi-animation.gif", range(0, 1, length = 24*8); framerate = 24) do ang
  rotate_cam!( sc.figure.scene.children[1], 2*π/(24*8), 0, 0 )
end