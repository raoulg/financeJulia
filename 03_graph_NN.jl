using Graphs, GraphIO
using GraphNeuralNetworks
using FileIO
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Flux
using Statistics, Random

g = load("data/processed/tfinance.jld2")["tfinance"]

oh(x) = Float32.(onehotbatch(x, 0:1))
y = oh(g.ndata.y);
X = g.ndata.x;
ntrain = Int(floor(g.num_nodes * 0.8))
ntest = g.num_nodes - ntrain

train_mask = shuffle(Bool.(vcat(ones(ntrain), zeros(ntest))));
test_mask = map(!, train_mask);

ytrain = y[:, train_mask];
ytest = y[:, test_mask];
sum(ytrain, dims=2)
sum(ytest, dims=2)

classes = length(unique(g.ndata.y))
nin = size(X, 1)
nhidden = 128
nout = classes

model = GNNChain(GCNConv(nin => nhidden, relu),
                GCNConv(nhidden => nhidden, relu), 
                Dense(nhidden, nout))

ps = Flux.params(model)
opt = Adam(1f-3)

function eval_loss_accuracy(X, y, mask, model, g)
    ŷ = model(g, X)
    l = logitcrossentropy(ŷ[:,mask], y[:,mask])
    acc = mean(onecold(ŷ[:,mask]) .== onecold(y[:,mask]))
    return (loss = round(l, digits=4), acc = round(acc*100, digits=2))
end 


function report(epoch)
    train = eval_loss_accuracy(X, y, train_mask, model, g)
    test = eval_loss_accuracy(X, y, test_mask, model, g)        
    println("Epoch: $epoch   Train: $(train)   Test: $(test)")
end

@time for epoch in 1:10
    gs = Flux.gradient(ps) do
        ŷ = model(g, X)
        logitcrossentropy(ŷ[:,train_mask], ytrain)
    end

    Flux.Optimise.update!(opt, ps, gs)
    
    epoch % 1 == 0 && report(epoch)
end

using MLJ: ConfusionMatrix, f1score, precision, recall
using BSON: @save, @load
# @save "gnnmodel.bson" model

@load "gnnmodel.bson" model
@time ŷ = model(g, X)

labels = y[1, :]
prediction = ŷ[1, :] .> 10
f1score(prediction, labels)
precision(prediction, labels)
recall(prediction, labels)
ConfusionMatrix()(prediction, labels)


ytest = rand(1:10, 100)
yreal = rand(1:10, 100)
ConfusionMatrix()(ytest, yreal)
