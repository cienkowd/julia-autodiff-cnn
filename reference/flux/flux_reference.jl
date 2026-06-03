# reference/flux/flux_reference.jl
#
# Rozwiazanie referencyjne we Flux.jl — to przykladowa implementacja, ktora definiuje
# architekture sieci, algorytm uczenia i hiperparametry (wg gist:
# https://gist.github.com/bchaber/4736db461e8d60dd70bde57e51f5bc24).
# Sluzy do porownania wydajnosci/dokladnosci z wlasna biblioteka (flex.jl) w artykule KM2.
#
# Architektura, optymalizator (Descent, eta=1e-2), batch=10 i 3 epoki sa IDENTYCZNE
# jak w naszej bibliotece — rozni sie tylko to, ze Flux przetwarza CALE batche tensorow
# (W x H x C x batch), a flex pojedyncze obrazy z akumulacja gradientu.
#
# Uruchomienie:
#   julia reference/flux/flux_reference.jl
# Wymaga: Flux, MLDatasets  (julia -e 'using Pkg; Pkg.add(["Flux","MLDatasets"])')

using MLDatasets, Flux
using Statistics: mean
using Random, Dates, Printf

function loader(data; batchsize::Int=1)
    x4dim = reshape(Float32.(data.features), 28, 28, 1, :)
    yhot = Flux.onehotbatch(data.targets, 0:9)
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

function build_net()
    Chain(
        Conv((3, 3), 1 => 6, pad=1, bias=false),
        MaxPool((2, 2)),
        Conv((3, 3), 6 => 16, pad=1, bias=false),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(784 => 84, relu),
        Dropout(0.4),
        Dense(84 => 10),
    )
end

function loss_and_accuracy(model, data)
    (x, y) = only(loader(data; batchsize=length(data)))
    yhat = model(x)
    loss = Flux.logitcrossentropy(yhat, y)
    acc = round(100 * mean(Flux.onecold(yhat) .== Flux.onecold(y)); digits=2)
    (; loss, acc)
end

function main()
    settings = (; eta=1e-2, epochs=3, batchsize=10, seed=1)
    Random.seed!(settings.seed)

    train_data = MLDatasets.FashionMNIST(split=:train)
    test_data = MLDatasets.FashionMNIST(split=:test)

    net = build_net()
    opt_state = Flux.setup(Descent(settings.eta), net)

    # rozgrzewka kompilacji na osobnej sieci (nie liczymy czasu kompilacji do benchmarku)
    let net0 = build_net(), st0 = Flux.setup(Descent(settings.eta), build_net())
        (x, y) = first(loader(train_data, batchsize=settings.batchsize))
        g = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), net0)
        Flux.update!(st0, net0, g[1])
    end

    Flux.trainmode!(net)
    stats = @timed for _ in 1:settings.epochs
        for (x, y) in loader(train_data, batchsize=settings.batchsize)
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), net)
            Flux.update!(opt_state, net, grads[1])
        end
    end

    Flux.testmode!(net)
    te = loss_and_accuracy(net, test_data)

    @printf("== FLUX REFERENCE (full, 3 epoki, seed=%d) ==\n", settings.seed)
    @printf("  time_per_epoch_s  %.2f\n", stats.time / settings.epochs)
    @printf("  alloc_gib         %.3f\n", stats.bytes / 2^30)
    @printf("  gc_frac           %.2f\n", 100 * stats.gctime / stats.time)
    @printf("  test_acc          %.2f\n", te.acc)
    @printf("  test_loss         %.4f\n", te.loss)

    # dopisz do reference/results.csv (tabela porownawcza przy rozwiazaniach referencyjnych)
    # schema: solution,framework,time_per_epoch_s,total_time_s,alloc_gib,test_acc,test_loss
    results = normpath(joinpath(@__DIR__, "..", "results.csv"))
    open(results, "a") do io
        println(io, join(("flux-reference", "Julia / Flux.jl",
                round(stats.time / settings.epochs; digits=2),
                round(stats.time; digits=2),
                round(stats.bytes / 2^30; digits=3),
                te.acc, round(te.loss; digits=4)), ","))
    end
    println("\nzapisano wiersz 'flux-reference' do $results")
end

main()
