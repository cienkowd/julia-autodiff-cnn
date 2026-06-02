# bench/benchmark.jl
#
# Powtarzalny harness benchmarkowy dla biblioteki autodiff (KM2).
# Mierzy zawsze ten sam zestaw metryk i dopisuje wiersz do bench/results.csv,
# dzieki czemu kazda optymalizacja ma porownywalne liczby "przed/po" do artykulu.
#
# Struktura sieci, parametry SGD i limity dokladnosci sa STALE (wymog polecenia) —
# zmienia sie tylko implementacja biblioteki, nie ten plik.
#
# Uzycie:
#   julia bench/benchmark.jl --mode quick --label "baseline"     # 1 epoka / podzbior (szybko, do PR)
#   julia bench/benchmark.jl --mode full  --label "in-place"     # 3 epoki / pelny zbior (do artykulu)
#   julia bench/benchmark.jl --gradcheck-only                    # sama weryfikacja poprawnosci
#   julia bench/benchmark.jl --mode quick --label x --no-append  # nie dopisuj do results.csv
#
# Wymaga: MLDatasets (FashionMNIST). Instalacja: julia -e 'using Pkg; Pkg.add("MLDatasets")'

using Printf
using Dates
using Random
using Statistics: mean

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "flex.jl"))

using MLDatasets

# ---------------------------------------------------------------------------
# Siec — IDENTYCZNA jak w KM1 (AWID-2026-CNN-flex.ipynb). Nie modyfikowac.
# ---------------------------------------------------------------------------
function build_net()
    Chain(
        Conv((3, 3), 1 => 6, pad=1, bias=false),
        MaxPool((2, 2)),
        Conv((3, 3), 6 => 16, pad=1, bias=false),
        MaxPool((2, 2)),
        flatten,
        Dense(784 => 84, relu),
        Dropout(0.4),
        Dense(84 => 10),
    )
end

const SETTINGS = (; eta = 1e-2, epochs = 3, batchsize = 10)

# ---------------------------------------------------------------------------
# Dane
# ---------------------------------------------------------------------------
function make_loader(data; batchsize, nsub=0, seed=1)
    x = reshape(Float32.(data.features), 28, 28, 1, :)
    y = onehotbatch(data.targets, 0:9)
    if nsub > 0
        Random.seed!(seed)
        idx = randperm(size(x, 4))[1:min(nsub, size(x, 4))]
        x = x[:, :, :, idx]
        y = y[:, idx]
    end
    DataLoader((x, y); batchsize, shuffle=true)
end

function eval_acc(net, data; nsub=0, seed=2)
    net.training[] = false
    X = Float32.(data.features)
    Y = data.targets
    n = size(X, 3)
    idx = nsub > 0 ? (Random.seed!(seed); randperm(n)[1:min(nsub, n)]) : (1:n)
    correct = 0
    total_loss = 0.0
    for i in idx
        xi = reshape(X[:, :, i], 28, 28, 1)
        yi = one_hot(Y[i])
        yhat = net(xi)
        total_loss += logitcrossentropy(yhat, yi)
        correct += (onecold(yhat, 0:9) == Y[i])
    end
    net.training[] = true
    (; acc = round(100 * correct / length(idx); digits=2), loss = total_loss / length(idx))
end

# ---------------------------------------------------------------------------
# Petla uczenia — IDENTYCZNA logika jak w KM1 (pojedyncze obrazy, update co batch).
# ---------------------------------------------------------------------------
function train_epoch!(net, opt, loader)
    for (x_batch, y_batch) in loader
        for i in 1:size(x_batch, 4)
            zerograd!(net.graph)
            net.x_node.data .= x_batch[:, :, :, i]
            net.y_node.data .= y_batch[:, i]
            forward!(net.graph)
            backward!(net.graph)
            accumulate!(opt, net.graph)
        end
        optimize!(opt, net.graph)
    end
end

# ---------------------------------------------------------------------------
# Weryfikacja poprawnosci: gradient z autodiff vs roznice skonczone.
# To jest "weryfikacja optymalizacji" z polecenia — dowod, ze zmiana nie
# popsula matematyki. Dropout wylaczony => forward deterministyczny.
# ---------------------------------------------------------------------------
function gradcheck(; n=8, eps=1.0f-3, seed=1, tol=2.0f-2, verbose=true)
    Random.seed!(seed)
    net = build_net()
    net.training[] = false
    net.x_node.data .= randn(Float32, 28, 28, 1)
    net.y_node.data .= one_hot(rand(0:9))
    zerograd!(net.graph)
    forward!(net.graph)
    backward!(net.graph)

    weights = GraphNode[nd for nd in net.graph if nd isa GraphWeight]
    loss() = (forward!(net.graph); net.loss_node.data[1])

    maxrel = 0.0f0
    for _ in 1:n
        w = rand(weights)
        i = rand(eachindex(w.data))
        g_ad = w.grad[i]
        o = w.data[i]
        w.data[i] = o + eps; lp = loss()
        w.data[i] = o - eps; lm = loss()
        w.data[i] = o
        g_fd = (lp - lm) / (2eps)
        rel = abs(g_ad - g_fd) / (abs(g_ad) + abs(g_fd) + 1.0f-8)
        maxrel = max(maxrel, rel)
    end
    ok = maxrel < tol
    verbose && @printf("gradcheck: max blad wzgledny = %.4e  (tol=%.0e)  -> %s\n",
        maxrel, tol, ok ? "OK" : "FAIL")
    return ok
end

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
git_commit() = try
    strip(read(`git -C $ROOT rev-parse --short HEAD`, String))
catch
    "unknown"
end

function run_benchmark(; mode="quick", label="unnamed", seed=1)
    if mode == "quick"
        epochs, ntrain, ntest = 1, 6000, 2000
    elseif mode == "full"
        epochs, ntrain, ntest = SETTINGS.epochs, 0, 0
    else
        error("nieznany tryb: $mode (quick|full)")
    end

    Random.seed!(seed)
    train = MLDatasets.FashionMNIST(split=:train)
    test = MLDatasets.FashionMNIST(split=:test)

    net = build_net()
    opt = GradientDescent(SETTINGS.eta)
    loader = make_loader(train; batchsize=SETTINGS.batchsize, nsub=ntrain, seed=seed)

    # rozgrzewka kompilacji (nie liczymy czasu kompilacji do benchmarku)
    let net0 = build_net(), opt0 = GradientDescent(SETTINGS.eta)
        net0.x_node.data .= 0
        net0.y_node.data .= one_hot(0)
        zerograd!(net0.graph)
        forward!(net0.graph)
        backward!(net0.graph)
        accumulate!(opt0, net0.graph)
        optimize!(opt0, net0.graph)
    end

    stats = @timed for _ in 1:epochs
        train_epoch!(net, opt, loader)
    end

    ev = eval_acc(net, test; nsub=ntest, seed=seed + 100)

    res = (;
        timestamp = string(now()),
        commit = git_commit(),
        label = label,
        mode = mode,
        epochs = epochs,
        seed = seed,
        time_s = round(stats.time; digits=2),
        time_per_epoch_s = round(stats.time / epochs; digits=2),
        alloc_gib = round(stats.bytes / 2^30; digits=3),
        gc_frac = round(100 * stats.gctime / stats.time; digits=2),
        test_acc = ev.acc,
        test_loss = round(ev.loss; digits=4),
    )
    return res
end

# ---------------------------------------------------------------------------
# Zapis wynikow
# ---------------------------------------------------------------------------
const RESULTS = joinpath(@__DIR__, "results.csv")
const HEADER = "timestamp,commit,label,mode,epochs,seed,time_s,time_per_epoch_s,alloc_gib,gc_frac,test_acc,test_loss"

function append_result(res)
    newfile = !isfile(RESULTS)
    open(RESULTS, "a") do io
        newfile && println(io, HEADER)
        println(io, join((res.timestamp, res.commit, res.label, res.mode, res.epochs,
                res.seed, res.time_s, res.time_per_epoch_s, res.alloc_gib,
                res.gc_frac, res.test_acc, res.test_loss), ","))
    end
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
function parse_cli(args)
    mode, label, seed, gradonly, noappend = "quick", "unnamed", 1, false, false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--mode";            mode = args[i+1];               i += 2
        elseif a == "--label";       label = args[i+1];              i += 2
        elseif a == "--seed";        seed = parse(Int, args[i+1]);   i += 2
        elseif a == "--gradcheck-only"; gradonly = true;             i += 1
        elseif a == "--no-append";   noappend = true;                i += 1
        else error("nieznany argument: $a")
        end
    end
    (; mode, label, seed, gradonly, noappend)
end

function main()
    o = parse_cli(ARGS)

    println("== WERYFIKACJA POPRAWNOSCI (gradcheck) ==")
    ok = gradcheck(seed=o.seed)
    o.gradonly && exit(ok ? 0 : 1)

    println("\n== BENCHMARK ($(o.mode), label=\"$(o.label)\") ==")
    res = run_benchmark(mode=o.mode, label=o.label, seed=o.seed)
    for k in keys(res)
        @printf("  %-18s %s\n", string(k), string(getfield(res, k)))
    end

    if o.noappend
        println("\n(--no-append: nie zapisano do results.csv)")
    else
        append_result(res)
        println("\nzapisano wiersz do $(RESULTS)")
    end

    println(ok ? "\nWYNIK: gradienty OK" : "\nWYNIK: gradienty FAIL — optymalizacja zmienila matematyke!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
