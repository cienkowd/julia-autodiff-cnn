using LinearAlgebra
using Random

abstract type Operator end

mutable struct GraphNode{OP,N}
  args::NTuple{N,GraphNode}
  grad
  data
  meta
end

const GraphWeight = GraphNode{:weight,0}
const GraphTensor = GraphNode{:tensor,0}

function GraphNode(data::T, trainable=false) where T
  if trainable
    return GraphNode{:weight,0}((), zero(data), data, nothing)
  else
    return GraphNode{:tensor,0}((), zero(data), data, nothing)
  end
end

function GraphNode(op::Symbol, args::Tuple, data::T, meta=nothing) where T
  N = length(args)
  grad = similar(data)
  return GraphNode{op,N}(args, grad, data, meta)
end

function graph(node)
  function visit!(node::GraphNode, visited, ordered)
    if node in visited
    else
      push!(visited, node)
      for arg in node.args
        visit!(arg, visited, ordered)
      end
      push!(ordered, node)
    end
    return nothing
  end
  ordered = Vector{GraphNode}()
  visited = Set{GraphNode}()
  visit!(node, visited, ordered)
  return ordered
end

function zerograd!(order::Vector{GraphNode})
  for node in order
    node.grad .= 0
  end
end

function primal!(tensor::GraphTensor) end
function primal!(weight::GraphWeight) end
function tangent!(tensor::GraphTensor) end
function tangent!(weight::GraphWeight) end

function forward!(order::Vector{GraphNode}, pairs...)
  for pair in pairs
    tensor, data = pair
    tensor.data .= data
  end
  for node in order
    primal!(node)
  end
end

function forwardd!(order::Vector{GraphNode}, pairs...)
  for pair in pairs
    tensor, grad = pair
    tensor.grad .= grad
  end
  for node in order
    tangent!(node)
  end
end

function adjoint!(::GraphTensor) end
function adjoint!(::GraphWeight) end

function backward!(order::Vector{GraphNode})
  seed = last(order)
  seed.grad .= 1
  for node in reverse(order)
    adjoint!(node)
  end
end

import Base: show
show(io::IO, x::GraphNode{OP,N}) where {OP,N} =
  print(io, "layer ", OP, " with ", N, " arg(s)")
show(io::IO, x::GraphWeight) = print(io, "weight")
show(io::IO, x::GraphTensor) = print(io, "tensor")

function optimize!(order, eta)
  for node in order
    if node isa GraphWeight
      node.data .-= eta * node.grad
    end
  end
end

function primal!(y::GraphNode{:mul,2})
  W, x = y.args
  mul!(y.data, W.data, x.data)            # zamiast y.data = W.data * x.data (alokacja)
  return nothing
end
function adjoint!(y::GraphNode{:mul,2})
  W, x = y.args
  W.grad .+= y.grad .* x.data'            # outer product, broadcast bez tymczasowych
  mul!(x.grad, W.data', y.grad, true, true)  # x.grad += W.data' * y.grad (in-place)
  return nothing
end

function primal!(z::GraphNode{:add,2})
  x, y = z.args
  z.data .= x.data .+ y.data
  return nothing
end
function adjoint!(z::GraphNode{:add,2})
  x, y = z.args
  x.grad .+= z.grad
  y.grad .+= z.grad
  return nothing
end

# Bariery funkcyjne: pola GraphNode sa ::Any (niestabilne typowo). Przekazanie ich do
# osobnej funkcji-kernela powoduje jeden dynamiczny dispatch, po ktorym wnetrze kompiluje
# sie pod konkretne typy tablic (szybkie petle SIMD zamiast boxowanych operacji).
@inline function _relu_fwd!(yd, xd)
  yd .= max.(0f0, xd)
  return nothing
end
@inline function _relu_back!(xg, xd, yd, yg)
  @inbounds for i in eachindex(xd, yd)
    if xd[i] == yd[i]
      xg[i] += yg[i]
    end
  end
  return nothing
end
function primal!(y::GraphNode{:relu,1})
  x, = y.args
  _relu_fwd!(y.data, x.data)
  return nothing
end
function adjoint!(y::GraphNode{:relu,1})
  x, = y.args
  _relu_back!(x.grad, x.data, y.data, y.grad)
  return nothing
end

function primal!(z::GraphNode{:dot,2})
  x, y = z.args
  z.data = dot(x.data, y.data)
  return nothing
end
function adjoint!(z::GraphNode{:dot,2})
  x, y = z.args
  x.grad += y.data .* z.grad
  y.grad += x.data .* z.grad
  return nothing
end

function primal!(y::GraphNode{:sum,1})
  x, = y.args
  y.data = sum(x.data)
  return nothing
end
function adjoint!(y::GraphNode{:sum,1})
  x, = y.args
  x.grad += y.grad
  return nothing
end

function primal!(y::GraphNode{:sigmoid,1})
  x, = y.args
  y.data = 1 ./ (1 .+ exp.(-x.data))
  return nothing
end
function adjoint!(y::GraphNode{:sigmoid,1})
  x, = y.args
  x.grad += exp.(-x.data) ./ (1 .+ exp.(-x.data)) .^ 2 .* y.grad
  return nothing
end

function primal!(z::GraphNode{:bce,2})
  x, y = z.args
  z.data = -(y.data .* log.(x.data) + (1 .- y.data) .* log.(1 .- x.data))
  return nothing
end
function adjoint!(z::GraphNode{:bce,2})
  x, y = z.args
  x.grad -= y.data ./ x.data .* z.grad
  x.grad += (1 .- y.data) ./ (1 .- x.data) .* z.grad
  return nothing
end

struct ReLU <: Operator end
struct Sigmoid <: Operator end

struct Dense <: Operator
  insize::Int64
  outsize::Int64
end

function (y::ReLU)(x)
  sz = length(x.data)
  return GraphNode(:relu, (x,), zeros(Float32, sz))
end

function (y::Sigmoid)(x)
  sz = length(x.data)
  return GraphNode(:sigmoid, (x,), zeros(sz))
end

function (y::Dense)(x)
  n, m = y.insize, y.outsize
  W = GraphNode(randn(Float32, m, n) .* Float32(sqrt(2.0 / n)), true)
  b = GraphNode(zeros(Float32, m), true)
  mul = GraphNode(:mul, (W, x), zeros(Float32, m))
  add = GraphNode(:add, (mul, b), zeros(Float32, m))
  return add
end

mutable struct GradientDescent
  nabla::Dict{Any,Any}
  eta::Float64
  s::Int64
  GradientDescent(eta) = new(Dict(), eta, 0)
end

function accumulate!(opt, order)
  for node in order
    if node isa GraphWeight
      if node in keys(opt.nabla)
        opt.nabla[node] .+= node.grad
      else
        opt.nabla[node] = copy(node.grad)
      end
    end
  end
  opt.s += 1
  return nothing
end

function step!(theta, alpha, nabla_f)
  theta .-= alpha .* nabla_f
  nabla_f .= 0.0
  return nothing
end

function optimize!(opt::GradientDescent, order)
  for node in order
    if node isa GraphWeight
      step!(node.data, opt.eta / opt.s, opt.nabla[node])
    end
  end
  opt.s = 0
  return nothing
end

struct FlattenOp <: Operator end
flatten = FlattenOp()

function (::FlattenOp)(x::GraphNode)
  GraphNode(:flatten, (x,), zeros(Float32, length(x.data)))
end

function primal!(y::GraphNode{:flatten,1})
  x, = y.args
  y.data .= reshape(x.data, :)
  return nothing
end

function adjoint!(y::GraphNode{:flatten,1})
  x, = y.args
  x.grad .+= reshape(y.grad, size(x.data))   # reshape to widok, .+= bez alokacji
  return nothing
end

struct SoftmaxOp <: Operator end

_softmax(v) = (e = exp.(v .- maximum(v)); e ./= sum(e); e)

function (::SoftmaxOp)(x::GraphNode)
  GraphNode(:softmax, (x,), zeros(Float32, length(x.data)))
end

function primal!(y::GraphNode{:softmax,1})
  x, = y.args
  y.data .= _softmax(x.data)
  return nothing
end

function adjoint!(y::GraphNode{:softmax,1})
  x, = y.args
  s = y.data
  J = diagm(s) .- s * s'
  x.grad += J' * y.grad
  return nothing
end

struct LogitCEOp <: Operator end
logit_ce = LogitCEOp()

function (::LogitCEOp)(yhat::GraphNode, y::GraphNode)
  GraphNode(:logit_ce, (yhat, y), zeros(Float32, 1))
end

# Bariery: yhat.data/yhat.grad sa ::Any. Wewnatrz kerneli typy sa konkretne,
# softmax liczony in-line (bez alokacji bufora) i typowo-stabilnie.
function _logit_ce(yh, y)
  m = maximum(yh)
  s = sum(v -> exp(v - m), yh)
  return log(s) + m - dot(yh, y)
end
function _logit_ce_back!(yg, yh, y, g)
  m = maximum(yh)
  s = sum(v -> exp(v - m), yh)
  @inbounds @simd for i in eachindex(yg)
    yg[i] += (exp(yh[i] - m) / s - y[i]) * g
  end
  return nothing
end
function primal!(z::GraphNode{:logit_ce,2})
  yhat, y = z.args
  z.data[1] = _logit_ce(yhat.data, y.data)
  return nothing
end
function adjoint!(z::GraphNode{:logit_ce,2})
  yhat, y = z.args
  _logit_ce_back!(yhat.grad, yhat.data, y.data, z.grad[1])
  return nothing
end

struct ReLUOp <: Operator end
relu = ReLUOp()

function (::ReLUOp)(x::GraphNode)
  GraphNode(:relu, (x,), zeros(Float32, size(x.data)...))
end

function pad2d(x::AbstractArray{T,3}, p::Int) where T
  H, W, C = size(x)
  out = zeros(T, H + 2p, W + 2p, C)
  out[p+1:p+H, p+1:p+W, :] .= x
  return out
end

function im2col(xp::AbstractArray{T,3}, H::Int, W::Int,
  kH::Int, kW::Int, C_in::Int) where T
  col = zeros(T, kH * kW * C_in, H * W)
  for j in 1:W, i in 1:H
    k = (j - 1) * H + i
    for ci in 1:C_in, kj in 1:kW, ki in 1:kH
      @inbounds col[ki+(kj-1)*kH+(ci-1)*kH*kW, k] = xp[i+ki-1, j+kj-1, ci]
    end
  end
  return col
end

function col2im(dc::AbstractArray{T,2}, H::Int, W::Int, C_in::Int,
  kH::Int, kW::Int, pad::Int) where T
  dxp = zeros(T, H + 2pad, W + 2pad, C_in)
  for j in 1:W, i in 1:H
    k = (j - 1) * H + i
    for ci in 1:C_in, kj in 1:kW, ki in 1:kH
      @inbounds dxp[i+ki-1, j+kj-1, ci] += dc[ki+(kj-1)*kH+(ci-1)*kH*kW, k]
    end
  end
  return dxp[pad+1:pad+H, pad+1:pad+W, :]
end

struct Conv2DOp <: Operator
  C_in::Int
  C_out::Int
  kernel_size::Int
  pad::Int
end

function (c::Conv2DOp)(x::GraphNode)
  kH = kW = c.kernel_size
  fan_in = kH * kW * c.C_in
  W = GraphNode(
    randn(Float32, kH, kW, c.C_in, c.C_out) .* Float32(sqrt(2f0 / fan_in)),
    true)
  H, Wd, _ = size(x.data)
  out = zeros(Float32, H, Wd, c.C_out)
  # col/Wmat: cache wynikow forward, wykorzystywany w backward (bez ponownego im2col)
  GraphNode(:conv, (x, W), out, (; pad=c.pad, col=Ref{Any}(nothing), Wmat=Ref{Any}(nothing)))
end

function primal!(y::GraphNode{:conv,2})
  x, W = y.args
  p = y.meta.pad
  H, Wd, C_in = size(x.data)
  kH, kW, _, C_out = size(W.data)

  xp = pad2d(x.data, p)
  col = im2col(xp, H, Wd, kH, kW, C_in)

  W_mat = reshape(permutedims(W.data, (4, 1, 2, 3)), C_out, kH * kW * C_in)

  out = W_mat * col

  y.data .= permutedims(reshape(out, C_out, H, Wd), (2, 3, 1))

  # zapamietaj na potrzeby backward (wagi i wejscie nie zmieniaja sie miedzy
  # forward a backward tej samej probki -> col i W_mat sa wciaz poprawne)
  y.meta.col[] = col
  y.meta.Wmat[] = W_mat
  return nothing
end

function adjoint!(y::GraphNode{:conv,2})
  x, W = y.args
  p = y.meta.pad
  H, Wd, C_in = size(x.data)
  kH, kW, _, C_out = size(W.data)

  # uzyj wartosci policzonych w forward zamiast liczyc pad2d/im2col/permutedims drugi raz
  col = y.meta.col[]::Matrix{Float32}
  W_mat = y.meta.Wmat[]::Matrix{Float32}

  g_mat = reshape(permutedims(y.grad, (3, 1, 2)), C_out, H * Wd)

  dW_mat = g_mat * col'
  W.grad .+= permutedims(reshape(dW_mat, C_out, kH, kW, C_in), (2, 3, 4, 1))

  dx_col = W_mat' * g_mat
  x.grad .+= col2im(dx_col, H, Wd, C_in, kH, kW, p)
  return nothing
end

struct MaxPool2DOp <: Operator
  pool::Int
end

function (p::MaxPool2DOp)(x::GraphNode)
  H, W, C = size(x.data)
  Ho, Wo = H ÷ p.pool, W ÷ p.pool
  GraphNode(:maxpool, (x,), zeros(Float32, Ho, Wo, C), (; pool=p.pool))
end

# Kernel typowo-stabilny (bariera): petle po tablicy rozmiaru splotu — najwiekszy zysk z #3.
function _maxpool_fwd!(yd, xd, pool)
  H, W, C = size(xd)
  Ho, Wo = H ÷ pool, W ÷ pool
  @inbounds for c in 1:C, oi in 1:Ho, oj in 1:Wo
    ri = (oi-1)*pool+1:oi*pool
    rj = (oj-1)*pool+1:oj*pool
    yd[oi, oj, c] = maximum(@view xd[ri, rj, c])
  end
  return nothing
end
function _maxpool_back!(xg, xd, yg, pool)
  H, W, C = size(xd)
  Ho, Wo = H ÷ pool, W ÷ pool
  @inbounds for c in 1:C, oi in 1:Ho, oj in 1:Wo
    ri = (oi-1)*pool+1:oi*pool
    rj = (oj-1)*pool+1:oj*pool
    patch = @view xd[ri, rj, c]
    ki, kj = Tuple(argmax(patch))
    xg[(oi-1)*pool+ki, (oj-1)*pool+kj, c] += yg[oi, oj, c]
  end
  return nothing
end
function primal!(y::GraphNode{:maxpool,1})
  x, = y.args
  _maxpool_fwd!(y.data, x.data, y.meta.pool)
  return nothing
end
function adjoint!(y::GraphNode{:maxpool,1})
  x, = y.args
  _maxpool_back!(x.grad, x.data, y.grad, y.meta.pool)
  return nothing
end

struct DropoutOp <: Operator
  rate::Float32
  training::Ref{Bool}
end

function (d::DropoutOp)(x::GraphNode)
  GraphNode(:dropout, (x,), similar(x.data),
    (; rate=d.rate, training=d.training, mask=Ref{Any}(nothing)))
end

function primal!(y::GraphNode{:dropout,1})
  x, = y.args
  m = y.meta
  if !m.training[]
    y.data .= x.data
  else
    mask = m.mask[]
    if mask === nothing                    # bufor alokowany raz, potem reuse
      mask = similar(x.data)
      m.mask[] = mask
    end
    rand!(mask)
    @. mask = (mask > m.rate) / (1f0 - m.rate)
    @. y.data = x.data * mask
  end
  return nothing
end

function adjoint!(y::GraphNode{:dropout,1})
  x, = y.args
  m = y.meta
  if !m.training[]
    x.grad .+= y.grad
  else
    x.grad .+= y.grad .* m.mask[]
  end
  return nothing
end

struct ConvDesc
  channels::Pair{Int,Int}
  ks::Int
  pad::Int
end
struct MaxPoolDesc
  pool::Int
end
struct DenseDesc
  inout::Pair{Int,Int}
  activation
end
struct DropoutDesc
  rate::Float32
end

Conv(ks::NTuple{2,Int}, ch::Pair{Int,Int}; pad::Int=0, bias::Bool=true) =
  ConvDesc(ch, ks[1], pad)
MaxPool(ks::NTuple{2,Int}) = MaxPoolDesc(ks[1])
Dense(inout::Pair{Int,Int}, act=nothing) = DenseDesc(inout, act)
Dropout(rate::Real) = DropoutDesc(Float32(rate))

struct Model
  x_node::GraphNode
  y_node::GraphNode
  out_node::GraphNode
  loss_node::GraphNode
  graph::Vector
  training::Ref{Bool}
end

function (m::Model)(x::AbstractArray)
  m.x_node.data .= x
  for node in m.graph
    node === m.loss_node && break
    primal!(node)
  end
  return copy(m.out_node.data)
end

function build_layer(d::ConvDesc, x::GraphNode, tr::Ref{Bool})
  Conv2DOp(d.channels[1], d.channels[2], d.ks, d.pad)(x)
end
function build_layer(d::MaxPoolDesc, x::GraphNode, tr::Ref{Bool})
  MaxPool2DOp(d.pool)(x)
end
function build_layer(d::DenseDesc, x::GraphNode, tr::Ref{Bool})
  W = GraphNode(randn(Float32, d.inout[2], d.inout[1]) .*
                Float32(sqrt(2f0 / d.inout[1])), true)
  b = GraphNode(zeros(Float32, d.inout[2]), true)
  mul = GraphNode(:mul, (W, x), zeros(Float32, d.inout[2]))
  add = GraphNode(:add, (mul, b), zeros(Float32, d.inout[2]))
  isnothing(d.activation) ? add : d.activation(add)
end
function build_layer(d::DropoutDesc, x::GraphNode, tr::Ref{Bool})
  DropoutOp(d.rate, tr)(x)
end
function build_layer(d::FlattenOp, x::GraphNode, tr::Ref{Bool})
  d(x)
end

function Chain(layers...)
  training = Ref(true)
  x_node = GraphNode(zeros(Float32, 28, 28, 1))
  y_node = GraphNode(zeros(Float32, 10))
  h = x_node
  for layer in layers
    h = build_layer(layer, h, training)
  end
  out_node = h
  loss_node = logit_ce(out_node, y_node)
  g = graph(loss_node)
  return Model(x_node, y_node, out_node, loss_node, g, training)
end

one_hot(c::Int, n::Int=10) = (v = zeros(Float32, n); v[c+1] = 1f0; v)

onecold(yhat::AbstractVector, labels=0:9) = labels[argmax(yhat)]
onecold(yhat::AbstractMatrix, labels=0:9) = [labels[argmax(yhat[:, i])] for i in axes(yhat, 2)]

logitcrossentropy(yhat::AbstractVector, y::AbstractVector) =
  let m = maximum(yhat)
    log(sum(exp.(yhat .- m))) + m - dot(yhat, y)
  end

function onehotbatch(labels, classes)
  idx = Dict(c => i for (i, c) in enumerate(classes))
  out = zeros(Float32, length(classes), length(labels))
  for (j, c) in enumerate(labels)
    out[idx[c], j] = 1f0
  end
  return out
end

struct DataLoader
  data::Tuple
  batchsize::Int
  shuffle::Bool
end

DataLoader(data::Tuple; batchsize::Int=1, shuffle::Bool=false) =
  DataLoader(data, batchsize, shuffle)

function Base.iterate(dl::DataLoader, state=nothing)
  n = size(dl.data[1], ndims(dl.data[1]))

  if state === nothing
    idx = dl.shuffle ? Random.shuffle(1:n) : collect(1:n)
    state = (idx, 1)
  end

  idx, s = state
  s > n && return nothing

  e = min(s + dl.batchsize - 1, n)
  batch = ntuple(i -> selectdim(dl.data[i], ndims(dl.data[i]), idx[s:e]),
    length(dl.data))
  return batch, (idx, e + 1)
end

Base.length(dl::DataLoader) =
  ceil(Int, size(dl.data[1], ndims(dl.data[1])) / dl.batchsize)