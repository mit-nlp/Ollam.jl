# ollam.jl
#
# author: Wade Shen
# swade@ll.mit.edu
# Copyright &copy; 2009 Massachusetts Institute of Technology, Lincoln Laboratory
# version 0.1
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
module Ollam
using Stage, LIBSVM, SVM, DataStructures
import Base: copy, start, done, next, length, dot
export LinearModel, RegressionModel, copy, score, best, train_perceptron, test_classification, test_regression,
       train_svm, train_mira, train_libsvm, lazy_map, indices, 
       print_confusion_matrix, hildreth, setup_hildreth, zero_one_loss, linear_regression_loss, 
       regress_perceptron, regress_mira

# ----------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------
immutable Map{I}
    flt::Function
    itr::I
end
lazy_map(f::Function, itr) = Map(f, itr)

function start(m :: Map) 
  s = start(m.itr)
  return s
end

function next(m :: Map, s) 
  n, ns = next(m.itr, s)
  return (m.flt(n), ns)
end

done(m :: Map, s) = done(m.itr, s)
length(m :: Map) = length(m.itr)

# linear algebra helpers
indices(a :: SparseMatrixCSC) = a.rowval
indices(a :: Vector)          = 1:length(a)
getnth(a :: SparseMatrixCSC, i :: Integer) = a.nzval[i]

sqr(a::Vector) = norm(a)^2
function sqr(a::SparseMatrixCSC)
  total = 0.0
  for i in indices(a)
    total += a[i] * a[i]
  end
  return total
end
dot(a::SparseMatrixCSC, b::Vector) = dot(b, a)
dot(a::SparseMatrixCSC, b::Matrix) = dot(b, a)
function dot(a::Union(Vector, SparseMatrixCSC), b::SparseMatrixCSC)
  total = 0.0
  for i in indices(b)
    total += a[i] * b[i]
  end
  return total
end
function dot(a::Matrix, b::SparseMatrixCSC) 
  total = 0.0
  for i in indices(b)
    total += a[1, i] * b[i]
  end
  return total
end
dot(a::Matrix, b::Vector) = (a * b)[1]

function print_confusion_matrix(confmat; width = 10, logger = Log(STDERR))
  width = max(5, width)
  sfmt = "%$(width)s"
  dfmt = "%$(width)d"
  ffmt = "%$(width).$(width-3)f"
  @eval s(x) = @sprintf($sfmt, x)
  @eval d(x) = @sprintf($dfmt, x)
  @eval f(x) = @sprintf($ffmt, x)
  total, errors = 0, 0
  accs = 0.0

  str = s("")
  for t in keys(confmat)
    str *= " " * s(t)
  end
  @sep logger
  @info logger "$str " * s("N") * " " * s("cl %")
  
  for t in keys(confmat)
    str = s(t)
    rtotal, rerrors = 0, 0
    for h in keys(confmat)
      str *= " " * d(confmat[t][h]) # @sprintf(" %10d", confmat[t][h])
      if t != h
        rerrors += confmat[t][h]
      end
      rtotal += confmat[t][h]
    end
    errors += rerrors
    total  += rtotal
    @info logger "$str" * " " * d(rtotal) * " " * f(1.0 - rerrors/rtotal) #@sprintf(" %10d %10.7f", rtotal, 1.0 - rerrors/rtotal)
    accs += 1.0 - rerrors/rtotal
  end
  @sep logger
  
  @info logger "overall accuracy = $(1.0 - errors/total), average class accuracy = $(accs / length(keys(confmat)))"
  
end

# ----------------------------------------------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------------------------------------------
type LinearModel{T}
  weights     :: Matrix{Float64}
  b           :: Vector{Float64}

  class_index :: Dict{T, Int32}
  index_class :: Vector{T}
end

dims(lm :: LinearModel)    = size(lm.weights, 2)
classes(lm :: LinearModel) = size(lm.weights, 1)

function LinearModel{T}(classes::Dict{T, Int32}, dims) 
  index = Array(T, length(classes))
  for (k, i) in classes
    index[i] = k
  end

  return LinearModel(zeros(length(index), dims), zeros(length(index)), classes, index)
end

RegressionModel(dims) = LinearModel((String=>Int32)["regressor" => 1], dims)

copy(lm :: LinearModel) = LinearModel(copy(lm.weights), copy(lm.b), copy(lm.class_index), copy(lm.index_class))
score(lm :: LinearModel, fv::Vector) = lm.weights * fv + lm.b
score(lm :: LinearModel, fv::SparseMatrixCSC) = vec(lm.weights * fv + lm.b)

function acc_update(model :: LinearModel, acc :: LinearModel)
  for x = 1:size(acc.weights, 1)
    for y = 1:size(acc.weights, 2)
      acc.weights[x, y] += model.weights[x, y]
    end
  end
end

function best{T <: FloatingPoint}(scores :: Vector{T}) 
  bidx = indmax(scores)
  return bidx, scores[bidx]
end

function test_classification(lm :: LinearModel, fvs, truth; record = (truth, hyp) -> nothing)
  errors = 0
  total  = 0
  
  for (fv, t) in zip(fvs, truth)
    scores  = score(lm, fv)
    bidx, b = best(scores)
    if lm.index_class[bidx] != t
      errors += 1
    end
    record(t, lm.index_class[bidx])
    total += 1
  end

  return errors / total
end

function test_regression(lm :: LinearModel, fvs, truth; lossfn = linear_regression_loss)
  N = 0
  total_se = 0.0
  for (fv, t) in zip(fvs, truth)
    scores  = score(lm, fv)
    bidx, b = best(scores)
    total_se += lossfn(b, t)^2
    N += 1
  end

  return total_se / N
end

# ----------------------------------------------------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------------------------------------------------
zero_one_loss(hyp, ref) = hyp == ref ? 0.0 : 1.0
linear_regression_loss(hyp, ref) = ref - hyp

# ----------------------------------------------------------------------------------------------------------------
# Perceptron
# ----------------------------------------------------------------------------------------------------------------
function perceptron_update(model, c, alpha, fv)
  for i in indices(fv)
    model.weights[c, i] += alpha * fv[i]
  end
end

function train_perceptron(fvs, truth, init_model; learn_rate = 1.0, average = true, iterations = 40, 
                          logger = Log(STDERR), verbose = true)
  model = copy(init_model)
  acc   = LinearModel(init_model.class_index, dims(init_model))
  numfv = 0
  for fv in fvs
    numfv += 1
  end
  avg_w = iterations * numfv

  for i = 1:iterations
    fj = 1
    for (fv, t) in zip(fvs, truth)
      scores     = score(model, fv)
      bidx, b    = best(scores)
      w          = (avg_w - (numfv * (i - 1) + fj) + 1)
      if model.index_class[bidx] != t
        for c = 1:classes(model)
          sign = model.index_class[c] == t ? 1.0 : (-1.0 / (classes(model) - 1))
          perceptron_update(model, c, sign * learn_rate, fv)
          perceptron_update(acc, c, w * sign * learn_rate, fv)
        end
      end
      fj += 1
    end
    if verbose
      @info logger @sprintf("iteration %3d complete (Training error rate: %7.3f%%)", i, test_classification(model, fvs, truth) * 100.0)
    end
  end
  
  if average
    return acc
  else
    return model
  end
end

function regress_perceptron(fvs, truth, init_model; learn_rate = 0.01, average = true, iterations = 40, C = 0.1,
                            logger = Log(STDERR), verbose = true, lossfn = linear_regression_loss)
  model = copy(init_model)
  acc   = LinearModel(init_model.class_index, dims(init_model))
  numfv = 0
  for fv in fvs
    numfv += 1
  end
  avg_w = iterations * numfv

  for i = 1:iterations
    fj = 1
    for (fv, t) in zip(fvs, truth)
      scores  = score(model, fv)
      bidx, b = best(scores)
      w       = (avg_w - (numfv * (i - 1) + fj) + 1)
      loss    = lossfn(b, t)
      alpha   = max(min(C, loss * learn_rate), -C)
      #@debug @sprintf("loss = %10.3f, alpha = %10.7f, expected = %10.7f, ref = %10.7f fv = %s", loss, alpha, b, t, fv)
      perceptron_update(model, 1, alpha, fv)
      perceptron_update(acc, 1, w * alpha, fv)
      fj += 1
    end
    if verbose
      @info logger @sprintf("iteration %3d complete (Training RMSE: %7.3f)", i, sqrt(test_regression(model, fvs, truth)))
    end
  end
  
  if average
    acc.weights /= avg_w
    return acc
  else
    return model
  end
end

# ----------------------------------------------------------------------------------------------------------------
# MIRA
# ----------------------------------------------------------------------------------------------------------------
function mira_update(weights, bidx, tidx, alpha, fv :: SparseMatrixCSC)
  ind = indices(fv)
  for i in 1:length(ind)
    idx = ind[i]
    tmp = alpha * fv.nzval[i] # direct access avoids fv[idx] binary search in getindex()
    weights[bidx, idx] -= tmp
    weights[tidx, idx] += tmp
  end
end

function mira_update(weights, bidx, tidx, alpha, fv :: Array)
  for idx in indices(fv)
    tmp = alpha * fv[idx] # slow for sparse because of getindex() [see above specialization]
    weights[bidx, idx] -= tmp
    weights[tidx, idx] += tmp
  end
end

type HildrethState
  k           :: Int32
  alpha       :: Vector{Float64}
  F           :: Vector{Float64}
  kkt         :: Vector{Float64}
  C           :: Float64
  A           :: Matrix{Float64}
  is_computed :: Vector{Bool}
  EPS         :: Float64
  ZERO        :: Float64
  MAX_ITER    :: Float64
end

function setup_hildreth(;k = 5, C = 0.1, EPS = 1e-8, ZERO = 0.0000000000000001, MAX_ITER = 10000) # assumes that the number of contraints == number of distances
  alpha       = zeros(k)
  F           = zeros(k)
  kkt         = zeros(k)
  A           = zeros(k, k)
  is_computed = falses(k)
  return HildrethState(k, alpha, F, kkt, C, A, is_computed, EPS, ZERO, MAX_ITER)
end

# translated from Ryan McDonald's MST Parser
function hildreth(a, b, h)
  max_kkt = -Inf
  max_kkt_i = -1

  for i = 1:h.k
    h.A[i, i] = dot(a[i], a[i])
    h.kkt[i] = h.F[i] = b[i]
    h.is_computed[i] = false
    if h.kkt[i] > max_kkt
      max_kkt   = h.kkt[i]
      max_kkt_i = i
    end
    h.alpha[i] = 0.0
  end

  iter = 0
  while max_kkt >= h.EPS && iter < h.MAX_ITER
    diff_alpha = h.A[max_kkt_i, max_kkt_i] <= h.ZERO ? 0.0 : h.F[max_kkt_i] / h.A[max_kkt_i, max_kkt_i]
    try_alpha  = h.alpha[max_kkt_i] + diff_alpha
    add_alpha  = 0.0
    
    if try_alpha < 0.0
      add_alpha = - h.alpha[max_kkt_i]
    elseif try_alpha > h.C
      add_alpha = h.C - h.alpha[max_kkt_i]
    else
      add_alpha = diff_alpha
    end

    h.alpha[max_kkt_i] += add_alpha

    if !h.is_computed[max_kkt_i]
      for i = 1:h.k
	h.A[i, max_kkt_i] = dot(a[i], a[max_kkt_i])
	h.is_computed[max_kkt_i] = true
      end
    end
    for i = 1:h.k
      h.F[i]  -= add_alpha * h.A[i, max_kkt_i]
      h.kkt[i] = h.F[i]
      if h.alpha[i] > (h.C - h.ZERO)
	h.kkt[i] = -h.kkt[i]
      elseif h.alpha[i] > h.ZERO
	h.kkt[i] = abs(h.F[i])
      end
    end		
    max_kkt   = -Inf
    max_kkt_i = -1
    for i = 1:h.k
      if h.kkt[i] > max_kkt 
        max_kkt   = h.kkt[i]
        max_kkt_i = i
      end
    end
    iter += 1
  end

  return h.alpha
end

type ScaledVec{T}
  k :: T
  v
end
getindex(dv :: ScaledVec, i :: Integer) = dv.k * dv.v[i]
dot(v1 :: ScaledVec, v2 :: ScaledVec) = v1.k * v2.k * dot(v1.v, v2.v)

function train_mira(fvs, truth, init_model; 
                    average = true, C = 0.1, k = 1, iterations = 20, lossfn = zero_one_loss,
                    logger = Log(STDERR), verbose = true)
  model = copy(init_model)
  acc   = LinearModel(init_model.class_index, dims(init_model))
  acc2  = LinearModel(init_model.class_index, dims(init_model))
  numfv = 0
  for fv in fvs
    numfv += 1
  end

  h       = setup_hildreth(k = min(k, length(model.class_index)), C = C)
  b       = Array(Float64, h.k)
  kidx    = Array(Int32, h.k)
  distvec = Array(ScaledVec, h.k) #Array(Union(SparseMatrixCSC, Vector), h.k)
  avg_w   = iterations * numfv
  
  for i = 1:iterations
    fj = 1
    alpha = 0.0
    for (fv, t) in zip(fvs, truth)
      scores    = score(model, fv)
      tidx      = model.class_index[t]
      tgt_score = scores[tidx]
      w         = (avg_w - (numfv * (i - 1) + fj) + 1)

      if h.k > 1 # K-best
        sorted = sortperm(scores, rev = true)
        for n = 1:h.k
          cidx        = sorted[n]
          score       = scores[cidx]
          class       = model.index_class[cidx]
          loss        = lossfn(t, class)
          dist        = tgt_score - score
        
          b[n]       = loss - dist
          distvec[n] = ScaledVec(2.0, fv) # 2 * fv # slow due to allocation
          kidx[n]    = cidx
        end
        
        alphas = hildreth(distvec, b, h)

        # update
        for n = 1:h.k
          mira_update(model.weights, kidx[n], tidx, alphas[n], fv)
          mira_update(acc.weights, kidx[n], tidx, w * alphas[n], fv)
        end
      else # 1-best
        bidx, b_score = best(scores)
        class = model.index_class[bidx]
        loss  = lossfn(t, class)
        dist  = tgt_score - b_score
        alpha = max(min((loss - dist) / (2 * sqr(fv)), C), -C)

        #@debug logger "truth: $t -- best class $(model.index_class[bidx]) -- best score: $b_score, truth score: $tgt_score"
        #@debug logger "loss = $loss, dist = $dist [$tgt_score - $b_score], denom = $(2 * norm(fv)^2), alpha = $alpha"
        mira_update(model.weights, bidx, tidx, alpha, fv)
        mira_update(acc.weights, bidx, tidx, w * alpha, fv)
      end

      fj += 1
    end
    if verbose
      @info logger @sprintf("iteration %3d complete (Training error rate: %7.3f%%)", i, test_classification(model, fvs, truth) * 100.0)
    end
  end
  
  if average
    return acc
  else
    return model
  end
end

function mira_regress_update(weights, bidx, alpha, fv :: SparseMatrixCSC)
  ind = indices(fv)
  for i in 1:length(ind)
    idx = ind[i]
    tmp = alpha * fv.nzval[i] # direct access avoids fv[idx] binary search in getindex()
    weights[bidx, idx] += tmp
  end
end

function mira_regress_update(weights, bidx, alpha, fv :: Array)
  for idx in indices(fv)
    tmp = alpha * fv[idx] # slow for sparse because of getindex() [see above specialization]
    weights[bidx, idx] += tmp
  end
end

function regress_mira(fvs, truth, init_model; 
                    average = true, C = 0.1, min_loss = 0.0, iterations = 20, lossfn = linear_regression_loss,
                    logger = Log(STDERR), verbose = true)
  model = copy(init_model)
  acc   = LinearModel(init_model.class_index, dims(init_model))
  numfv = 0
  for fv in fvs
    numfv += 1
  end

  avg_w   = iterations * numfv
  
  for i = 1:iterations
    fj = 1
    alpha = 0.0
    for (fv, t) in zip(fvs, truth)
      scores        = score(model, fv)
      w             = (avg_w - (numfv * (i - 1) + fj) + 1)
      bidx, b_score = best(scores)
      loss          = lossfn(b_score, t)
      if abs(loss) < min_loss 
        loss = 0.0
      end
      alpha         = max(min(loss / sqr(fv), C), -C)
      
      mira_regress_update(model.weights, bidx, alpha, fv)
      #if fj % 100 == 0 # abs(loss) > 7.0
      #  @debug "$fj: $(model.weights) <- $loss"
      #end
      mira_regress_update(acc.weights, bidx, w * alpha, fv)

      fj += 1
    end
    if verbose
      @info logger @sprintf("iteration %3d complete (Training RMSE: %7.3f)", i, sqrt(test_regression(model, fvs, truth)))
    end
  end
  
  if average
    acc.weights /= avg_w
    return acc
  else
    return model
  end
end

# ----------------------------------------------------------------------------------------------------------------
# SVM
# ----------------------------------------------------------------------------------------------------------------
immutable SVMNode
  index::Int32
  value::Float64
end

immutable SVMModel
  param::LIBSVM.SVMParameter
  nr_class::Int32
  l::Int32
  SV::Ptr{Ptr{LIBSVM.SVMNode}}
  sv_coef::Ptr{Ptr{Float64}}
  rho::Ptr{Float64}
  probA::Ptr{Float64}
  probB::Ptr{Float64}
  sv_indices::Ptr{Int32}

  label::Ptr{Int32}
  nSV::Ptr{Int32}

  free_sv::Int32
end

function transfer_sv(p::Ptr{LIBSVM.SVMNode})
  ret  = (LIBSVM.SVMNode)[]
  head = unsafe_load(p)

  while head.index != -1
    push!(ret, head)
    p += 16
    head = unsafe_load(p)
  end
  return ret
end

function transfer(svm; logger = Log(STDERR))
  # unpack svm model
  ptr    = unsafe_load(convert(Ptr{SVMModel}, svm.ptr))
  nSV    = pointer_to_array(ptr.nSV, ptr.nr_class)
  xSV    = pointer_to_array(ptr.SV, ptr.l)
  SV     = (Array{LIBSVM.SVMNode, 1})[ transfer_sv(x) for x in xSV ]
  xsvc   = pointer_to_array(ptr.sv_coef, ptr.nr_class)
  svc    = (Array{Float64, 1})[ pointer_to_array(x, ptr.l) for x in xsvc ]
  labels = pointer_to_array(ptr.label, ptr.nr_class)
  rho    = pointer_to_array(ptr.rho, 1)
  @debug logger "# of SVs = $(length(SV)), labels = $labels, rho = $rho, $(svm.labels)"
  
  # precompute classifier weights
  start = 1
  weights = zeros(svm.nfeatures)

  for i = 1:ptr.nr_class
    for sv_offset = 0:(nSV[i]-1)
      sv = SV[start + sv_offset]
      for d = 1:length(sv)
        weights[sv[d].index] += svc[1][start + sv_offset] * sv[d].value
      end
    end
    start += nSV[i]
  end
  b = -rho[1]

  if svm.labels[1] == -1
    weights = -weights
    b       = -b
  end

  return (weights, b)
end

function train_libsvm(fvs, truth; C = 1.0, nu = 0.5, cache_size = 200.0, eps = 0.0001, shrinking = true, verbose = false, gamma = 0.5, logger = Log(STDERR))
  i = 1
  classes = Dict{Any, Int32}()

  for t in truth
    if !(t in keys(classes))
      classes[t] = i
      i += 1
    end
  end


  feats = hcat(fvs...)
  model = LinearModel(classes, size(feats, 1))

  svms  = Array(Any, length(classes))
  refs  = Array(Any, length(classes))

  for (t, ti) in classes
    @timer logger "training svm for class $t (index: $ti)" begin
      refs[ti] = @spawn begin
        svm_t = svmtrain(map(c -> c == t ? 1 : -1, truth), feats; 
                         gamma = gamma, C = C, nu = nu, kernel_type = int32(0), degree = int32(1), svm_type = int32(0),
                         cache_size = cache_size, eps = eps, shrinking = shrinking, verbose = verbose)
        transfer(svm_t, logger = logger)
      end
    end
  end

  for c = 1:length(refs)
    svms[c] = fetch(refs[c])
  end

  for c = 1:length(svms)
    weights_c, b_c = svms[c]
    for i = 1:length(weights_c)
      model.weights[c, i] = weights_c[i]
    end
    model.b[c] = b_c
  end

  return model
end

function train_svm(fvs, truth; C = 0.01, batch_size = -1, norm = 2, iterations = 100, logger = Log(STDERR))
  i = 1
  classes = Dict{Any, Int32}()

  for t in truth
    if !(t in keys(classes))
      classes[t] = i
      i += 1
    end
  end

  fs = hcat(fvs...)
  feats = vcat(fs, ones(1, size(fs, 2)))

  if batch_size == -1
    batch_size = size(feats, 2)
  end

  model = LinearModel(classes, size(fs, 1))

  svms  = Array(Any, length(classes))
  refs  = Array(Any, length(classes))

  for (t, ti) in classes
    @timer logger "training svm for class $t (index: $ti)" begin
      refs[ti] = @spawn begin
        svm_t = cddual(feats, map(c -> c == t ? 1 : -1, truth);
                       C = C, norm = norm, randomized = true, maxpasses = iterations)
        (svm_t.w[1:end-1], svm_t.w[end])
      end
    end
  end

  for c = 1:length(refs)
    svms[c] = fetch(refs[c])
  end

  for c = 1:length(svms)
    weights_c, b_c = svms[c]
    for i = 1:length(weights_c)
      model.weights[c, i] = weights_c[i]
    end
    model.b[c] = b_c
  end

  return model
end

end # module end
