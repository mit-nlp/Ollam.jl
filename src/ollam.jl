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
module ollam
using Stage, LIBSVM, SVM
import Base: copy, start, done, next
export LinearModel, copy, score, best, train_perceptron, test_classification, train_svm, train_mira, train_svm2

# ----------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------
logger = Log(STDERR)

immutable Map{I}
    flt::Function
    itr::I
end
lazy_map(f::Function, itr) = Map(f, itr)

function start(m::Map) 
  s = start(itr)
  return s
end

function next(m :: Map, s) 
  n = next(m.itr, s)
  return (flt(n), s)
end
done(m :: Map, s) = done(m.itr)

# const fpattern = r"^(.*?)\$\((.*)(%.*?)\)\$(.*)$"
# sprintf(fmt::String,args...) = @eval @sprintf($fmt,$(args...))

# macro f_str(str)
#   local result = ""
#   local s = str
#   local m = match(fpattern, s)
#   if m == nothing
#     s
#   else
#     while m != nothing
#       println("debug1: $m")
#       result *= m.captures[1]
#       fmt = m.captures[3]
#       result *= @sprintf("%10.2f", 10.2) #eval(m.captures[2]))
#       println("debug2: $result")
#       println("result: $result")
#       s = m.captures[4]
#       m = match(fpattern, s)
#     end
#     result
#   end
# end  

# ----------------------------------------------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------------------------------------------
type LinearModel{T}
  weights     :: Matrix{Float64}
  b           :: Vector{Float64}

  class_index :: Dict{T, Int32}
  index_class :: Array{T, 1}
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

copy(lm :: LinearModel) = LinearModel(copy(lm.weights), copy(lm.b), copy(lm.class_index), copy(lm.index_class))
score(lm :: LinearModel, fv) = [ (lm.weights[c, :] * fv)[1] + lm.b[c] for c = 1:size(lm.weights, 1) ]

function best{T <: FloatingPoint}(scores :: Vector{T}) 
  bidx = indmax(scores)
  return bidx, scores[bidx]
end

function test_classification(lm :: LinearModel, fvs, truth)
  errors = 0

  for (fv, t) in zip(fvs, truth)
    scores  = score(lm, fv)
    bidx, b = best(scores)
    if lm.index_class[bidx] != t
      errors += 1
    end
  end

  return errors / length(truth)
end

# ----------------------------------------------------------------------------------------------------------------
# Perceptron
# ----------------------------------------------------------------------------------------------------------------
function train_perceptron(fvs, truth, init_model; learn_rate = 1.0, average = true, iterations = 40, log = Log(STDERR))
  model = copy(init_model)
  acc   = LinearModel(init_model.class_index, dims(init_model))

  for i = 1:iterations
    for (fv, t) in zip(fvs, truth)
      scores     = score(model, fv)
      bidx, b    = best(scores)
      if model.index_class[bidx] != t
        for c = 1:classes(model)
          sign = model.index_class[c] == t ? 1.0 : (-1.0 / (classes(model) - 1))
          model.weights[c, :] += sign * learn_rate * fv'
          if average
            acc.weights += model.weights
          end
        end
      end
    end
    @info log @sprintf("iteration %3d complete (Training error rate: %7.3f%%)", i, test_classification(model, fvs, truth) * 100.0)
  end
  
  if average
    acc.weights /= (length(fvs) * iterations)
    return acc
  else
    return model
  end
end

# ----------------------------------------------------------------------------------------------------------------
# MIRA
# ----------------------------------------------------------------------------------------------------------------
function train_mira(fvs, truth, init_model; average = true, C = 0.1, k = 3, iterations = 40, lossfn = (a, b) -> a == b ? 0.0 : 1.0, log = Log(STDERR))
  model = copy(init_model)
  acc   = LinearModel(init_model.class_index, dims(init_model))

  for i = 1:iterations
    for (fv, t) in zip(fvs, truth)
      scores        = score(model, fv)
      tidx          = model.class_index[t]
      tgt_score     = scores[tidx]
      bidx, b_score = best(scores)

      # K-best
      # b       = (Float64)[]
      # sorted  = sort(enumerate(scores), rev = true, by = x -> x[2])
      # distvec = (AbstractVector{Float64})[]
      # for n = 1:min(k, length(sorted))
      #   cidx, score = sorted[n]
      #   class       = model.index_class[cidx]
      #   targetP     = class == t
      #   loss        = targetP ? 0.0 : 1.0
      #   dist        = tgt_score - score

      #   push!(b, loss - dist)
      #   push!(distvec, spzero(dims(model)))
      # end

      # 1-best
      class = model.index_class[bidx]
      loss  = lossfn(t, class)
      dist  = tgt_score - b_score
      alpha = min((loss - dist) / (2 * dot(fv, fv)), C)
      #@debug logger "loss = $loss, dist = $dist [$tgt_score - $b_score], denom = $(2 * dot(fv, fv)), alpha = $alpha"

      model.weights[bidx, :] -= alpha * fv'
      model.weights[tidx, :] += alpha * fv'
      if average
        acc.weights += model.weights
      end
    end
    @info log @sprintf("iteration %3d complete (Training error rate: %7.3f%%)", i, test_classification(model, fvs, truth) * 100.0)
  end
  
  if average
    acc.weights /= (length(fvs) * iterations)
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

function transfer(svm)
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
  weights = zeros(svm.nfeatures) #Array(Float64, svm.nfeatures)

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

function train_svm(fvs, truth; C = 1.0, nu = 0.5, cache_size = 200.0, eps = 0.0001, shrinking = true, verbose = false, log = Log(STDERR))
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
        svm_t = svmtrain(map(c -> c == t ? 1 : -1, truth), feats; gamma = 0.5,
                         C = C, nu = nu, kernel_type = int32(0), degree = int32(1), svm_type = int32(0),
                         cache_size = cache_size, eps = eps, shrinking = shrinking, verbose = verbose)
        transfer(svm_t)
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

  return model # transfer(classes, svms)
end

function train_svm2(fvs, truth; C = 0.01, batch_size = -1, iterations = 100)
  i = 1
  classes = Dict{Any, Int32}()

  for t in truth
    if !(t in keys(classes))
      classes[t] = i
      i += 1
    end
  end

  feats = hcat(fvs...)
  if batch_size == -1
    batch_size = size(feats, 2)
  end

  model = LinearModel(classes, size(feats, 1))

  svms  = Array(Any, length(classes))
  refs  = Array(Any, length(classes))

  for (t, ti) in classes
    @timer logger "training svm for class $t (index: $ti)" begin
      refs[ti] = @spawn begin
        svm_t = svm(feats, map(c -> c == t ? 1 : -1, truth);
                    lambda = C, T = iterations, k = batch_size)
        (svm_t.w, 0.0)
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

  return model # transfer(classes, svms)
end

end # module end
