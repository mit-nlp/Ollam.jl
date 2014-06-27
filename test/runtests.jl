# runtest.jl
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
using ollam
using MNIST, Stage, LIBSVM, PyCall
@pyimport scipy.ndimage.interpolation as interpolation
import Base: length, start, done, next
using Base.Test

# Helpers
log = Log(STDERR)

immutable EachCol{T}
  matrix :: Array{T, 2}
end
length(e :: EachCol)      = size(e.matrix, 2)
start(e :: EachCol)       = 1
next(e :: EachCol, state) = (e.matrix[:, state], state + 1)
done(e :: EachCol, state) = state > length(e) ? true : false

# ----------------------------------------------------------------------------------------------------------------
# test hildreth
# ----------------------------------------------------------------------------------------------------------------
h = setup_hildreth(C = Inf, k = 2)
@test abs(hildreth((SparseMatrixCSC)[ spzeros(1235, 1),
                sparsevec([470=>-1.0, 1070=>-1.0, 1231=>-1.0, 1232=>-1.0, 1233=>-1.0, 1234=>-1.0, 1235=>-1.0, 496=>1.0, 
                           1058=>1.0, 1226=>1.0, 1227=>1.0, 1228=>1.0, 1229=>1.0, 1230=>1.0]) ], [ 0.0, 0.9733606705258293 ], h)[2] - 0.069526) < 0.00001

# ----------------------------------------------------------------------------------------------------------------
# test lazy maps
# ----------------------------------------------------------------------------------------------------------------
xxx = [1, 2, 3]
for i in lazy_map(f -> f+1, xxx)
  @info log i
end

# ----------------------------------------------------------------------------------------------------------------
# setup MNIST task
# ----------------------------------------------------------------------------------------------------------------
const c0 = [ i for i = 0:27, j = 0:27 ] #mgrid[28, 28]
const c1 = [ j for i = 0:27, j = 0:27 ]

function moments(image)
  si  = sum(image)
  m0  = sum(c0 .* image)/si
  m1  = sum(c1 .* image)/si
  m00 = sum((c0 - m0) .* (c0 - m0) .* image)/si
  m11 = sum((c1 - m1) .* (c1 - m1) .* image)/si
  m01 = sum((c0 - m0) .* (c1 - m1) .* image)/si
  return [ m0, m1 ], [ m00 m01; m01 m11 ]
end

function deskew(matrix)
  output = zeros(size(matrix))
  for j = 1:size(matrix, 2)
    image   = reshape(matrix[:, j], (28, 28))
    c, v    = moments(image)
    alpha   = v[1, 2] / v[1, 1]
    affine  = [1.0 0.0; alpha 1.0]
    ocenter = [ size(image, 1)/2.0, size(image, 2)/2.0 ]
    offset  = c - (affine * ocenter)
    mat     = interpolation.affine_transform(image, affine, offset = offset, order = 1)
    for i = 1:size(output, 1)
      output[i, j] = mat[i]
    end
  end

  return output
end

@info log "reading training data"
const train_raw, train_truth = traindata()
const test_raw, test_truth   = testdata()
const classes                = Dict{Float64, Int32}()
const train                  = deskew(train_raw) / 255.0 #hcat([ deskew(train_raw[:, i]) for i = 1:size(train_raw, 2) ]...)
const test                   = deskew(test_raw) / 255.0 #hcat([ deskew(test_raw[:, i]) for i = 1:size(test_raw, 2) ]...) #test_raw / 255.0

# ----------------------------------------------------------------------------------------------------------------
# Testing support
# ----------------------------------------------------------------------------------------------------------------
type Test
  name :: String
  trainer :: Function
  expected :: Float64
end

i = 1
for t in test_truth
  if !(t in keys(classes))
    classes[t] = i
    i += 1
  end
end
@info log "truth set: $classes"

# ----------------------------------------------------------------------------------------------------------------
# Online learner test
# ----------------------------------------------------------------------------------------------------------------
tests = [ 
  # Test("perceptron",           (init) -> train_perceptron(EachCol(train), train_truth, init; iterations = 50, average = false),  12.74),
  # Test("averaged perceptron",  (init) -> train_perceptron(EachCol(train), train_truth, init; iterations = 50, average = true),   12.69),
  # Test("1-best MIRA",          (init) -> train_mira(EachCol(train), train_truth, init; k = 1, iterations = 30, average = false), 14.26),
  # Test("averaged 1-best MIRA", (init) -> train_mira(EachCol(train), train_truth, init; k = 1, iterations = 30, average = true),   8.23),
  # Test("5-best MIRA",          (init) -> train_mira(EachCol(train), train_truth, init; k = 5, iterations = 30, average = false), 11.46),
  # Test("averaged 5-best MIRA", (init) -> train_mira(EachCol(train), train_truth, init; k = 5, iterations = 30, average = true),   7.72),
  # Test("linear SVM",           (init) -> train_svm(EachCol(train), train_truth; C = 1.0, iterations = 100),                      12.86),
  # Test("linear libSVM",        (init) -> train_libsvm(EachCol(train[:, 1:20000]), train_truth[1:20000], 
  #                                                     C = 1.0, cache_size = 250.0, eps = 0.001, shrinking = true),                8.85),
]
  
for t in tests
  init  = LinearModel(classes, length(train[:, 1]))
  @timer log "training $(t.name) model" model = t.trainer(init)
  res = test_classification(model, EachCol(test), test_truth) * 100.0
  @info log  @sprintf("%s test set error rate: %7.3f%%", t.name, res)
  @test abs(res - t.expected) < 0.001
end

# baseline test with degree-3 RBF kernel
@timer log "libsvm direct" model = svmtrain(train_truth, train, C = 10.0, verbose = true, shrinking = true)
(predicted_labels, decision_values) = svmpredict(model, test);
@info log "test svm direct: $((1.0 - mean(predicted_labels .== test_truth))*100.0)" 

# setup models, train and evaluate
# @timer log "training linear (julia-implementation) SVM model" model = train_svm(EachCol(train), train_truth; C = 0.1, iterations = 100)
# @info log @sprintf("SVM (julia-implementation) test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

# init  = LinearModel(classes, length(train[:, 1]))
# @timer log "training 1-best MIRA model" model = train_mira(EachCol(train), train_truth, init; k = 1, iterations = 30, average = false)
# @info log @sprintf("1-best MIRA test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

# init  = LinearModel(classes, length(train[:, 1]))
# @timer log "training averaged 1-best MIRA model" model = train_mira(EachCol(train), train_truth, init; k = 1, iterations = 30, average = true, C = 0.1)
# @info log @sprintf("averaged 1-best MIRA test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

# init  = LinearModel(classes, length(train[:, 1]))
# @timer log "training 5-best MIRA model" model = train_mira(EachCol(train), train_truth, init; k = 5, iterations = 30, average = false)
# @info log @sprintf("5-best MIRA test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

# init  = LinearModel(classes, length(train[:, 1]))
# @timer log "training averaged 5-best MIRA model" model = train_mira(EachCol(train), train_truth, init; k = 5, iterations = 30, average = true, C = 0.1)
# @info log @sprintf("averaged 5-best MIRA test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

# @timer log "training linear SVM (libsvm) model" model = train_libsvm(EachCol(train[:, 1:20000]), train_truth[1:20000]; 
#                                                                      C = 1.0, cache_size = 250.0, eps = 0.001, shrinking = true)
# @info log @sprintf("SVM (libsvm) test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

# init  = LinearModel(classes, length(train[:, 1]))
# @timer log "training perceptron model" model = train_perceptron(EachCol(train), train_truth, init; iterations = 50, average = false)
# @info log @sprintf("perceptron test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

# init  = LinearModel(classes, length(train[:, 1]))
# @timer log "training averaged perceptron model" model = train_perceptron(EachCol(train), train_truth, init; iterations = 50, average = true)
# @info log @sprintf("averaged perceptron test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)
