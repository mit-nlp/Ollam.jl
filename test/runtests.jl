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
using Ollam
using MNIST, Stage, LIBSVM, GZip
import Base: length, start, done, next
using Base.Test

# ----------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------
immutable EachCol{T}
  matrix :: Array{T, 2}
end
length(e :: EachCol)      = size(e.matrix, 2)
start(e :: EachCol)       = 1
next(e :: EachCol, state) = (e.matrix[:, state], state + 1)
done(e :: EachCol, state) = state > length(e) ? true : false

type T
  name     :: String
  trainer  :: Function
  expected :: Float64
  testset  :: Matrix
end

# ----------------------------------------------------------------------------------------------------------------
# test hildreth
# ----------------------------------------------------------------------------------------------------------------
h = setup_hildreth(C = Inf, k = 2)
@expect abs(hildreth((SparseMatrixCSC)[ spzeros(1235, 1),
                sparsevec([470=>-1.0, 1070=>-1.0, 1231=>-1.0, 1232=>-1.0, 1233=>-1.0, 1234=>-1.0, 1235=>-1.0, 496=>1.0, 
                           1058=>1.0, 1226=>1.0, 1227=>1.0, 1228=>1.0, 1229=>1.0, 1230=>1.0]) ], [ 0.0, 0.9733606705258293 ], h)[2] - 0.069526) < 0.00001

# ----------------------------------------------------------------------------------------------------------------
# test lazy maps
# ----------------------------------------------------------------------------------------------------------------
xxx = [1, 2, 3]
for i in lazy_map(f -> f+1, xxx)
  @expect i == xxx[i-1] + 1
end

# ----------------------------------------------------------------------------------------------------------------
# Setup buzz task
# ----------------------------------------------------------------------------------------------------------------
# @info "reading Tom's Hardware train/test data"

# f = gzopen("TomsHardware.data.gz")
# srand(0)
# const toms_data    = hcat([ ((randn()+1.0) + (i % 100)) for i = 1:5000, j = 1:2 ], [ i % 100 for i = 1:5000 ])' # readcsv(f)'
# #const toms_data    = readcsv(f)'
# close(f)

# const rall         = toms_data[1:end-1, :]
# const rall_truth   = vec(toms_data[end, :])
# const m            = mean(rall, 2)
# const s            = std(rall, 2)
# const rtrain       = [ rall[i, j] - m[i] / s[i] for i = 1:size(rall, 1), j = 1:int(size(rall, 2)*0.8) ] #rall[:, 1:int(size(rall, 2)*0.8)]
# const rtrain_truth = rall_truth[1:int(size(rall, 2)*0.8)]
# const rtest        = [ rall[i, j] - m[i] / s[i] for i = 1:size(rall, 1), j = (int(size(rall, 2)*0.8)+1):size(rall, 2) ] #rall[:, 1:int(size(rall, 2)*0.8)]
# const rtest_truth  = rall_truth[int(size(rall, 2)*0.8)+1:end]
# @debug "size of train: $(size(rtrain)), test: $(size(rtest))"

# # ----------------------------------------------------------------------------------------------------------------
# # Regression tests
# # ----------------------------------------------------------------------------------------------------------------
# regression_tests = [ 
#   # T("perceptron",           (init) -> regress_perceptron(EachCol(rtrain), rtrain_truth, init; iterations = 300, average = false, learn_rate = 0.0000001, C = 0.1),  12.74, rtest),
#   # T("averaged perceptron",  (init) -> train_perceptron(EachCol(train), train_truth, init; iterations = 50, average = true),   12.69, test),
#   T("1-best MIRA",          (init) -> regress_mira(EachCol(rtrain), rtrain_truth, init; iterations = 50, average = true, C = 0.1, min_loss = 0.05), 14.26, rtest),
#   # T("averaged 1-best MIRA", (init) -> train_mira(EachCol(train), train_truth, init; k = 1, iterations = 30, average = true),   8.23, test),
#   # T("5-best MIRA",          (init) -> train_mira(EachCol(train), train_truth, init; k = 5, iterations = 30, average = false), 11.46, test),
#   # T("averaged 5-best MIRA", (init) -> train_mira(EachCol(train), train_truth, init; k = 5, iterations = 30, average = true),   7.72, test),
#   # T("linear SVM",           (init) -> train_svm(EachCol(train), train_truth; C = 0.001, iterations = 100),                     8.73, test),
#   # T("linear libSVM",        (init) -> train_libsvm(EachCol(train[:, 1:20000]), train_truth[1:20000], 
#   #                                                     C = 1.0, cache_size = 250.0, eps = 0.001, shrinking = true),             8.85, test),
# ]

# # Ollam tests  
# for t in regression_tests
#   init  = RegressionModel(length(rtrain[:, 1]))
#   @timer "training $(t.name) model" model = t.trainer(init)
#   res = sqrt(test_regression(model, EachCol(t.testset), rtest_truth))
#   @info @sprintf("%s test set error rate: %7.3f", t.name, res)
#   @expect abs(res - t.expected) < 0.001
# end

# ----------------------------------------------------------------------------------------------------------------
# setup MNIST task
# ----------------------------------------------------------------------------------------------------------------
@info "reading MNIST train/test data"
const train_raw, train_truth    = traindata()
const test_raw, test_truth      = testdata()
const classes                   = Dict{Float64, Int32}()
const train                     = train_raw / 255.0
const test                      = test_raw / 255.0
# const train_deskew, test_deskew = begin 
#   f   = gzopen("mnist-deslant.jd.gz")
#   trd = deserialize(f)
#   trt = deserialize(f)
#   tsd = deserialize(f)
#   tst = deserialize(f)
#   trd / 255.0, tsd / 255.0
# end

# ----------------------------------------------------------------------------------------------------------------
# Testing support
# ----------------------------------------------------------------------------------------------------------------
i = 1
for t in test_truth
  if !(t in keys(classes))
    classes[t] = i
    i += 1
  end
end
@info "truth set: $classes"

# ----------------------------------------------------------------------------------------------------------------
# Classification tests
# ----------------------------------------------------------------------------------------------------------------
classification_tests = [ 
  # deskew tests
  # T("5-best MIRA [deskew]",          (init) -> train_mira(EachCol(train_deskew), train_truth, init; k = 5, iterations = 30, average = false), 6.73, test_deskew),
  # T("averaged 5-best MIRA [deskew]", (init) -> train_mira(EachCol(train_deskew), train_truth, init; k = 5, iterations = 30, average = true),  4.54, test_deskew),
  # T("linear SVM [deskew]",           (init) -> train_svm(EachCol(train_deskew), train_truth; C = 0.001, iterations = 100),                    5.40, test_deskew),
  # T("linear libSVM",  (init) -> train_libsvm(EachCol(train_deskew), train_truth, C = 1.0, cache_size = 250.0, eps = 0.001, shrinking = true), 7.83, test_deskew),

  # baselines
  T("perceptron",           (init) -> train_perceptron(EachCol(train), train_truth, init; iterations = 50, average = false),  12.74, test),
  T("averaged perceptron",  (init) -> train_perceptron(EachCol(train), train_truth, init; iterations = 50, average = true),   12.69, test),
  T("1-best MIRA",          (init) -> train_mira(EachCol(train), train_truth, init; k = 1, iterations = 30, average = false), 14.26, test),
  T("averaged 1-best MIRA", (init) -> train_mira(EachCol(train), train_truth, init; k = 1, iterations = 30, average = true),   8.23, test),
  T("5-best MIRA",          (init) -> train_mira(EachCol(train), train_truth, init; k = 5, iterations = 30, average = false), 11.46, test),
  T("averaged 5-best MIRA", (init) -> train_mira(EachCol(train), train_truth, init; k = 5, iterations = 30, average = true),   7.72, test),
  T("linear SVM",           (init) -> train_svm(EachCol(train), train_truth; C = 0.001, iterations = 100),                     8.73, test),
  T("linear libSVM",        (init) -> train_libsvm(EachCol(train[:, 1:20000]), train_truth[1:20000], 
                                                      C = 1.0, cache_size = 250.0, eps = 0.001, shrinking = true),             8.85, test),
]

# Ollam tests  
for t in classification_tests
  init  = LinearModel(classes, length(train[:, 1]))
  @timer "training $(t.name) model" model = t.trainer(init)
  res = test_classification(model, EachCol(t.testset), test_truth) * 100.0
  @info @sprintf("%s test set error rate: %7.3f%%", t.name, res)
  @expect abs(res - t.expected) < 0.001
end

# baseline test with degree-3 RBF kernel
@timer "libsvm 2nd-order rbf direct" model = svmtrain(train_truth, train, C = 10.0, verbose = true, shrinking = true)
(predicted_labels, decision_values) = svmpredict(model, test)
@info "test libsvm 2nd-order rbf direct: $((1.0 - mean(predicted_labels .== test_truth))*100.0)" 
@expect abs(((1.0 - mean(predicted_labels .== test_truth))*100.0) - 3.87) < 0.001

