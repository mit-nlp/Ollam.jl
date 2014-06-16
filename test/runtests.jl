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
using MNIST, Stage, SVM
import Base: length, start, done, next

# Helpers
log = Log(STDERR)

immutable EachCol{T}
  matrix :: Array{T, 2}
end
length(e :: EachCol)      = size(e.matrix, 2)
start(e :: EachCol)       = 1
next(e :: EachCol, state) = (e.matrix[:, state], state + 1)
done(e :: EachCol, state) = state > length(e) ? true : false

# setup MNIST task
@info log "reading training data"
const train, train_truth = traindata()
const test, test_truth   = testdata()
const classes            = Dict{Integer, Int32}()
i                        = 1
for t in test_truth
  if !(t in keys(classes))
    classes[t] = i
    i += 1
  end
end
@info log "truth set: $classes"

# setup models, train and evaluate
# @timer log "training linear (julia-implementation) SVM model" model = train_svm2(EachCol(train / 255.0), train_truth; C = 0.01, iterations = 100)
# @info log @sprintf("SVM (julia-implementation) test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

# init  = LinearModel(classes, length(train[:, 1]))
# @timer log "training MIRA model" model = train_mira(EachCol(train / 255.0), train_truth, init; iterations = 30, average = false)
# @info log @sprintf("MIRA test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

# init  = LinearModel(classes, length(train[:, 1]))
# @timer log "training averaged MIRA model" model = train_mira(EachCol(train), train_truth, init; iterations = 30, average = true, C = 0.1)
# @info log @sprintf("averaged MIRA test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

@timer log "training linear SVM (libsvm) model" model = train_svm(EachCol(train[:, 1:20000] / 255.0), train_truth[1:20000]; C = 1.0, cache_size = 250.0, eps = 0.2, shrinking = false, verbose = false)
@info log @sprintf("SVM (libsvm) test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

init  = LinearModel(classes, length(train[:, 1]))
@timer log "training perceptron model" model = train_perceptron(EachCol(train), train_truth, init; iterations = 50, average = false)
@info log @sprintf("perceptron test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

init  = LinearModel(classes, length(train[:, 1]))
@timer log "training averaged perceptron model" model = train_perceptron(EachCol(train), train_truth, init; iterations = 50, average = true)
@info log @sprintf("averaged perceptron test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)
